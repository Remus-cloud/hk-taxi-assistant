"""SQL-backed taxi surcharges, identified from toll-marked route geometry."""
from pathlib import Path
import json,math,hashlib
import duckdb
from pyproj import Transformer

_XY=Transformer.from_crs(4326,2326,always_xy=True)

def initialize(root):
    root=Path(root);path=root/'configs/harbour_tolls_v1.json'
    raw=path.read_bytes();cfg=json.loads(raw)
    if cfg.get('approval_status')!='approved':raise ValueError('Unapproved harbour toll rules')
    geometry=(root/cfg['geometry_file']).read_text(encoding='utf-8')
    sha=hashlib.sha256(raw+geometry.encode()).hexdigest()
    db=root/'data/processed/taxi_surcharges_v1.duckdb';db.parent.mkdir(parents=True,exist_ok=True)
    with duckdb.connect(str(db)) as con:
        con.execute('BEGIN')
        try:
            con.execute('CREATE TABLE IF NOT EXISTS surcharge_versions(version VARCHAR PRIMARY KEY, sha256 VARCHAR, geometry_json VARCHAR, config_json VARCHAR)')
            con.execute('CREATE TABLE IF NOT EXISTS harbour_tolls(version VARCHAR,tunnel_id VARCHAR,name VARCHAR,toll_cent INTEGER,return_cent INTEGER,source_url VARCHAR,evidence_text VARCHAR,PRIMARY KEY(version,tunnel_id))')
            old=con.execute('SELECT sha256 FROM surcharge_versions WHERE version=?',[cfg['version']]).fetchone()
            if old and old[0]!=sha:raise ValueError('Surcharge version changed; create a new version')
            if not old:
                con.execute('INSERT INTO surcharge_versions VALUES (?,?,?,?)',[cfg['version'],sha,geometry,raw.decode()])
                for rule in cfg['rules']:
                    if any(type(rule[k]) is not int or rule[k]<0 for k in ['toll_cent','return_cent']):raise ValueError('Surcharge amounts must be nonnegative integer cents')
                    con.execute('INSERT INTO harbour_tolls VALUES (?,?,?,?,?,?,?)',[cfg['version'],rule['tunnel_id'],rule['name'],rule['toll_cent'],rule['return_cent'],rule['source_url'],rule['evidence_text']])
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK');raise
    return db

def load(root):
    root=Path(root)
    cfg=json.loads((root/'configs/harbour_tolls_v1.json').read_text(encoding='utf-8'))
    with duckdb.connect(str(root/'data/processed/taxi_surcharges_v1.duckdb'),read_only=True) as con:
        stored=con.execute('SELECT geometry_json,config_json FROM surcharge_versions WHERE version=?',[cfg['version']]).fetchone()
        if not stored:raise ValueError('Missing surcharge rule version')
        columns=['tunnel_id','name','toll_cent','return_cent','source_url','evidence_text']
        rows=con.execute('SELECT '+','.join(columns)+' FROM harbour_tolls WHERE version=?',[cfg['version']]).fetchall()
    return json.loads(stored[1]),json.loads(stored[0]),{row[0]:dict(zip(columns,row)) for row in rows}

def point_segment(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
    t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/den)) if den else 0
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)

def identify(route,geometry,config):
    coords=route.get('geometry_coordinates')
    tolls=route.get('extras',{}).get('tollways',{}).get('values')
    if not coords or tolls is None:return {'status':'unavailable','crossings':[],'reason':'route_geometry_or_toll_flags_missing'}
    xy=[_XY.transform(*p[:2]) for p in coords]
    refs={}
    for tunnel in geometry['tunnels']:
        seg=[]
        for way in tunnel['ways']:
            points=[_XY.transform(*p[:2]) for p in way['coordinates']]
            seg.extend(zip(points,points[1:]))
        refs[tunnel['tunnel_id']]=seg
    tolerance=config['match_tolerance_m'];minimum=config['minimum_match_m']
    runs=[];current=None;distance=0;geometric_without_toll=False
    for i,(a,b) in enumerate(zip(xy,xy[1:])):
        length=math.dist(a,b)
        if not length:continue
        charged=any(start<=i<end and value==1 for start,end,value in tolls)
        n=max(1,math.ceil(length/tolerance));step=length/n
        for j in range(n):
            t=(j+.5)/n;p=(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))
            found=next((key for key,segs in refs.items() if any(point_segment(p,x,y)<=tolerance for x,y in segs)),None)
            if found and not charged:geometric_without_toll=True
            key=found if charged else None
            if key:
                if current and current['tunnel_id']==key:current['length_m']+=step;current['end_m']=distance+step
                else:
                    if current:runs.append(current)
                    current={'tunnel_id':key,'start_m':distance,'end_m':distance+step,'length_m':step}
            elif current:runs.append(current);current=None
            distance+=step
    if current:runs.append(current)
    valid=[v for v in runs if v['length_m']>=minimum]
    # A short map-data gap must not turn one crossing into two tolls.
    merged=[]
    for v in valid:
        if merged and v['tunnel_id']==merged[-1]['tunnel_id'] and v['start_m']-merged[-1]['end_m']<minimum:
            merged[-1]['end_m']=v['end_m'];merged[-1]['length_m']+=v['length_m']
        else:merged.append(dict(v))
    has_charge=any(v==1 for a,b,v in tolls)
    status='identified' if merged else ('unresolved_toll' if has_charge or geometric_without_toll else 'no_toll_detected')
    return {'status':status,'crossings':merged,'method':'ORS tollways + sourced OSM tunnel geometry',
        'match_tolerance_m':tolerance,'minimum_match_m':minimum}

def quote(root,route,state):
    cfg,geometry,rules=load(root)
    enabled=json.loads((Path(root)/'configs/journey_multi_v1.json').read_text(encoding='utf-8')).get('toll_amount_enabled',False)
    detection=identify(route,geometry,cfg) if enabled else {'status':'disabled','crossings':[]}
    base=route['estimate']['fare_cent'];crossings=detection['crossings']
    result={'rule_version':cfg['version'],'detection':detection,'distance_cent':base,'outbound_cent':0,'return_cent':0,
        'subtotal_cent':base,'boarding_assumption':state.get('conditions',{}).get('boarding_type') or 'unspecified',
        'status':detection['status'],'exemption':None,'source_url':cfg['source_url'],'crossings':[]}
    if not crossings:return result
    for crossing in crossings:
        rule=rules[crossing['tunnel_id']]
        result['outbound_cent']+=rule['toll_cent']
        result['crossings'].append(dict(crossing,name=rule['name'],toll_cent=rule['toll_cent']))
    origin=state.get('origin') or {};selected=origin.get('selected_id')
    candidate=next((c for c in origin.get('candidates',[]) if c['id']==selected),{})
    if result['boarding_assumption']=='cross_harbour_stand' or candidate.get('boarding_type')=='cross_harbour_stand':
        result['exemption']='cross_harbour_stand'
    elif len(crossings)%2==0:
        result['exemption']='same_side_final_destination'
    else:
        result['return_cent']=rules[crossings[-1]['tunnel_id']]['return_cent']
    result['subtotal_cent']=base+result['outbound_cent']+result['return_cent']
    result['status']='harbour_tolls_included'
    return result
