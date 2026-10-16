"""Source-backed candidate endpoints and bounded distance-only journey estimates."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from opencc import OpenCC
from pyproj import Transformer
from dotenv import dotenv_values
from .fare_calculator import load_distance_rule, calculate_distance

_CC=OpenCC('t2s')
_TRANSFORM=Transformer.from_crs(2326,4326,always_xy=True,allow_ballpark=False)


def norm(value):
    return re.sub(r'[^\w]', '', _CC.convert(unicodedata.normalize('NFKC',value)).casefold())


def metres(a,b):
    lon1,lat1,lon2,lat2=map(math.radians,[*a,*b])
    x=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371000*2*math.asin(min(1,math.sqrt(x)))


def coordinates(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError('Expected [longitude,latitude].')
    lon,lat=value
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):
        raise ValueError('Coordinates must be finite numbers.')
    if not (-180<=lon<=180 and -90<=lat<=90): raise ValueError('Invalid WGS84 coordinates.')
    return list(value)


def select_anchor(query,rows,kind='auto'):
    from .place_resolution import select_anchor as select
    return select(query,rows,kind)


def parse_osm(raw):
    if raw.lstrip().startswith(b'{'):
        elements=json.loads(raw).get('elements',[])
        nodes={str(n['id']):{'coordinate':[n['lon'],n['lat']],'tags':n.get('tags',{})}
               for n in elements if n.get('type')=='node'}
        ways=[{'id':str(w['id']),'tags':w.get('tags',{}),'node_ids':[str(n) for n in w.get('nodes',[]) if str(n) in nodes]}
              for w in elements if w.get('type')=='way' and 'highway' in w.get('tags',{})]
        return nodes,ways
    tree=ET.fromstring(raw)
    nodes={n.attrib['id']:{'coordinate':[float(n.attrib['lon']),float(n.attrib['lat'])],
        'tags':{t.attrib['k']:t.attrib['v'] for t in n.findall('tag')}} for n in tree.findall('node')}
    ways=[]
    for w in tree.findall('way'):
        tags={t.attrib['k']:t.attrib['v'] for t in w.findall('tag')}
        ids=[n.attrib['ref'] for n in w.findall('nd')]
        if 'highway' in tags:
            ways.append({'id':w.attrib['id'],'tags':tags,'node_ids':[i for i in ids if i in nodes]})
    return nodes,ways


def forbidden(tags):
    if tags.get('bridge') not in {None,'no'} or tags.get('tunnel') not in {None,'no'}: return 'grade_separated'
    if tags.get('layer','0') not in {'0','0.0'}: return 'grade_separated'
    if tags.get('highway') in {'footway','pedestrian','steps','cycleway','path','motorway','motorway_link','trunk','trunk_link','busway','bus_guideway','platform'}: return 'not_pickup_access_road'
    if tags.get('area')=='yes': return 'area_not_road'
    for key in ['access','vehicle','motor_vehicle','motorcar']:
        if tags.get(key) in {'no','private','permit','customers','destination'}: return 'restricted_or_conditional_access'
        if key+':conditional' in tags: return 'conditional_access'
    if tags.get('bus') in {'designated','only'} or tags.get('psv')=='designated': return 'bus_only'
    return None


def infer_kind(anchor,nodes,stands):
    if anchor['kind']!='auto': return anchor['kind']
    q=norm(anchor['query'])
    districts={norm(s.get('district_tc','')) for s in stands}
    if q in districts: return 'area'
    for n in nodes.values():
        tags=n['tags']
        if tags.get('place') and any(norm(tags.get(k,''))==q for k in ['name','name:zh','name:en']): return 'area'
    return 'place'


def candidates_for(anchor,nodes,ways,stands,config):
    radius=config['search_radius_m']
    adjacency={node:[] for node in nodes}
    rejected=[]
    for way in ways:
        reason=forbidden(way['tags'])
        if reason:
            if any(metres(anchor['coordinate'],nodes[n]['coordinate'])<=radius for n in way['node_ids']):
                rejected.append({'source_id':'osm-way-'+way['id'],'reason':reason})
            continue
        for node in way['node_ids']: adjacency[node].append(way)
    pool=[]
    for stand in stands:
        if '市區' not in stand.get('status_tc','') or '過海' in stand.get('status_tc',''): continue
        coord=[stand['longitude'],stand['latitude']]
        offset=metres(anchor['coordinate'],coord)
        if offset<=radius:
            pool.append({'id':'stand-'+str(stand['stand_id']),'name':stand['location_tc']+'（官方市区的士站）',
                'coordinate':coord,'kind':'official_taxi_stand','priority':0,'road_key':'stand-'+str(stand['stand_id']),
                'anchor_offset_m':round(offset,1),'source_url':'https://data.gov.hk/en-data/dataset/hk-td-tis_37-taxi-stands',
                'boarding_evidence':'official_stand_dataset','curb_permission_verified':False})
    for node,linked in adjacency.items():
        if not linked: continue
        n=nodes[node]
        offset=metres(anchor['coordinate'],n['coordinate'])
        if offset>radius: continue
        tags=n['tags']
        if tags.get('barrier') in {'gate','lift_gate','bollard','block','cycle_barrier','chain','bus_trap'} or forbidden(tags):
            rejected.append({'source_id':'osm-node-'+node,'reason':'barrier_or_restriction'})
            continue
        named=[w for w in linked if w['tags'].get('name') and w['tags'].get('highway')!='service']
        service=[w for w in linked if w['tags'].get('highway')=='service']
        vehicle_entrance=tags.get('entrance') in {'yes','main','service'} and any(tags.get(k) in {'yes','designated'} for k in ['motor_vehicle','motorcar','vehicle'])
        if vehicle_entrance:
            typ,priority='vehicle_entrance',1
        elif named and service:
            typ,priority='service_road_junction',2
        elif named:
            typ,priority='ground_road_reference',3
        else:
            # Do not generate interior driveways, unnamed paths, or indoor doors.
            continue
        road=(named or linked)[0]
        name=road['tags'].get('name','车辆入口')
        pool.append({'id':'osm-node-'+node,'name':name+(' / 车行通道接入点' if typ=='service_road_junction' else ' / 路线参考点'),
            'coordinate':n['coordinate'],'kind':typ,'priority':priority,
            'road_key':norm(name) if named else 'osm-way-'+road['id'],
            'anchor_offset_m':round(offset,1),'source_url':'https://www.openstreetmap.org/node/'+node,
            'osm_way_ids':[w['id'] for w in linked], 'curb_permission_verified':False})
    # Remove same-road clusters, preserving stronger evidence before proximity.
    ordered=sorted(pool,key=lambda c:(c['priority'],c['anchor_offset_m'],c['id']))
    unique=[]
    for c in ordered:
        duplicate=next((u for u in unique if u['id']==c['id'] or
            (u['road_key']==c['road_key'] and metres(u['coordinate'],c['coordinate'])<=config['dedup_distance_m'])),None)
        if duplicate:
            rejected.append({'source_id':c['id'],'reason':'same_road_duplicate','retained':duplicate['id']})
        else: unique.append(c)
    # Preserve one best candidate of each evidence kind, then fill to the approved cap.
    selected=[]
    for typ in ['official_taxi_stand','vehicle_entrance','service_road_junction','ground_road_reference']:
        candidate=next((c for c in unique if c['kind']==typ),None)
        if candidate is not None and len(selected)<config['max_candidates']: selected.append(candidate)
    for c in unique:
        if len(selected)>=config['max_candidates']: break
        if c not in selected: selected.append(c)
    return {'anchor':dict(anchor,kind=infer_kind(anchor,nodes,stands)), 'candidates':selected,
            'rejected':rejected,'pool_count':len(pool), 'retained_count':len(selected),
            'note':'Nearby candidates are alternatives to choose, not confirmed doors; offsets are straight-line, not walking distances.'}


def route_pairs(origin,destination):
    a,b=origin['candidates'],destination['candidates']
    if not a or not b: return []
    if origin['anchor']['kind']=='area' and destination['anchor']['kind']=='area': return [(a[0],b[0])]
    # Fixed first destination is explicitly displayed as an assumption, not silently selected.
    if len(a)>1: return [(c,b[0]) for c in a[:3]]
    return [(a[0],c) for c in b[:3]]


class MapError(RuntimeError): pass
class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None


class MapClient:
    def __init__(self,root,config,budget=None):
        self.root=Path(root)
        self.config=config
        self.limit=config['max_routes_per_request'] if budget is None else budget
        self.ors_calls=0
        self.public_calls=0

    def _request(self,url,body=None,key=None):
        headers={'User-Agent':'hk-taxi-assistant-map-probe/1','Accept':'application/json'}
        if body is not None:
            headers.update({'Content-Type':'application/json','Accept':'application/geo+json'})
        if key: headers['Authorization']=key
        try:
            req=Request(url,data=json.dumps(body).encode() if body is not None else None,headers=headers,
                        method='POST' if body is not None else 'GET')
            with build_opener(NoRedirect()).open(req,timeout=self.config['timeout_seconds']) as response:
                return response.read()
        except HTTPError as exc: raise MapError(f'HTTP {exc.code}; no automatic retry.') from None
        except (URLError,OSError,TimeoutError): raise MapError('Network/TLS error or timeout; no automatic retry.') from None

    def locations(self,query):
        self.public_calls+=1
        return json.loads(self._request('https://www.map.gov.hk/gs/api/v1.0.0/locationSearch?'+urlencode({'q':query})))

    def geometry(self,coordinate):
        lon,lat=coordinates(coordinate)
        delta_lat=self.config['search_radius_m']/111320
        delta_lon=delta_lat/math.cos(math.radians(lat))
        bbox=f'{lon-delta_lon},{lat-delta_lat},{lon+delta_lon},{lat+delta_lat}'
        self.public_calls+=1
        raw=self._request('https://www.openstreetmap.org/api/0.6/map?'+urlencode({'bbox':bbox}))
        return parse_osm(raw)

    def route(self,a,b,via=None):
        if self.ors_calls>=self.limit: raise MapError('ORS call budget exhausted.')
        for endpoint in [a]+list(via or [])+[b]:
            lon,lat=coordinates(endpoint['coordinate'])
            if not (113.8<lon<114.5 and 22.1<lat<22.6): raise MapError('Endpoint outside the supported Hong Kong review envelope.')
        local=dotenv_values(self.root/'.env',encoding='utf-8-sig',interpolate=False)
        key=str(os.environ.get('ORS_API_KEY',local.get('ORS_API_KEY')) or '').strip()
        if not key: raise MapError('Missing ORS_API_KEY; Key not displayed.')
        payload={'coordinates':[p['coordinate'] for p in [a]+list(via or [])+[b]],'extra_info':['tollways'],
                 'options':{'avoid_features':['ferries'],'avoid_borders':'all'}}
        self.ors_calls+=1
        data=json.loads(self._request('https://api.openrouteservice.org/v2/directions/driving-car/geojson',payload,key))
        return inspect_route(data,a,b)


def inspect_route(data,a,b):
    try:
        feature=data['features'][0]
        props=feature['properties']
        distance=props['summary']['distance']
        duration=props['summary']['duration']
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in [distance,duration]): raise ValueError()
        geom=[coordinates(p[:2]) for p in feature['geometry']['coordinates']]
        first,last=coordinates(geom[0]),coordinates(geom[-1])
        steps=[s for segment in props.get('segments',[]) for s in segment.get('steps',[])]
    except (KeyError,IndexError,TypeError,ValueError): raise MapError('Invalid or empty route response.') from None
    named=[s.get('name','') for s in steps if s.get('name','') not in {'','-'}]
    problems=[]
    # Conservative review signal, not an assertion that every flyover in a route is invalid.
    if named and any(t in (named[0]+' '+named[-1]).lower() for t in ['flyover','天橋','天桥']):
        problems.append('endpoint_named_road_may_be_flyover')
    warnings=props.get('warnings',[])
    advisory=[w for w in warnings if w.get('code')==1 and 'restrictions on some roads' in w.get('message','')]
    if any(w not in advisory for w in warnings): problems.append('provider_warning_requires_review')
    return {'origin_id':a['id'],'destination_id':b['id'],'origin_name':a['name'],'destination_name':b['name'],
      'origin_coordinate':a['coordinate'],'destination_coordinate':b['coordinate'],
      'distance_m':distance,'duration_s':duration,'routed_endpoints':[first,last],
      'geometry_coordinates':geom,'extras':props.get('extras',{}),
      'snap_offsets_m':[round(metres(a['coordinate'],first),1),round(metres(b['coordinate'],last),1)],
      'road_names':named,'warnings':warnings,'route_advisories':advisory,'review_flags':problems,
      'status':'needs_route_review' if problems else 'provisional_distance_estimate',
      'attribution':data.get('metadata',{}).get('attribution'),'taxi_stopping_permission_verified':False}


def estimate_pairs(client,origin,destination,rule,as_of=None,pairs=None):
    results=[]
    for a,b in route_pairs(origin,destination) if pairs is None else pairs:
        if len(results)>=client.config['max_routes_per_request']: break
        try:
            r=client.route(a,b)
            if not r['review_flags']:
                r['estimate']=calculate_distance(r['distance_m'],rule,as_of)
            results.append(r)
        except MapError as exc:
            results.append({'origin_id':a['id'],'destination_id':b['id'],'status':'tool_error','error':str(exc)})
            break
    return results


def make_answer(origin,destination,results,selection_confirmed=False):
    usable=[r for r in results if r.get('estimate')]
    broad=origin['anchor']['kind']=='area' or destination['anchor']['kind']=='area'
    lines=[f"起点：{origin['anchor']['query']}；终点：{destination['anchor']['query']}。"]
    if broad:
        lines.append('粗略估算：以以下明确列出的代表地点附近为例，不代表整个区域的最低／最高费用。')
    else:
        lines.append('以下按附近可选择的路线接入点分别估算；这些点不全是已确认的上落客点。')
    used_destinations={r['destination_id'] for r in results}
    other_destinations=[c for c in destination['candidates'] if c['id'] not in used_destinations]
    if results and len(used_destinations)==1:
        lines.append('本轮终点假设：'+results[0].get('destination_name',next((c['name'] for c in destination['candidates'] if c['id'] in used_destinations),'所列候选点'))+'。')
    if other_destinations:
        lines.append('也可改选终点：'+'；'.join(c['id']+' '+c['name'] for c in other_destinations)+'。')
    for r in results:
        if r.get('estimate'):
            lines.append(f"{r['origin_id']} {r['origin_name']} → {r['destination_id']} {r['destination_name']}：约{r['distance_m']/1000:.2f}公里，距离计程约HK${r['estimate']['fare_hkd']}。")
        else:
            lines.append(f"{r['origin_id']} → {r['destination_id']}：路线／工具待核验，本项暂不报价。")
    if len(usable)>1:
        prices=[r['estimate']['fare_cent'] for r in usable]
        lines.append(f'这些已算候选的距离计程范围为 HK${min(prices)/100:.2f}–{max(prices)/100:.2f}，不是实际总价区间。')
    if not usable: lines.append('目前没有通过路线检查的估算，请补充具体建筑／入口或稍后再试；不会猜测距离。')
    lines.append('均未包含咪表计时、道路收费、行李及其他附加费用；实际以咪表及适用收费为准。')
    if usable:
        lines.append('依据：ORS道路距离；运输署收费表 '+usable[0]['estimate'].get('source_url','https://www.td.gov.hk/en/transport_in_hong_kong/public_transport/taxi/taxi_fare_of_hong_kong/'))
    if selection_confirmed:
        lines.append('已按所选候选显示本会话的路线结果；可以继续补充地址细化，尚非实时重新报价。')
    else:
        lines.append('你有更具体的街口、建筑或车站出口吗？可以据此细化。' if broad else '你准备从哪个候选位置上车、到哪个位置下车？')
    return '\n'.join(lines)


def discover(client,query,kind,stands):
    from .place_resolution import discover_place
    return discover_place(client,query,kind,stands)
