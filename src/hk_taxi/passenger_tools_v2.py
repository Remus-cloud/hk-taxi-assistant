"""Approved deterministic passenger tools. No network or arbitrary SQL."""
from pathlib import Path
from copy import deepcopy
import json,re,hashlib,math
import duckdb
from .runtime_assets import verify_asset
from pyproj import Geod
from .structured_guard import norm,RuleStore,evaluate_baggage

MODES={'street_or_stand','ordinary_phone_booking','fleet_prebooked','unknown'}
KINDS={'ordinary_stand','cross_harbour_stand','fleet_prebooked_pickup','prebooked_pickup','unknown'}
NEW_DEFAULTS=dict(service_mode='unknown',animal_role='unknown',pickup_facility_kind='unknown',bags=None,pet_count=None,fleet_price_mode='unknown',fleet_booking_cent=None,platform_agreed_total_cent=None)

def upgrade_state(state):
    s=deepcopy(state);s['version']='unified-passenger-v2'
    for k,v in NEW_DEFAULTS.items():s['conditions'].setdefault(k,deepcopy(v))
    return s

def cents(value):
    if value is not None and (type(value) is not int or value<0):raise ValueError('Money must be nonnegative integer cents')
    return value

def apply_condition_patch(state,patch):
    """null=no change; cancellation is expressed as an actual mode transition."""
    s=upgrade_state(state);c=s['conditions'];previous=c['service_mode']
    mode=patch.get('service_mode')
    if mode is not None and mode not in MODES:raise ValueError('Unknown service mode')
    if mode not in {None,'unknown'} and mode!=previous:
        for k in ['fleet_booking_cent','platform_agreed_total_cent']:c[k]=None
        c['fleet_price_mode']='unknown';c['pickup_facility_kind']='unknown'
        s.pop('pickup_facilities',None)
    price_mode=patch.get('fleet_price_mode')
    if price_mode not in {None,'unknown',c['fleet_price_mode']}:
        c['fleet_booking_cent']=None;c['platform_agreed_total_cent']=None
    for k in NEW_DEFAULTS:
        v=patch.get(k)
        if v is None or v=='unknown':continue
        if k.endswith('_cent'):cents(v)
        if k=='pet_count' and (type(v) is not int or v<0):raise ValueError('Invalid pet count')
        if k=='pickup_facility_kind' and v not in KINDS:raise ValueError('Invalid facility kind')
        if k=='animal_role' and v not in {'guide_dog','ordinary_pet','unknown'}:raise ValueError('Invalid animal role')
        if k=='bags':
            for item in v:
                if type(item.get('count')) is not int or item['count']<1:raise ValueError('Bag count required')
        c[k]=deepcopy(v)
    # Channel-based guesses cannot restore stale platform prices after cancellation.
    if c['service_mode']!='fleet_prebooked':
        c.update(fleet_booking_cent=None,platform_agreed_total_cent=None,fleet_price_mode='unknown')
    if patch.get('animal_role') not in {None,'unknown',state['conditions'].get('animal_role')} and patch.get('pet_count') is None:c['pet_count']=None
    return s

def fee_components(root,route,state):
    """Known-subtotal, never a fabricated all-inclusive price."""
    c=state['conditions'];t=route.get('toll_quote') or {};distance=route['estimate']['fare_cent']
    result=dict(distance_cent=cents(distance),outbound_toll_cent=None,return_toll_cent=None,luggage_cent=None,
       ordinary_phone_booking_cent=None,animal_cent=None,waiting_cent=None,other_road_cent=None,
       platform_agreed_total_cent=None,fleet_booking_cent=None,not_included=['waiting','other_road_charges'],assumptions=[],evidence_refs=[route['estimate']['source_url']])
    if t.get('status') in {'identified','harbour_tolls_included','no_toll_detected'}:
        result.update(outbound_toll_cent=t.get('outbound_cent',0),return_toll_cent=t.get('return_cent',0))
    else:result['not_included'].append('unconfirmed_harbour_tolls')
    if t.get('source_url'):result['evidence_refs'].append(t['source_url'])
    if t.get('return_cent'):result['assumptions'].append('ordinary_one_way_return_toll')
    if t.get('exemption'):result['assumptions'].append(t['exemption'])
    rules=json.loads((Path(root)/'configs/passenger_tools_v2.json').read_text(encoding='utf-8'))
    # Source hash pins the approved fee table rather than a model-provided amount.
    for ref in rules['fee_evidence']:
        if hashlib.sha256((Path(root)/ref['path']).read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Fee evidence hash mismatch')
    fee=rules['urban_surcharges'];bags=c.get('bags');bag_total=0;unknown=False;bag_audit=[]
    if bags is not None:
        store=RuleStore(root)
        for b in bags:
            outcome=evaluate_baggage(store,location=b.get('location'),total_cm=b.get('total_cm'),item_type=b.get('item_type'),mobility_use=b.get('mobility_use'),light_status=b.get('light_status'))
            bag_audit.append(outcome)
            if outcome['outcome']=='exempt':continue
            if outcome['outcome']=='general_charge_condition_met':bag_total+=fee['luggage_per_item_cent']*b['count']
            else:unknown=True
        if not unknown or bag_total>0:result['luggage_cent']=bag_total
        if unknown:
            result['not_included'].append('unresolved_luggage')
            result['assumptions'].append('Some luggage conditions unresolved; a 140cm boundary is not a complete definition of light baggage')
    elif c.get('luggage')=='none':result['luggage_cent']=0
    if result['luggage_cent'] is None:result['not_included'].append('unresolved_luggage')
    result['luggage_predicates']=bag_audit
    mode=c.get('service_mode','unknown')
    if mode=='ordinary_phone_booking':result['ordinary_phone_booking_cent']=fee['ordinary_phone_booking_cent']
    elif mode in {'street_or_stand','fleet_prebooked'}:result['ordinary_phone_booking_cent']=0
    else:result['not_included'].append('unconfirmed_booking_mode')
    if c.get('animal_role')=='ordinary_pet' and c.get('pet_count') is not None:result['animal_cent']=fee['animal_per_head_cent']*c['pet_count']
    elif c.get('animal_role')=='ordinary_pet':result['not_included'].append('unresolved_pet_count')
    elif c.get('animal_role')=='guide_dog':result['not_included'].append('guide_dog_not_automatically_charged_as_pet')
    if mode=='fleet_prebooked':
        result['assumptions'].append('meter_reference_only_unless_platform_agreement_provided')
        if c.get('fleet_price_mode')=='agreed_total' and c.get('platform_agreed_total_cent') is not None:
            result['platform_agreed_total_cent']=cents(c['platform_agreed_total_cent'])
        elif c.get('fleet_price_mode')=='meter_plus_booking' and c.get('fleet_booking_cent') is not None:result['fleet_booking_cent']=cents(c['fleet_booking_cent'])
        else:result['not_included'].append('unconfirmed_fleet_agreement')
    components=['distance_cent','outbound_toll_cent','return_toll_cent','luggage_cent','ordinary_phone_booking_cent','animal_cent','fleet_booking_cent']
    result['meter_reference_cent']=sum(cents(result[k]) or 0 for k in components)
    result['total_known_cent']=result['platform_agreed_total_cent'] if result['platform_agreed_total_cent'] is not None else result['meter_reference_cent']
    result['price_basis']='platform_agreed_total' if result['platform_agreed_total_cent'] is not None else 'meter_reference' if mode=='fleet_prebooked' else 'known_meter_components'
    result['evidence_refs']=list(dict.fromkeys(result['evidence_refs']+[r['source_url'] for r in rules['fee_evidence']]))
    return result

def find_pickup_facilities(root,resolved_place,requested_kind,taxi_type='urban',service_mode='unknown'):
    if requested_kind not in KINDS-{'unknown'}:raise ValueError('Specify a facility category')
    if taxi_type not in {'urban','nt','lantau'}:raise ValueError('Unsupported taxi type')
    names=[resolved_place.get('query',''),resolved_place.get('name','')];coordinate=resolved_place.get('coordinate')
    cfg=json.loads((Path(root)/'configs/passenger_tools_v2.json').read_text(encoding='utf-8'));facilities=[];missing=[]
    verify_asset(root,cfg['facility_database'])
    with duckdb.connect(str(Path(root)/cfg['facility_database']),read_only=True) as db:
        rows=db.execute('SELECT facility_id,name,kind,service_modes,taxi_type_scope,area,longitude,latitude,source_url,evidence_id,place_keys FROM pickup_facilities WHERE kind=?',[requested_kind]).fetchall()
    for fid,name,kind,modes,scope,area,lon,lat,url,eid,keys in rows:
        known_scope=json.loads(scope)
        if known_scope and taxi_type not in known_scope:continue
        if not any(norm(k) in norm(n) or norm(n) in norm(k) for k in json.loads(keys) for n in names if len(norm(n))>=2):continue
        facilities.append(dict(id=fid,name=name,kind=kind,coordinate=[lon,lat] if lon is not None and lat is not None else None,floor=area,distance_m=None,distance_basis='not_available_without_verified_coordinate',source_url=url,evidence_id=eid,taxi_type_scope=known_scope or None,selection_reason='official named facility; confirm booking/order allocation; listing alone does not establish permission for each taxi type'))
    if requested_kind in {'ordinary_stand','cross_harbour_stand'}:
        allowed_type={'urban':'市區','nt':'新界','lantau':'大嶼山'}[taxi_type]
        # Over-harbour stands are urban taxi facilities; generic status need not contain 市區.
        with duckdb.connect(str(Path(root)/'data/processed/hk_taxi.duckdb'),read_only=True) as db:
            rows=db.execute('SELECT stand_id,location_tc,status_tc,longitude,latitude FROM taxi_stands WHERE (contains(status_tc, ?) OR (? AND contains(status_tc, ?))) AND ((? AND contains(status_tc, ?)) OR (? AND contains(status_tc, ?)))',
              [allowed_type,taxi_type=='urban','過海',requested_kind=='cross_harbour_stand','過海',requested_kind=='ordinary_stand',allowed_type]).fetchall()
        geod=Geod(ellps='WGS84')
        ranked=[]
        for fid,name,status,lon,lat in rows:
            if coordinate is None or lon is None or lat is None:continue
            distance=abs(geod.inv(coordinate[0],coordinate[1],lon,lat)[2])
            ranked.append(dict(id='stand:'+str(fid),name=name,kind=requested_kind,coordinate=[lon,lat],floor=None,distance_m=round(distance),distance_basis='straight_line_not_walking',source_url=cfg['ordinary_stands_source_url'],evidence_id='taxi_stands:'+str(fid),selection_reason=status))
        ranked.sort(key=lambda f:(f['distance_m'],f['id']))
        # Preserve exact floor-based official instructions ahead of merely nearby stands.
        facilities.extend(ranked[:3])
        if coordinate is None:missing.append('resolved_coordinate_required_for_nearest_search')
    if requested_kind=='fleet_prebooked_pickup' and service_mode!='fleet_prebooked':missing.append('confirm_fleet_prebooked_service')
    if not facilities:missing.append('no_verified_facility_matching_place_and_category')
    return dict(facilities=facilities[:3],unresolved_requirements=missing,automatic_boarding_exemption=False)
