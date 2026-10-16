"""Offline behavioral checks of the approved tools; no LLM output assumptions."""
from pathlib import Path
from copy import deepcopy
import json
import pytest
from hk_taxi.passenger_tools_v2 import upgrade_state,apply_condition_patch,fee_components,find_pickup_facilities
from hk_taxi.unified_passenger_v2 import apply_plan_v2,execute_journey_v2,UnifiedAssistant,empty_state
from test_unified import endpoint,plan,FakeBackend
R=Path(__file__).resolve().parents[1]

def bag(**kw):
    b=dict(count=1,location='trunk',total_cm=None,item_type='bag',mobility_use=None,light_status=None);b.update(kw);return b
def state():
    s=empty_state();s.update(origin=endpoint('A'),destination=endpoint('B'));return s
def route():return dict(estimate=dict(fare_cent=7940,source_url='https://www.td.gov.hk'),toll_quote=dict(status='identified',outbound_cent=2500,return_cent=2500,source_url='https://www.td.gov.hk'))

def test_cancel_clears_only_booking_assumptions_and_keeps_endpoints():
    s=state();s=apply_condition_patch(s,dict(service_mode='fleet_prebooked',fleet_price_mode='agreed_total',platform_agreed_total_cent=45000,pickup_facility_kind='fleet_prebooked_pickup'))
    n=apply_plan_v2(s,plan(conditions=dict(luggage='两件放尾箱',taxi_type=None,service_mode='street_or_stand',pickup_facility_kind='ordinary_stand',bags=[bag(count=2)])),FakeBackend(),R)
    assert n['origin']==s['origin'] and n['destination']==s['destination']
    assert n['conditions']['platform_agreed_total_cent'] is None and n['conditions']['service_mode']=='street_or_stand'
    q=fee_components(R,route(),n)
    assert q['luggage_cent']==1200 and q['total_known_cent']==14140 and q['waiting_cent'] is None and q['other_road_cent'] is None
    assert s['conditions']['platform_agreed_total_cent']==45000

def test_mode_null_unknown_preserves_confirmed():
    s=apply_condition_patch(state(),dict(service_mode='ordinary_phone_booking'))
    for patch in [dict(service_mode=None),dict(service_mode='unknown')]:assert apply_condition_patch(s,patch)['conditions']['service_mode']=='ordinary_phone_booking'

@pytest.mark.parametrize('total,light,expected',[(140,None,None),(141,None,600),(140,True,0),(141,True,None)])
def test_140_boundary_not_complete_definition(total,light,expected):
    s=apply_condition_patch(state(),dict(bags=[bag(location='passenger_compartment',total_cm=total,light_status=light)]))
    assert fee_components(R,route(),s)['luggage_cent']==expected

@pytest.mark.parametrize('item',['wheelchair','crutch'])
def test_mobility_exemption_before_trunk_condition(item):
    s=apply_condition_patch(state(),dict(bags=[bag(item_type=item,mobility_use=True)]))
    assert fee_components(R,route(),s)['luggage_cent']==0
    s['conditions']['bags'][0]['mobility_use']=None
    assert fee_components(R,route(),s)['luggage_cent'] is None

def test_phone_channel_not_automatically_ordinary_fee():
    s=apply_condition_patch(state(),dict(service_mode='fleet_prebooked'))
    assert fee_components(R,route(),s)['ordinary_phone_booking_cent']==0
    s=apply_condition_patch(s,dict(service_mode='ordinary_phone_booking'))
    assert fee_components(R,route(),s)['ordinary_phone_booking_cent']==500
    assert fee_components(R,route(),state())['ordinary_phone_booking_cent'] is None

def test_agreed_total_not_double_counted():
    s=apply_condition_patch(state(),dict(service_mode='fleet_prebooked',fleet_price_mode='agreed_total',platform_agreed_total_cent=45000,bags=[bag(count=2)]))
    q=fee_components(R,route(),s)
    assert q['total_known_cent']==45000 and q['meter_reference_cent']==14140

def test_price_agreement_change_clears_incompatible_old_amount():
    s=apply_condition_patch(state(),dict(service_mode='fleet_prebooked',fleet_price_mode='agreed_total',platform_agreed_total_cent=45000))
    s=apply_condition_patch(s,dict(fleet_price_mode='meter_plus_booking',fleet_booking_cent=1000))
    assert s['conditions']['platform_agreed_total_cent'] is None
    s=apply_condition_patch(s,dict(fleet_price_mode='agreed_total'))
    assert s['conditions']['fleet_booking_cent'] is None and s['conditions']['platform_agreed_total_cent'] is None

def test_guide_dog_not_assumed_pet():
    s=apply_condition_patch(state(),dict(animal_role='guide_dog'))
    assert fee_components(R,route(),s)['animal_cent'] is None
    s=apply_condition_patch(s,dict(animal_role='ordinary_pet',pet_count=2))
    assert fee_components(R,route(),s)['animal_cent']==1000

def test_rule_hypothesis_cannot_change_current_service():
    s=state();p=plan(tasks=['rules'],rag_query='如果车队预约怎么算',conditions=dict(luggage=None,taxi_type=None,service_mode='fleet_prebooked'))
    assert apply_plan_v2(s,p,FakeBackend(),R)==s

def test_mixed_known_unknown_bags_keep_known_subtotal():
    s=apply_condition_patch(state(),dict(bags=[bag(count=2),bag(location='passenger_compartment',total_cm=140)]))
    q=fee_components(R,route(),s)
    assert q['luggage_cent']==1200 and 'unresolved_luggage' in q['not_included']

def test_missing_toll_detection_is_unknown_not_free():
    r=route();r.pop('toll_quote');q=fee_components(R,r,state())
    assert q['outbound_toll_cent'] is None and q['return_toll_cent'] is None
    assert 'unconfirmed_harbour_tolls' in q['not_included']

def test_cancel_reuses_all_previous_route_cache_entries():
    s=state();b=FakeBackend()
    first,_=execute_journey_v2(s,b,R,'2026-10-17');calls=b.calls
    s=apply_condition_patch(s,dict(service_mode='street_or_stand',bags=[bag(count=2)]))
    again,_=execute_journey_v2(s,b,R,'2026-10-17')
    assert b.calls==calls and all(r['reused_session_result'] for r in again)
    assert all(r['quote_components']['luggage_cent']==1200 for r in again)

def test_sql_crossharbour_results_and_no_automatic_waiver():
    for query,coord in [('上环',[114.151,22.287]),('旺角',[114.17,22.32]),('九龙塘',[114.179,22.337])]:
        found=find_pickup_facilities(R,dict(query=query,coordinate=coord),'cross_harbour_stand')
        assert len(found['facilities'])==3
        assert not found['automatic_boarding_exemption']
        assert all('過海' in f['selection_reason'] and f['distance_basis']=='straight_line_not_walking' for f in found['facilities'])
        assert [f['distance_m'] for f in found['facilities']]==sorted(f['distance_m'] for f in found['facilities'])

@pytest.mark.parametrize('name',['西九龙','迪士尼','海洋公园','青衣站','啟德郵輪碼頭','M+博物館'])
def test_fleet_directory_generalization_without_invented_coordinates(name):
    found=find_pickup_facilities(R,dict(query=name),'fleet_prebooked_pickup',service_mode='fleet_prebooked')
    assert found['facilities'] and all(f['coordinate'] is None for f in found['facilities'])
    assert all(f['kind']=='fleet_prebooked_pickup' for f in found['facilities'])

def test_sql_injection_is_just_unmatched_text():
    assert not find_pickup_facilities(R,dict(query="'; DROP TABLE pickup_facilities;--"),'fleet_prebooked_pickup')['facilities']
    assert find_pickup_facilities(R,dict(query='海洋公园'),'fleet_prebooked_pickup')['facilities']

def test_explicit_facility_selection_binds_boarding_exemption():
    s=state();found=find_pickup_facilities(R,dict(query='上环',coordinate=[114.151,22.287]),'cross_harbour_stand');s['pickup_facilities']=found
    f=found['facilities'][0];p=plan(facility_choice=f['id'])
    n=apply_plan_v2(s,p,FakeBackend(),R)
    assert n['origin']['selected_id']==f['id'] and n['conditions']['boarding_type']=='cross_harbour_stand'
    moved=apply_plan_v2(n,plan(origin=dict(query='C',kind='place')),FakeBackend(),R)
    assert moved['conditions']['boarding_type']=='ordinary'

def test_new_controller_handles_actual_multiturn_without_network(monkeypatch):
    import hk_taxi.unified_passenger_v2 as mod
    b=FakeBackend();a=object.__new__(UnifiedAssistant);a.root=R;a.backend=b;a.as_of='2026-10-17';a.place_resolution_audit=[]
    p1=plan(conditions=dict(luggage=None,taxi_type=None,service_mode='fleet_prebooked'))
    p2=plan(conditions=dict(luggage='两件尾箱行李',taxi_type=None,service_mode='street_or_stand',bags=[bag(count=2)]))
    plans=iter([p1,p2]);a.route=lambda *args:(next(plans),dict(replay=True))
    s,r1=a.handle('我预约了车队的士',state(),generate=False);calls=b.calls
    n,r2=a.handle('取消预约，改排普通站，两件行李放尾箱',s,generate=False)
    assert r1['status']==r2['status']=='processed' and b.calls==calls
    assert n['conditions']['service_mode']=='street_or_stand' and '行李 HK$12.00' in r2['answer']
    assert len(n['history'])==2

def test_payment_only_keeps_all_trip_conditions_and_does_not_route():
    a=object.__new__(UnifiedAssistant);a.root=R;a.backend=FakeBackend();a.as_of='2026-10-17';a.place_resolution_audit=[]
    p=plan(tasks=['rules'],action='keep',rag_query='的士接受什么支付方式')
    a.route=lambda *args:(p,dict(replay=True))
    class Rag:
        def prepare_vectors(self,q):return {},0
        def retrieve(self,q,v):return {'hits':[]},[]
    a.get_rag=lambda:Rag()
    s=apply_condition_patch(state(),dict(service_mode='fleet_prebooked',bags=[bag(count=2)]));n,r=a.handle('那支付呢？',s,generate=False)
    assert a.backend.calls==0 and n['origin']==s['origin'] and n['conditions']==s['conditions']
    assert r['journey_results']==[] and 'retrieval' in r
