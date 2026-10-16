from copy import deepcopy
from pathlib import Path
import json
import pytest
from hk_taxi.harbour_tolls import load,identify,quote
from hk_taxi.journey_context import repair,update_pending
from hk_taxi.unified import empty_state,route_key,apply_plan,validate_plan
R=Path(__file__).resolve().parents[1]

def plan(**kw):
    p={'tasks':['rules'],'action':'keep','origin':None,'destination':None,'origin_choice':None,'destination_choice':None,
       'conditions':{'luggage':None,'taxi_type':None},'rag_query':'落客点规定','clarification':''}
    p.update(kw);return p

def pending():
    s=empty_state();s.update(origin={'anchor':{'query':'测试医院','name':'測試醫院','kind':'place'},'candidates':[]},
        destination={'anchor':{'query':'测试公园','name':'測試公園','kind':'place'},'candidates':[]},pending_followup={'type':'journey_access'})
    return s

def test_short_dropoff_reply_repairs_only_current_intent():
    old=plan();new,audit=repair('测试公园的落客点',pending(),old)
    assert new['tasks']==['journey'] and new['access_update']=='destination'
    assert new['origin'] is None and new['destination'] is None and audit['changed']
    assert old['tasks']==['rules']

def test_generic_dropoff_not_marked_as_explicit_candidate_confirmation():
    p,_=repair('测试公园的落客点',pending(),plan(destination_choice='invented'))
    assert p['destination_choice'] is None

def test_regulatory_followup_does_not_repeat_trip_price():
    p,_=repair('测试公园可以随便落客吗？',pending(),plan(tasks=['journey','rules']))
    assert p['tasks']==['rules']

def test_missing_rule_subquery_recovers_exact_user_text():
    from hk_taxi.journey_context import recover_rule_query
    p=plan(tasks=['journey','rules'],rag_query='')
    raw,changed=recover_rule_query(json.dumps(p),'在这里下车，两件行李放尾箱要加多少钱？')
    assert changed and json.loads(raw)['rag_query']=='在这里下车，两件行李放尾箱要加多少钱？'

@pytest.mark.parametrize('q',['测试公园可以随便落客吗？','测试公园的落客规定是什么？','Is drop-off allowed there?'])
def test_real_regulatory_question_not_changed(q):
    p=plan();new,audit=repair(q,pending(),p)
    assert new==p and not audit['changed']

def test_mixed_access_and_receipt_keeps_both_tasks():
    p,_=repair('就在那个落客点，另外可以拿收据吗',pending(),plan(rag_query='能否拿收据'))
    assert set(p['tasks'])=={'rules','journey'} and '收据' in p['rag_query']

def test_no_active_pending_request_does_not_rewrite_question():
    p=plan();assert repair('测试公园的落客点',empty_state(),p)[0]==p

def test_schema_limits_via_and_boarding_conditions():
    schema=json.loads((R/'prompts/unified/router.schema.json').read_text(encoding='utf-8'))
    p=plan(tasks=['journey'],rag_query='',via=[{'query':'测试','kind':'area'}])
    assert validate_plan(json.dumps(p),schema)==p
    p['via']*=2
    with pytest.raises(ValueError):validate_plan(json.dumps(p),schema)

def test_via_route_key_different_and_direction_sensitive():
    a={'id':'a','coordinate':[114.1,22.3]};b={'id':'b','coordinate':[114.2,22.3]};v={'id':'v','coordinate':[114.15,22.3]}
    assert route_key(a,b,'2026-10-16')!=route_key(a,b,'2026-10-16',[v])

def make_geometry():
    return {'tunnels':[{'tunnel_id':'western','ways':[{'coordinates':[[114.15,22.28],[114.15,22.31]]}]}]}

CFG={'match_tolerance_m':40,'minimum_match_m':200}

def route(coords,values):
    return {'geometry_coordinates':coords,'extras':{'tollways':{'values':values}},'estimate':{'fare_cent':7940}}

def test_synthetic_geometry_requires_toll_flags_and_length():
    r=route([[114.15,22.28],[114.15,22.31]],[[0,1,1]])
    assert len(identify(r,make_geometry(),CFG)['crossings'])==1
    r['extras']['tollways']['values']=[[0,1,0]]
    assert not identify(r,make_geometry(),CFG)['crossings']
    r=route([[114.15,22.28],[114.15,22.2805]],[[0,1,1]])
    assert not identify(r,make_geometry(),CFG)['crossings']

def test_parallel_coastal_road_not_tunnel():
    r=route([[114.16,22.28],[114.16,22.31]],[[0,1,1]])
    assert identify(r,make_geometry(),CFG)['status']=='unresolved_toll'

def test_missing_route_data_does_not_guess_tolls():
    assert identify({},make_geometry(),CFG)['status']=='unavailable'

def test_sql_single_hiring_and_cross_harbour_stand_exemption(monkeypatch):
    import hk_taxi.harbour_tolls as m
    cfg={'version':'test','source_url':'https://www.td.gov.hk'}
    rules={'western':{'name':'西隧','toll_cent':2500,'return_cent':2500}}
    monkeypatch.setattr(m,'load',lambda root:(cfg,{},rules))
    monkeypatch.setattr(m,'identify',lambda *a:{'status':'identified','crossings':[{'tunnel_id':'western'}]})
    s=empty_state();r={'estimate':{'fare_cent':7940}}
    assert quote(R,r,s)['subtotal_cent']==12940
    s['conditions']['boarding_type']='cross_harbour_stand'
    assert quote(R,r,s)['subtotal_cent']==10440
    assert r['estimate']['fare_cent']==7940

def test_round_trip_two_real_tolls_no_return_surcharge(monkeypatch):
    import hk_taxi.harbour_tolls as m
    cfg={'version':'test','source_url':'https://www.td.gov.hk'}
    rules={'western':{'name':'西隧','toll_cent':2500,'return_cent':2500}}
    monkeypatch.setattr(m,'load',lambda root:(cfg,{},rules))
    monkeypatch.setattr(m,'identify',lambda *a:{'status':'identified','crossings':[{'tunnel_id':'western'},{'tunnel_id':'western'}]})
    q=quote(R,{'estimate':{'fare_cent':12000}},empty_state())
    assert q['outbound_cent']==5000 and q['return_cent']==0 and q['subtotal_cent']==17000
    assert q['exemption']=='same_side_final_destination'

def test_pure_rule_question_cannot_change_boarding_or_active_trip():
    s=pending();before=deepcopy(s)
    p=plan(conditions={'luggage':None,'taxi_type':None,'boarding_type':'cross_harbour_stand'})
    assert apply_plan(s,p,object())==before

def test_unset_condition_patch_preserves_previous_boarding():
    s=empty_state();s['conditions']['boarding_type']='cross_harbour_stand'
    p=plan(tasks=['journey'],rag_query='')
    assert apply_plan(s,p,object())['conditions']['boarding_type']=='cross_harbour_stand'

def test_empty_task_boarding_update_still_recalculates_without_losing_luggage():
    schema=json.loads((R/'prompts/unified/router.schema.json').read_text(encoding='utf-8'))
    p=plan(tasks=[],action='update',rag_query='',conditions={'luggage':None,'taxi_type':None,'boarding_type':'cross_harbour_stand'})
    validated=validate_plan(json.dumps(p),schema)
    assert validated['tasks']==['journey']
    s=pending();s['conditions']['luggage']='两件放尾箱'
    updated=apply_plan(s,validated,object())
    assert updated['conditions']['boarding_type']=='cross_harbour_stand' and updated['conditions']['luggage']=='两件放尾箱'

def test_sql_initializer_is_idempotent_and_versions_cannot_be_overwritten(tmp_path):
    from hk_taxi.harbour_tolls import initialize
    (tmp_path/'configs').mkdir();(tmp_path/'data/reference').mkdir(parents=True)
    c={'approval_status':'approved','version':'fixture-v1','geometry_file':'data/reference/geometry.json',
       'rules':[{'tunnel_id':'test','name':'测试隧道','toll_cent':2500,'return_cent':2500,'source_url':'https://www.td.gov.hk','evidence_text':'test'}]}
    (tmp_path/'data/reference/geometry.json').write_text(json.dumps(make_geometry()),encoding='utf-8')
    p=tmp_path/'configs/harbour_tolls_v1.json';p.write_text(json.dumps(c),encoding='utf-8')
    assert initialize(tmp_path)==initialize(tmp_path)
    c['rules'][0]['toll_cent']=3000;p.write_text(json.dumps(c),encoding='utf-8')
    with pytest.raises(ValueError,match='version changed'):initialize(tmp_path)
