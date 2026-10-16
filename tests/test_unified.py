from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from hk_taxi.unified import (empty_state,compact_state,choose,apply_plan,route_key,execute_journey,
    Budget,BudgetError,PlanError,validate_plan,UnifiedAssistant)
from hk_taxi.fare_calculator import DistanceRule
from hk_taxi.multi_journey import MapError

ROOT=Path(__file__).resolve().parents[1]

def endpoint(name='A'):
    return {'anchor':{'query':name,'name':name,'kind':'place'},'candidates':[
       {'id':name+str(i),'name':name+'点'+str(i),'kind':'ground_road_reference','coordinate':[114.17+i*.001,22.3]} for i in range(2)]}

def plan(**updates):
    p={'tasks':['journey'],'action':'update','origin':None,'destination':None,'origin_choice':None,
       'destination_choice':None,'conditions':{'luggage':None,'taxi_type':None},'rag_query':'','clarification':''}
    p.update(updates);return p

class FakeBackend:
    turn_limit=3
    def __init__(self):self.calls=0
    def same_place(self,q,e):return q==e['anchor']['query']
    def discover(self,q,kind):
        if q=='missing':raise MapError('missing')
        return endpoint(q)
    def route(self,a,b):
        self.calls+=1
        return {'origin_id':a['id'],'destination_id':b['id'],'origin_name':a['name'],'destination_name':b['name'],
          'origin_coordinate':a['coordinate'],'destination_coordinate':b['coordinate'],'distance_m':3000,
          'duration_s':500,'review_flags':[],'source_mode':'test_stub','requested_at':'2026-10-15','status':'provisional_distance_estimate'}

def populated():
    state=empty_state();state.update(origin=endpoint('A'),destination=endpoint('B'))
    state['conditions']['luggage']='两件，车尾箱'
    return state

def test_only_origin_updates_preserving_destination_and_luggage():
    old=populated();new=apply_plan(old,plan(origin={'query':'C','kind':'place'}),FakeBackend())
    assert new['origin']['anchor']['name']=='C'
    assert new['destination']==old['destination'] and new['conditions']==old['conditions']
    assert old['origin']['anchor']['name']=='A'

def test_rule_green_taxi_question_does_not_change_active_trip():
    old=populated()
    p=plan(tasks=['rules'],origin={'query':'C','kind':'place'},conditions={'taxi_type':'nt','luggage':'none'},rag_query='绿的规则')
    assert apply_plan(old,p,FakeBackend())==old

def test_explicit_luggage_clear_keeps_locations():
    old=populated();new=apply_plan(old,plan(conditions={'luggage':'none','taxi_type':None}),FakeBackend())
    assert new['conditions']['luggage']=='none'
    assert new['origin']==old['origin'] and new['destination']==old['destination']

def test_failed_update_does_not_mutate_old_state():
    old=populated();before=deepcopy(old)
    with pytest.raises(MapError):apply_plan(old,plan(origin={'query':'missing','kind':'place'}),FakeBackend())
    assert old==before

def test_candidate_hint_must_be_unique():
    with pytest.raises(PlanError):choose(endpoint('A'),'点')
    assert choose(endpoint('A'),'A1')['selected_id']=='A1'

def test_reverse_ambiguous_rejected():
    with pytest.raises(PlanError):apply_plan(populated(),plan(action='reverse'),FakeBackend())

def test_reverse_selected_pair_preserves_conditions():
    old=populated();old['origin']=choose(old['origin'],'A1');old['destination']=choose(old['destination'],'B0')
    new=apply_plan(old,plan(action='reverse'),FakeBackend())
    assert new['origin']['selected_id']=='B0' and new['destination']['selected_id']=='A1'
    assert new['conditions']==old['conditions']

def test_new_trip_clears_previous_luggage():
    new=apply_plan(populated(),plan(action='new',origin={'query':'C','kind':'place'}),FakeBackend())
    assert new['conditions']['luggage']=='unknown' and new['destination'] is None

def test_same_place_does_not_drop_chosen_access():
    old=populated();old['origin']=choose(old['origin'],'A1')
    new=apply_plan(old,plan(origin={'query':'A','kind':'place'}),FakeBackend())
    assert new['origin']['selected_id']=='A1'

def test_route_key_is_direction_date_and_coordinate_sensitive():
    a,b=endpoint('A')['candidates'][0],endpoint('B')['candidates'][0]
    key=route_key(a,b,'2026-10-15')
    assert key!=route_key(b,a,'2026-10-15')
    assert key!=route_key(a,b,'2026-10-16')
    b=deepcopy(b);b['coordinate'][0]+=.001
    assert key!=route_key(a,b,'2026-10-15')

def test_durable_budget_reserves_failures_and_prevents_extra(tmp_path):
    b=Budget(tmp_path/'budget.json',{'router':1})
    b.reserve('router')
    resumed=Budget(tmp_path/'budget.json',{'router':1})
    with pytest.raises(BudgetError):resumed.reserve('router')
    assert resumed.counts['router']==1

def test_plan_schema_rejects_unknown_tool_and_invented_fields():
    schema=json.loads((ROOT/'prompts/unified/router.schema.json').read_text(encoding='utf-8'))
    p=plan(tasks=['rules'],rag_query='规则')
    assert validate_plan(json.dumps(p),schema)==p
    p['api_key']='not-allowed'
    with pytest.raises(PlanError):validate_plan(json.dumps(p),schema)

def test_invalid_plan_no_rules_query():
    schema=json.loads((ROOT/'prompts/unified/router.schema.json').read_text(encoding='utf-8'))
    with pytest.raises(PlanError):validate_plan(json.dumps(plan(tasks=['rules'])),schema)

def test_compact_state_does_not_send_whole_history_or_routes():
    s=populated();s['history']=[{'large':'history'}];s['routes']={'secret':'unused'}
    text=json.dumps(compact_state(s))
    assert 'large' not in text and 'secret' not in text

def test_same_pair_reuses_only_session_result(monkeypatch):
    import hk_taxi.unified as module
    rule=DistanceRule('v','2024-07-14','2026-10-15','https://www.td.gov.hk',2000,2900,200,10250,210,140)
    monkeypatch.setattr(module,'load_distance_rule',lambda *a:rule)
    s=populated();backend=FakeBackend()
    first,_=execute_journey(s,backend,ROOT,'2026-10-15')
    assert backend.calls==2
    second,_=execute_journey(s,backend,ROOT,'2026-10-15')
    assert backend.calls==2 and all(r['reused_session_result'] for r in second)
    assert all(r['estimate']['fare_hkd']=='39.50' for r in first)

def test_route_failure_does_not_drop_rule_answer(monkeypatch):
    assistant=UnifiedAssistant.__new__(UnifiedAssistant)
    assistant.root=ROOT;assistant.as_of='2026-10-15';assistant.budget=None;assistant.backend=FakeBackend()
    p=plan(tasks=['journey','rules'],origin={'query':'missing','kind':'place'},rag_query='的士打印机坏了收据')
    assistant.route=lambda *a:(p,{})
    schema=json.loads((ROOT/'prompts/rag_answer_v1.schema.json').read_text(encoding='utf-8'))
    source={'source_id':'S1','source_url':'https://www.td.gov.hk','title':'收据','text':'手写收据'}
    fake_rag=SimpleNamespace(prepare_vectors=lambda *a:({},0),retrieve=lambda *a:({'hits':[{'chunk':{'chunk_id':'receipt'}}]},[source]),schema=schema)
    assistant.get_rag=lambda:fake_rag
    answer={'status':'answered','answer_parts':[{'text':'打印机坏时可发手写收据。','source_ids':['S1']}],'clarifying_questions':[],'limitations':[]}
    assistant.answer_generator=SimpleNamespace(generate=lambda *a:SimpleNamespace(answer=json.dumps(answer),truncated=False,unexpected_thinking=False,usage={},returned_model='mock'))
    assistant.mixed_system='mock'
    old=populated();new,result=assistant.handle('新起点加收据问题',old)
    assert '手写收据' in result['answer']
    assert result['slot_update_failed'] and new['origin']==old['origin']

def test_empty_task_condition_update_is_strictly_bounded():
    schema=json.loads((ROOT/'prompts/unified/router.schema.json').read_text(encoding='utf-8'))
    p=plan(tasks=[],conditions={'luggage':'none','taxi_type':None})
    assert validate_plan(json.dumps(p),schema)==p
    state=apply_plan(populated(),p,FakeBackend())
    assert state['conditions']['luggage']=='none'
    with pytest.raises(PlanError): validate_plan(json.dumps(plan(tasks=[])),schema)
    p['destination']={'query':'C','kind':'place'}
    with pytest.raises(PlanError): validate_plan(json.dumps(p),schema)

def test_compound_stand_resolution_uses_official_data_not_place_hardcode():
    from hk_taxi.unified import explicit_stand_endpoint
    stands=[{'stand_id':1,'location_tc':'测试路','location_en':'Test Road','status_tc':'市區的士站','longitude':114.17,'latitude':22.3}]
    result=explicit_stand_endpoint('某建筑附近测试路的士站',stands)
    assert result['selected_id']=='stand-1'
    assert result['selection_basis']=='user_explicit_stand_name'
    assert explicit_stand_endpoint('测试路在哪里',stands) is None
    assert explicit_stand_endpoint('某建筑测试路的士站',stands+stands) is None

def test_selecting_origin_does_not_expand_previous_destination(monkeypatch):
    import hk_taxi.unified as module
    rule=DistanceRule('v','2024-07-14','2026-10-15','https://www.td.gov.hk',2000,2900,200,10250,210,140)
    monkeypatch.setattr(module,'load_distance_rule',lambda *a:rule)
    s=populated();backend=FakeBackend()
    execute_journey(s,backend,ROOT,'2026-10-15')
    assert s['destination']['selected_id']=='B0'
    s['origin']=choose(s['origin'],'A1')
    results,_=execute_journey(s,backend,ROOT,'2026-10-15')
    assert len(results)==1 and results[0]['destination_id']=='B0' and backend.calls==2

def test_tariff_before_effective_date_stops_before_map_call(monkeypatch):
    import hk_taxi.unified as module
    rule=DistanceRule('v','2024-07-14','2026-10-15','https://www.td.gov.hk',2000,2900,200,10250,210,140)
    monkeypatch.setattr(module,'load_distance_rule',lambda *a:rule)
    backend=FakeBackend()
    with pytest.raises(ValueError):execute_journey(populated(),backend,ROOT,'2024-07-13')
    assert backend.calls==0
