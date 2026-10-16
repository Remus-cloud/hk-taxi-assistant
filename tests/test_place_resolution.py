from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from hk_taxi.place_resolution import select_anchor,resolve_anchor,propose,is_station
from hk_taxi.multi_journey import inspect_route,MapError
from hk_taxi.unified import journey_text,apply_plan,empty_state

def row(zh,en='',x=835831,y=815410):return {'nameZH':zh,'nameEN':en,'x':x,'y':y}

def test_station_beats_far_same_name_village_and_service_records():
    rows=[row('測試灣','Test Bay'),row('測試灣','Test Tsai',x=818194),
        row('自動心臟除顫器 - 測試灣站','Automated Defibrillator - Test Bay Station'),
        row('港鐵測試灣站','MTR Test Bay Station')]
    anchor=select_anchor('测试湾',rows,'area')
    assert anchor['name']=='港鐵測試灣站' and anchor['resolution_method']=='area_station'

@pytest.mark.parametrize('query,name',[('湾仔','港鐵灣仔站'),('铜锣湾','港鐵銅鑼灣站'),('Wan Chai','MTR Wan Chai Station'),('Central','MTR Central Station')])
def test_station_prefix_normalization(query,name):
    assert is_station(query,row(name,name))
    assert not is_station(query,row(name+'-A1進出口',name+'-A1 Access'))

def test_area_station_lookup_filters_unrelated_search_hits():
    calls=[]
    def locations(q):
        calls.append(q)
        return [row('山峰','Peak')] if q.endswith('站') else [row('登峰觀景台','View Point')]
    seen=[]
    client=SimpleNamespace(locations=locations,config={'search_radius_m':400},place_resolver=lambda q,k,rs:(seen.extend(rs) or {'relation':'no_match'}))
    with pytest.raises(ValueError):resolve_anchor(client,'某景区','area')
    assert len(calls)==2 and all(r['nameZH']!='山峰' for r in seen)

def test_unique_one_character_completion_without_place_dictionary():
    assert select_anchor('紫荆广场',[row('金紫荊廣場')])['resolution_method']=='unique_name_completion'
    assert select_anchor('凤凰广场',[row('新鳳凰廣場')])['name']=='新鳳凰廣場'
    with pytest.raises(ValueError):select_anchor('酒店',[row('大酒店')])
    with pytest.raises(ValueError):select_anchor('香港大学',[row('香港大學圖書館')])

def test_same_completion_in_different_districts_requires_choice():
    with pytest.raises(ValueError,match='同名地点'):
        select_anchor('凤凰广场',[row('新鳳凰廣場'),row('新鳳凰廣場',x=818194)])

def test_area_fetches_station_when_broad_results_do_not_include_it():
    calls=[]
    def lookup(q):
        calls.append(q)
        return [row('港鐵測試站','MTR Test Station')] if q.endswith('站') else [row('測試','Test'),row('測試','Elsewhere',x=818194)]
    client=SimpleNamespace(locations=lookup,config={'search_radius_m':400})
    anchor,_=resolve_anchor(client,'测试','area')
    assert calls==['测试','测试站'] and anchor['name']=='港鐵測試站'

def test_visitor_queries_must_match_official_record_and_nearby_landmark():
    def locations(q):
        return [row('游客中心','Visitor Centre')] if q=='游客中心' else [row('山上觀景亭','View Point')]
    client=SimpleNamespace(locations=locations,config={'search_radius_m':400},
        place_resolver=lambda *a:{'relation':'visitor_hub','reference_index':0,'queries':['游客中心']})
    anchor,_=resolve_anchor(client,'山上景区','place')
    assert anchor['name']=='游客中心' and anchor['query']=='山上景区' and anchor['representative']
    distant=lambda q:[row('游客中心','Visitor Centre',x=818194)] if q=='游客中心' else [row('山上觀景亭')]
    client.locations=distant
    with pytest.raises(ValueError):resolve_anchor(client,'山上景区','place')

def test_model_cannot_supply_coordinates_or_unknown_reference():
    data={'relation':'alias','reference_index':99,'queries':[],'reason':'guess'}
    gen=SimpleNamespace(generate=lambda *a:SimpleNamespace(answer=json.dumps(data),truncated=False,unexpected_thinking=False))
    with pytest.raises(ValueError):propose(gen,'foo','place',[row('bar')])
    data.update(reference_index=0,coordinate=[114,22])
    with pytest.raises(ValueError):propose(gen,'foo','place',[row('bar')])

def raw_route(warnings,name='Ground Road'):
    return {'features':[{'geometry':{'coordinates':[[114.17,22.3],[114.18,22.31]]},'properties':{
        'summary':{'distance':3000,'duration':500},'segments':[{'steps':[{'name':name}]}],'warnings':warnings}}]}

def endpoint(id,coord):return {'id':id,'name':id,'coordinate':coord}

def test_approved_general_warning_does_not_block_route():
    w={'code':1,'message':'There may be restrictions on some roads'}
    r=inspect_route(raw_route([w]),endpoint('A',[114.17,22.3]),endpoint('B',[114.18,22.31]))
    assert r['route_advisories']==[w] and r['review_flags']==[]

def test_unknown_warning_and_flyover_still_block():
    for w,name in [([{'code':99,'message':'unknown restriction'}],'Ground Road'),([], '大坑道天橋')]:
        r=inspect_route(raw_route(w,name),endpoint('A',[114.17,22.3]),endpoint('B',[114.18,22.31]))
        assert r['review_flags']

def test_failed_endpoint_message_identifies_side_and_preserves_state():
    s=empty_state();before=deepcopy(s)
    plan={'tasks':['journey'],'action':'new','origin':{'query':'测试','kind':'place'},'destination':None,
        'origin_choice':None,'destination_choice':None,'conditions':{'luggage':None,'taxi_type':None}}
    backend=SimpleNamespace(discover=lambda *a:(_ for _ in ()).throw(MapError('timeout')))
    with pytest.raises(MapError,match='起点'):
        apply_plan(s,plan,backend)
    assert s==before

def test_real_failure_not_hidden_as_unknown_route():
    text,_=journey_text(empty_state(),[],'终点“测试”：未找到匹配的地点。')
    assert '终点“测试”' in text and '未找到匹配' in text

def test_currency_does_not_turn_fare_lines_into_math():
    from hk_taxi.web_chat import split_answer
    record={'answer':'参考车费：HK$39.50\n路线金额 HK$39.50。'}
    body,_=split_answer(record)
    assert body.count('HK\\$39.50')==2
    assert split_answer({'answer':body})[0]==body

def test_category_ambiguity_asks_for_place_instead_of_guessing():
    client=SimpleNamespace(locations=lambda q:[row('第一醫院'),row('第二醫院',x=818194)],
        config={'search_radius_m':400},place_resolver=lambda *a:{'relation':'ambiguous','reference_index':None,'queries':[]})
    with pytest.raises(ValueError,match='你指哪一处'):
        resolve_anchor(client,'医院','place')
