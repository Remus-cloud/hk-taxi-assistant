from copy import deepcopy
from pathlib import Path
import json
import pytest
from hk_taxi.structured_guard import (RuleStore,evaluate_baggage,protect_answer,
    normalize_answer,protected_topics,script_text)

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=json.loads((ROOT/'prompts/rag_answer_v1.schema.json').read_text(encoding='utf-8'))
@pytest.fixture(scope='module')
def store():return RuleStore(ROOT)
def parsed(text,ids=None):
    return {'status':'answered','answer_parts':[{'text':text,'source_ids':ids or []}], 'clarifying_questions':[],'limitations':[]}
def guard(q,text,store,lang='zh-Hans',sources=()):
    return protect_answer(ROOT,q,parsed(text),list(sources),lang,SCHEMA,store)

@pytest.mark.parametrize('size,expected',[('139.9','dimension_condition_not_met'),('140','dimension_condition_not_met'),('140.000','dimension_condition_not_met'),('140.1','general_charge_condition_met'),('141','general_charge_condition_met')])
def test_strict_boundary(store,size,expected):
    r=evaluate_baggage(store,'passenger_compartment',size)
    assert r['outcome']==expected and r['complete_legal_definition_established'] is False

@pytest.mark.parametrize('item',['wheelchair','crutch'])
@pytest.mark.parametrize('location',['trunk','passenger_compartment'])
def test_explicit_exemption_before_general_condition(store,item,location):
    assert evaluate_baggage(store,location,200,item,True)['outcome']=='exempt'

def test_unknown_aid_use_not_assumed_exempt(store):
    assert evaluate_baggage(store,'trunk',50,'wheelchair',None)['outcome']=='needs_clarification'
def test_non_mobility_cargo_has_no_automatic_exemption(store):
    assert evaluate_baggage(store,'trunk',50,'wheelchair',False)['outcome']=='general_charge_condition_met'
def test_trunk_small_bag_not_cabin_exemption(store):
    assert evaluate_baggage(store,'trunk',50)['outcome']=='general_charge_condition_met'
def test_stroller_not_wheelchair(store):
    assert evaluate_baggage(store,'trunk',80,'stroller',True)['outcome']=='general_charge_condition_met'
def test_confirmed_light_bag_and_conflict(store):
    assert evaluate_baggage(store,'passenger_compartment',None,light_status=True)['outcome']=='exempt'
    assert evaluate_baggage(store,'passenger_compartment',141,light_status=True)['outcome']=='needs_clarification'
def test_unknown_condition_not_false(store):
    assert evaluate_baggage(store)['outcome']=='needs_clarification'
    assert evaluate_baggage(store,'passenger_compartment')['outcome']=='needs_clarification'
@pytest.mark.parametrize('value',['NaN','Infinity','-1',True,'abc'])
def test_invalid_size(store,value):
    with pytest.raises(ValueError):evaluate_baggage(store,'passenger_compartment',value)

def test_date_binding_keeps_valid_and_blocks_inferred_date(store):
    r=guard('市区起步29元和行李6元都是同一天开始吗？','市区起步2024-07-14生效。行李也是2024-07-14开始。',store)
    text='\n'.join(p['text'] for p in r['parsed']['answer_parts'])
    assert '这次费率自2024-07-14' in text
    assert '不能确认行李' in text and '行李也是' not in text
    assert r['parsed']['status']=='partial_answer' and not r['audit']['validation_errors']
    assert store.get('R02')['effective_from'] is None
def test_condition_not_definition(store):
    r=guard('140厘米以内就是轻便行李的完整定义吗？','140厘米以内即视为轻便行李。',store)
    text=r['rendered_answer']
    assert '即视为' not in text and '不能据此认定' in text
def test_user_requested_weight_gap_remains(store):
    r=guard('轻便行李最多几公斤？','没有重量规定。',store)
    assert '本次提供的相关段落未给出公斤上限' in r['rendered_answer']
    assert r['audit']['coverage_ids']==['C03']
def test_not_every_number_is_a_dimension_sum(store):
    r=guard('行李长140厘米，宽高不知道，收费吗？','需要了解总尺寸。',store)
    assert '尺寸总和140厘米：' not in r['rendered_answer']
def test_compare_two_sums(store):
    r=guard('车厢行李尺寸总和140厘米与141厘米分别是否收费？','140就是轻便。',store)
    assert '总和140厘米：未触发' in r['rendered_answer'] and '总和141厘米：触发' in r['rendered_answer']
def test_stroller_uses_same_criteria(store):
    r=guard('婴儿车放车尾箱收费吗？','婴儿车都免费。',store)
    assert '婴儿车按照上述准则' in r['rendered_answer'] and '婴儿车都免费' not in r['rendered_answer']

def test_payment_clause_in_same_part_survives(store):
    r=guard('Visa可以吗，小背包收费吗？','Visa上车先问司机，小背包收费完全由司机决定。',store)
    assert 'Visa上车先问司机' in r['rendered_answer']
    assert '收费完全由司机决定' not in r['rendered_answer']
def test_implicit_driver_continuation_cannot_escape_owned_answer(store):
    r=guard('轻便背包收费吗？','这件物品当然要收费。这完全由司机决定。',store)
    assert '这完全由司机决定' not in r['rendered_answer']
def test_unrelated_same_number_is_not_base_fare_evidence(store):
    r=guard('市区起步价和隧道费？','市区起步价29元。该隧道收费29元。',store)
    assert '该隧道收费29元' in r['rendered_answer']
def test_receipt_mention_is_not_baggage_fee_question(store):
    q='收据应列出行李收费明细吗？'
    assert protected_topics(q)['baggage'] is False
    data=parsed('本次资料不足以确认该明细要求。')
    r=protect_answer(ROOT,q,data,[],'zh-Hans',SCHEMA,store)
    assert r['parsed']==data and not r['audit']['removed_fragments']
def test_outside_topic_unchanged(store):
    data=parsed('并非每辆都支持Visa；打印机坏了应给手写收据。')
    r=protect_answer(ROOT,'Visa和收据怎么处理？',data,[],'zh-Hans',SCHEMA,store)
    assert data==r['parsed'] and r['sources']==[]
def test_explicit_nt_base_not_replaced_with_urban(store):
    r=guard('新界绿的起步多少钱？','新界起步为25.5元。',store)
    assert '25.5' in r['rendered_answer'] and '29' not in r['rendered_answer']
def test_unknown_taxi_type_asks_instead_of_urban_default(store):
    r=guard('起步多少钱？','一定29元。',store)
    assert '29元' not in r['rendered_answer'] and r['parsed']['clarifying_questions']

def test_hans_normalization_not_original_source():
    sources=[{'source_id':'S1','text':'殘疾乘客的輪椅及拐杖無須收費。','title':'車費收據','source_url':'https://example.test/車費'}]
    original=deepcopy(sources)
    p=parsed('收據打印機失靈時應發出手寫收據。原文：“殘疾乘客的輪椅及拐杖無須收費。”')
    out,changes=normalize_answer(p,'zh-Hans',sources)
    assert '收据打印机失灵时应发出手写收据' in out['answer_parts'][0]['text']
    assert '“殘疾乘客的輪椅及拐杖無須收費。”' in out['answer_parts'][0]['text']
    assert changes and sources==original and '收據' in p['answer_parts'][0]['text']
def test_hant_and_english():
    assert '收據' in script_text('收据','zh-Hant')
    assert script_text('English Visa HK$29','en')=='English Visa HK$29'
def test_english_rule_rendering(store):
    r=guard('When did the urban taxi base fare and luggage fee start?','Both started in 2024.',store,'en')
    assert '2024-07-14' in r['rendered_answer'] and 'do not establish' in r['rendered_answer']
def test_original_response_and_citation_ids_preserved(store):
    source={'source_id':'S8','chunk_id':'old','title':'舊來源','text':'test','source_url':'https://example.test/a'}
    p=parsed('行李6元自2024年生效。',['S8']);saved=deepcopy(p)
    r=protect_answer(ROOT,'行李费何时生效？',p,[source],'zh-Hans',SCHEMA,store)
    assert p==saved and r['sources'][0]==source
    assert any(s['source_id']=='S9' for s in r['sources']) and not r['audit']['validation_errors']
def test_sql_lookup_is_parameterized(store):
    with pytest.raises(ValueError):store.get("R01' OR '1'='1")
def test_external_database_reference_is_rejected(tmp_path):
    cfg=json.loads((ROOT/'configs/structured_guard_v1.json').read_text(encoding='utf-8'))
    cfg['database']=str(ROOT/cfg['database']);cfg['database_sha256']='0'*64
    (tmp_path/'configs').mkdir();(tmp_path/'configs/structured_guard_v1.json').write_text(json.dumps(cfg),encoding='utf-8')
    with pytest.raises(ValueError,match='escapes the project root'):RuleStore(tmp_path)
def test_unavailable_rule_store_preserves_independent_payment(tmp_path):
    p=parsed('Visa上车先问司机。行李6元从2024年开始。')
    r=protect_answer(tmp_path,'Visa可以吗，行李费何时生效？',p,[],'zh-Hans',SCHEMA)
    assert 'Visa上车先问司机' in r['rendered_answer']
    assert '从2024年开始' not in r['rendered_answer']
    assert r['audit']['failed_closed'] and r['parsed']['status']=='partial_answer'

def test_unified_hook_applies_guard_without_live_clients(monkeypatch):
    from types import SimpleNamespace
    from hk_taxi.unified import UnifiedAssistant,empty_state
    from hk_taxi.rag_answer import render_answer
    a=UnifiedAssistant.__new__(UnifiedAssistant);a.root=ROOT;a.budget=None;a.as_of='2026-10-15'
    raw=parsed('可以。如果收據打印機失靈，司機應發出手寫收據。')
    plan={'tasks':['rules'],'action':'keep','origin':None,'destination':None,'origin_choice':None,'destination_choice':None,'conditions':{'taxi_type':None,'luggage':None},'rag_query':'打印机坏了能给收据吗？','clarification':''}
    a.route=lambda *args:(plan,{'raw':'stub'})
    fake=SimpleNamespace(schema=SCHEMA,prepare_vectors=lambda *a:({},0),retrieve=lambda *a:({'hits':[]},[]),generate_from_sources=lambda *a:{'parsed':deepcopy(raw),'validation_errors':[],'rendered_answer':render_answer(raw,[],'zh-Hans')})
    a.get_rag=lambda:fake;a.backend=SimpleNamespace()
    state,r=a.handle('打印机坏了能给收据吗？',empty_state(),language='zh-Hans')
    assert '手写收据' in r['answer'] and '手寫收據' not in r['answer']
    assert '手寫收據' in r['generation']['parsed']['answer_parts'][0]['text']
    assert r['structured_guard']['audit']['script_changes']
    import hk_taxi.structured_guard as guard_module
    def failure(*args,**kwargs):raise ValueError('unavailable evidence')
    monkeypatch.setattr(guard_module,'protect_answer',failure)
    _,failed=a.handle('打印机坏了能给收据吗？',empty_state(),language='zh-Hans')
    assert failed['structured_guard']['failed_closed'] is True
    assert '手寫收據' not in failed['answer']
    assert failed['generation']['parsed']==raw
