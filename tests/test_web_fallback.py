from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from hk_taxi.web_fallback import needs_web,parse_web_response,answer_with_web
from hk_taxi.generation import GenerationAPIError
from hk_taxi.unified import UnifiedAssistant,empty_state,validate_plan
from hk_taxi.web_chat import split_answer

ROOT=Path(__file__).resolve().parents[1]

def response(text='车内禁止吸烟。',annotations=None,tools=None):
    return {'status':'completed','model':'test','output':list(tools or [])+[
        {'type':'message','content':[{'type':'output_text','text':text,'annotations':annotations or []}]}]}

def test_web_trigger_uses_coverage_not_similarity():
    for status in ['partial_answer','insufficient_evidence','out_of_scope']:
        assert needs_web({'parsed':{'status':status}})
    assert not needs_web({'parsed':{'status':'answered','limitations':[]}})
    assert needs_web({'parsed':{'status':'answered'}},{'parsed':{'status':'partial_answer'}})
    assert not needs_web({'validation_errors':['invalid_json']})

def test_provider_citation_registry_and_language():
    d=response(annotations=[{'type':'url_citation','url':'https://www.td.gov.hk/test','title':'官方規例','start_index':0,'end_index':8}])
    r=parse_web_response(d,'zh-Hant',True)
    assert '車內禁止' in r['answer'] and r['sources'][0]['source_id']=='W1'
    assert 'https://www.td.gov.hk/test' in r['answer']

def test_unverified_links_are_not_published():
    r=parse_web_response(response('解释。[假来源](https://fake.example/test)'),'zh-Hans')
    assert 'https://fake.example' not in r['answer']

def test_unmapped_numeric_citations_are_removed():
    r=parse_web_response(response('解释。[1][23]'),'zh-Hans')
    assert '[1]' not in r['answer'] and '[23]' not in r['answer']

def test_search_requires_real_sources():
    with pytest.raises(GenerationAPIError):parse_web_response(response(),'zh-Hans',True)
    with pytest.raises(GenerationAPIError):parse_web_response(response(tools=[{'type':'web_search_call'}]),'zh-Hans')

def test_unsafe_links_not_registered():
    with pytest.raises(GenerationAPIError):
        parse_web_response(response(annotations=[{'type':'url_citation','url':'javascript:alert(1)'}]),'zh-Hans',True)

def test_partial_generation_not_shown_as_completed():
    d=response();d['status']='incomplete'
    with pytest.raises(GenerationAPIError):parse_web_response(d,'en')

def test_source_footer_does_not_hide_later_web_answer():
    record={'answer':'本地内容。[S1](https://local.test)\n\n来源（官方原文）：\n\n- [S1 · 标题](https://local.test)\n\n网页补充。[W1](https://web.test)\n\n网页来源：\n\n- [W1 · 网页](https://web.test)',
            'retrieval':{'sources':[{'source_id':'S1'}]},'web_search':{'sources':[{'source_id':'W1'}]}}
    body,sources=split_answer(record)
    assert '网页补充' in body and '来源（官方原文）' not in body and '网页来源：' not in body
    assert [s['source_id'] for s in sources]==['S1','W1']

def make_assistant(tasks):
    a=UnifiedAssistant.__new__(UnifiedAssistant);a.root=ROOT;a.budget=None;a.backend=None
    a.route=lambda *args:({'tasks':tasks,'action':'keep','origin':None,'destination':None,'origin_choice':None,
        'destination_choice':None,'conditions':{'luggage':None,'taxi_type':None},'rag_query':'规则','clarification':''},{})
    a.answer_generator=object()
    return a

@pytest.mark.parametrize('task',['general','out_of_scope'])
def test_general_route_keeps_trip_and_answers(monkeypatch,task):
    import hk_taxi.web_fallback as module
    calls=[]
    def fake(*args):calls.append(args);return {'answer':'水在标准大气压下约100°C沸腾。','sources':[]}
    monkeypatch.setattr(module,'answer_with_web',fake)
    a=make_assistant([task]);s=empty_state();s['origin']={'anchor':{'query':'A','name':'A','kind':'place'},'candidates':[]}
    new,r=a.handle('水多少度沸腾？',s)
    assert len(calls)==1 and '100' in r['answer'] and new['origin']==s['origin']
    assert new['history'][-1]['answer']==r['answer']

def test_general_generate_false_makes_no_call(monkeypatch):
    import hk_taxi.web_fallback as module
    monkeypatch.setattr(module,'answer_with_web',lambda *a:pytest.fail('must not call'))
    a=make_assistant(['general']);a.handle('hello',empty_state(),generate=False)

def test_general_task_allowed_by_router_schema():
    a=make_assistant(['general']);p,_=a.route()
    schema=json.loads((ROOT/'prompts/unified/router.schema.json').read_text(encoding='utf-8'))
    assert validate_plan(json.dumps(p),schema)['tasks']==['general']

def test_short_followup_sends_recent_qa_not_routes():
    captured=[]
    d=response('伦敦。')
    api=SimpleNamespace(create=lambda **kwargs:(captured.append(kwargs) or SimpleNamespace(model_dump=lambda **kw:d)))
    g=SimpleNamespace(settings=SimpleNamespace(model='same-model',max_tokens=1500,temperature=.1),client=SimpleNamespace(responses=api))
    s=empty_state();s['history']=[{'question':'法国首都？','answer':'巴黎。'}];s['routes']={'coordinates':'private-route-data'}
    answer_with_web(g,'那英国呢？',s,'zh-Hans')
    payload=json.loads(captured[0]['input'][1]['content'])
    assert payload['recent_turns'][0]['answer']=='巴黎。'
    assert 'coordinates' not in captured[0]['input'][1]['content']

@pytest.mark.parametrize('status,fail,expected_calls',[('answered',False,0),('insufficient_evidence',False,1),('partial_answer',True,1)])
def test_local_coverage_and_web_failure_do_not_lose_local_answer(monkeypatch,status,fail,expected_calls):
    import hk_taxi.web_fallback as module
    import hk_taxi.structured_guard as guard
    calls=[]
    def web(*args):
        calls.append(args)
        if fail:raise GenerationAPIError('联网暂不可用')
        return {'answer':'司机不得在车内吸烟。','sources':[]}
    monkeypatch.setattr(module,'answer_with_web',web)
    parsed={'status':status,'answer_parts':[{'text':'本地收费答案。','source_ids':['S1']}],'limitations':[],'clarifying_questions':[]}
    monkeypatch.setattr(guard,'protect_answer',lambda *args:{'parsed':parsed,'rendered_answer':'本地收费答案。','sources':[]})
    a=make_assistant(['rules'])
    generated={'parsed':parsed,'validation_errors':[],'rendered_answer':'本地收费答案。'}
    rag=SimpleNamespace(schema={},prepare_vectors=lambda *args:({},0),retrieve=lambda *args:({'hits':[]},[]),generate_from_sources=lambda *args:deepcopy(generated))
    a.get_rag=lambda:rag
    _,r=a.handle('规则问题',empty_state())
    assert len(calls)==expected_calls
    if fail or status=='answered':assert '本地收费答案' in r['answer']
    else:assert '司机不得' in r['answer'] and '本地收费答案' not in r['answer']
