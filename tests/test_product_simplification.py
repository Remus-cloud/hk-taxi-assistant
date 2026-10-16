from pathlib import Path
import json
from copy import deepcopy
import pytest
from hk_taxi.fare_calculator import DistanceRule,calculate_distance
from hk_taxi.structured_guard import protect_answer,RuleStore
from hk_taxi.unified import execute_journey,journey_text
from test_unified import populated,FakeBackend
ROOT=Path(__file__).resolve().parents[1]
RULE=DistanceRule('v','2024-07-14','2026-10-15','https://www.td.gov.hk',2000,2900,200,10250,210,140)

@pytest.mark.parametrize('when',['2026-10-16','2026-11-01','2027-01-01'])
def test_review_date_is_not_an_expiry(when):
    result=calculate_distance(5000,RULE,when)
    assert result['fare_cent']==6050 and result['verified_as_of']=='2026-10-15'

def test_current_journey_reaches_route_and_returns_amount(monkeypatch):
    import hk_taxi.unified as module
    monkeypatch.setattr(module,'load_distance_rule',lambda *a:RULE)
    backend=FakeBackend();state=populated()
    rows,note=execute_journey(state,backend,ROOT,'2026-10-16')
    assert backend.calls>0 and any(r.get('estimate') for r in rows)
    text,refs=journey_text(state,rows,note)
    assert text.startswith('参考车费：HK$') and refs
    assert '核验' not in text and 'verified' not in text and '快照' not in text
    assert '等候' in text and '道路收费' in text

def test_plain_fare_question_has_no_unasked_date_notice():
    schema=json.loads((ROOT/'prompts/rag_answer_v1.schema.json').read_text(encoding='utf-8'))
    p={'status':'answered','answer_parts':[{'text':'市区起步29元。','source_ids':[]}],'clarifying_questions':[],'limitations':[]}
    r=protect_answer(ROOT,'市区红的起步多少钱？',p,[],'zh-Hans',schema)
    body=''.join(x['text'] for x in r['parsed']['answer_parts'])
    assert '29' in body and '2026-10-15' not in body and '2024-07-14' not in body and '核验' not in body

def test_date_facts_still_answered_when_explicitly_requested():
    schema=json.loads((ROOT/'prompts/rag_answer_v1.schema.json').read_text(encoding='utf-8'))
    p={'status':'answered','answer_parts':[],'clarifying_questions':[],'limitations':[]}
    r=protect_answer(ROOT,'市区起步29元与行李6元都是同一天开始的吗？',p,[],'zh-Hans',schema)
    text=''.join(x['text'] for x in r['parsed']['answer_parts'])
    assert '2024-07-14' in text and '不能确认行李' in text and '2026-10-15' not in text

def test_ui_does_not_warn_about_review_dates(monkeypatch,tmp_path):
    from test_web_chat import app_with_fake
    app,_,engine=app_with_fake(monkeypatch,tmp_path)
    assert not app.exception
    text=' '.join(m.value for m in app.markdown)+ ' '.join(c.value for c in app.caption)
    assert '待复核' not in text and '核验至' not in text and not engine.calls

def test_failed_route_has_no_false_estimate_or_internal_error():
    state=populated();text,refs=journey_text(state,[],'HTTP 429; no automatic retry.')
    assert 'HTTP' not in text and '参考车费' not in text and not refs
