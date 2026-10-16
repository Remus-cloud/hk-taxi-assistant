import json
from pathlib import Path
import pytest
from hk_taxi.rag_answer import select_language,validate_answer,render_answer,build_user_message
R=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('question,requested,expected',[
 ('小背包要收费吗？',None,'zh-Hans'),('小背包要收費嗎？',None,'zh-Hant'),
 ('Can I get a receipt?',None,'en'),('請用簡體中文回答收據問題',None,'zh-Hans'),
 ('用中文问，please answer in English',None,'en'),('小背包收费吗？','zh-Hant','zh-Hant'),
 ('123',None,'zh-Hans')])
def test_language_policy(question,requested,expected):assert select_language(question,requested)==expected

def sources():return [{'source_id':'S1','chunk_id':'internal-do-not-send','title':'車費收據','source_url':'https://www.td.gov.hk/receipt','text':'打印機失靈，應發出手寫收據。'}]

def answer(sid='S1'):return {'status':'answered','answer_parts':[{'text':'应提供手写收据。','source_ids':[sid]}],'clarifying_questions':[],'limitations':[]}

def schema():return json.loads((R/'prompts/rag_answer_v1.schema.json').read_text(encoding='utf-8'))

def test_unknown_reference_rejected():
    _,errors=validate_answer(json.dumps(answer('S99')),sources(),schema())
    assert 'unknown_source:S99' in errors

def test_fenced_json_not_silently_repaired():
    _,errors=validate_answer('```json\n'+json.dumps(answer())+'\n```',sources(),schema())
    assert errors==['invalid_json']

def test_model_generated_url_rejected():
    data=answer();data['answer_parts'][0]['text']='See https://untrusted.example'
    _,errors=validate_answer(json.dumps(data),sources(),schema());assert any(e.startswith('generated_url') for e in errors)

def test_renderer_uses_program_source_mapping():
    data,errors=validate_answer(json.dumps(answer()),sources(),schema());assert not errors
    text=render_answer(data,sources(),'zh-Hans')
    assert '[S1](https://www.td.gov.hk/receipt)' in text
    assert '車費收據' in text

def test_prompt_does_not_send_internal_ids_or_gold():
    template=(R/'prompts/rag_answer_v1.user.template.txt').read_text(encoding='utf-8')
    message=build_user_message(template,'receipt?',sources(),'en')
    assert 'internal-do-not-send' not in message and 'gold_evidence' not in message
    assert 'English (en)' in message and '打印機失靈' in message

def test_schema_rejects_unexpected_fields():
    data=answer();data['invented_url']='https://untrusted.example'
    _,errors=validate_answer(json.dumps(data),sources(),schema());assert errors
