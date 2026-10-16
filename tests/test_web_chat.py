from copy import deepcopy
from pathlib import Path
import json
import pytest
from hk_taxi.web_chat import WebChatSession,split_answer,trip_summary,safe_source_url
from hk_taxi.unified import empty_state

ROOT=Path(__file__).resolve().parents[1]
class FakeEngine:
    def __init__(self):self.calls=[];self.closed=False
    def close(self):self.closed=True
    def handle(self,question,state,language=None):
        self.calls.append((question,deepcopy(state),language))
        if question=='fail':state['last_user_question']='should not commit';raise RuntimeError('secret-provider-detail')
        state['last_user_question']=question
        record={'answer':'应提供手写收据。 [S1](https://www.td.gov.hk/example)\n\n来源（官方原文）：\n\n- [S1 · 車費收據](https://www.td.gov.hk/example)',
                'status':'processed','retrieval':{'sources':[{'source_id':'S1','title':'車費收據','source_url':'https://www.td.gov.hk/example','text':'打印機失靈應發出手寫收據。','verified_as_of':'2026-10-15'}]},'citations':[]}
        return state,record

def session(tmp_path):
    engine=FakeEngine();chat=WebChatSession(tmp_path,lambda:engine);return chat,engine

def test_lazy_initialization_and_export_do_not_call_engine(tmp_path):
    calls=[];chat=WebChatSession(tmp_path,lambda:calls.append(True))
    assert chat.export_text()=='' and not calls
def test_multi_turn_state_inheritance_and_persistence(tmp_path):
    chat,engine=session(tmp_path)
    assert chat.ask('first','zh-Hans')['success']
    assert chat.ask('second','zh-Hant')['success']
    assert len(engine.calls)==2 and engine.calls[1][1]['last_user_question']=='first'
    assert engine.calls[1][2]=='zh-Hant'
    path=tmp_path/'artifacts/web_sessions'/(chat.session_id+'.json')
    assert json.loads(path.read_text(encoding='utf-8'))['state']['last_user_question']=='second'
    assert len(chat.messages)==4
def test_failure_rolls_back_context_and_sanitizes_unknown_errors(tmp_path):
    chat,engine=session(tmp_path);chat.ask('first');before=deepcopy(chat.state)
    r=chat.ask('fail')
    assert not r['success'] and chat.state==before
    assert 'secret-provider-detail' not in json.dumps(chat.messages)
    assert len(engine.calls)==2
def test_save_failure_does_not_repeat_paid_call(tmp_path):
    (tmp_path/'artifacts').write_text('not a directory')
    chat,engine=session(tmp_path);result=chat.ask('question')
    assert result['success'] and chat.last_save_warning and len(engine.calls)==1
    assert len(chat.messages)==2
def test_new_chat_clears_context_but_reuses_model_instance(tmp_path):
    chat,engine=session(tmp_path);chat.ask('first');oldid=chat.session_id
    assert chat.new_chat() and not chat.messages and chat.state==empty_state()
    assert chat.session_id!=oldid and chat._engine is engine
    assert (tmp_path/'artifacts/web_sessions'/(oldid+'.json')).exists()
def test_sessions_do_not_share_state(tmp_path):
    a,ae=session(tmp_path);b,be=session(tmp_path);a.ask('first')
    assert b.state==empty_state() and b.messages==[] and a.session_id!=b.session_id
def test_busy_session_does_not_issue_second_call(tmp_path):
    chat,engine=session(tmp_path);chat._lock.acquire()
    try:assert not chat.ask('second')['success'] and not engine.calls
    finally:chat._lock.release()
@pytest.mark.parametrize('question,language',[(' ',None),('question','invalid')])
def test_invalid_submission_has_no_side_effects(tmp_path,question,language):
    chat,engine=session(tmp_path)
    with pytest.raises(ValueError):chat.ask(question,language)
    assert not engine.calls and not chat.messages
def test_source_footer_collapsed_without_losing_inline_citation(tmp_path):
    chat,_=session(tmp_path);r=chat.ask('question')['record']
    body,sources=split_answer(r)
    assert '[S1]' in body and '来源（官方原文）' not in body and sources[0]['title']=='車費收據'
def test_guard_added_sources_have_precedence():
    r={'answer':'条件。 [S7](https://example.test)','retrieval':{'sources':[]},'structured_guard':{'sources':[{'source_id':'S7','title':'规则'}]}}
    assert split_answer(r)[1][0]['source_id']=='S7'
def test_unsafe_source_links_rejected():
    assert safe_source_url('javascript:alert(1)') is None
    assert safe_source_url('https://user:key@example.test') is None
    assert safe_source_url('https://www.td.gov.hk/path')=='https://www.td.gov.hk/path'
def test_trip_summary_uses_selected_candidate():
    state=empty_state();state['origin']={'anchor':{'name':'图书馆'},'selected_id':'a','candidates':[{'id':'a','name':'摩顿台'}]}
    assert trip_summary(state)[0]=='摩顿台'

def app_with_fake(monkeypatch,tmp_path):
    from streamlit.testing.v1 import AppTest
    import hk_taxi.web_chat as module
    monkeypatch.setattr(module,'page_status',lambda *a,**k:{'configured':True,'verified_as_of':'2026-10-15','fare_outdated':True})
    chat,engine=session(tmp_path)
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=20)
    app.session_state['taxi_chat']=chat;app.run()
    return app,chat,engine
def test_app_load_language_switch_and_rerun_have_zero_calls(monkeypatch,tmp_path):
    app,chat,engine=app_with_fake(monkeypatch,tmp_path)
    assert not app.exception and not engine.calls
    app.segmented_control[0].set_value('繁體').run()
    app.run();assert not app.exception and not engine.calls
def test_suggestion_prefills_only(monkeypatch,tmp_path):
    app,chat,engine=app_with_fake(monkeypatch,tmp_path)
    app.button(key='suggest_乘车与收费').click().run()
    assert not engine.calls and app.chat_input(key='question').value=='市区红的起步多少钱？'
def test_chat_submission_once_and_source_expander(monkeypatch,tmp_path):
    app,chat,engine=app_with_fake(monkeypatch,tmp_path)
    app.chat_input(key='question').set_value('打印机坏了还能拿收据吗？').run()
    assert not app.exception and len(engine.calls)==1 and len(chat.messages)==2
    assert any('来源与计价依据' in e.label for e in app.expander)
    app.run();assert len(engine.calls)==1
    app.chat_input(key='question').set_value('再问一次').run()
    assert len(engine.calls)==2 and engine.calls[1][1]['last_user_question']=='打印机坏了还能拿收据吗？'
def test_new_chat_button_does_not_query(monkeypatch,tmp_path):
    app,chat,engine=app_with_fake(monkeypatch,tmp_path)
    app.chat_input(key='question').set_value('first').run()
    app.button(key='new_chat').click().run()
    assert not app.exception and not chat.messages and len(engine.calls)==1
def test_ui_error_does_not_expose_stack_or_repeat(monkeypatch,tmp_path):
    app,chat,engine=app_with_fake(monkeypatch,tmp_path)
    app.chat_input(key='question').set_value('fail').run()
    assert not app.exception and len(engine.calls)==1
    assert all('secret-provider-detail' not in m.value for m in app.markdown)
    app.run();assert len(engine.calls)==1
