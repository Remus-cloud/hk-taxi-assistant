"""Web runtime selection, session isolation and citations; external APIs stubbed."""
from pathlib import Path
from copy import deepcopy
import json
from hk_taxi.web_chat_passenger_v2 import WebChatSession
from hk_taxi.passenger_tools_v2 import upgrade_state
from test_web_chat import FakeEngine
R=Path(__file__).resolve().parents[1]

def test_default_web_factory_is_lazy_and_selects_pretrained(monkeypatch,tmp_path):
    import hk_taxi.web_chat_passenger_v2 as module
    calls=[];engine=FakeEngine()
    def factory(root,reranker):calls.append((root,reranker));return engine
    monkeypatch.setattr(module,'UnifiedAssistant',factory)
    (tmp_path/'configs').mkdir();(tmp_path/'configs/web_runtime_v2.json').write_text(json.dumps({'reranker':'pretrained'}),encoding='utf-8')
    chat=WebChatSession(tmp_path)
    assert not calls and chat._engine is None and chat.state['version']=='unified-passenger-v2'
    assert chat.ask('支付方式？')['success'] and calls==[(tmp_path,'pretrained')]
    chat.ask('那收据呢？');assert len(calls)==1 and len(engine.calls)==2

def test_new_conversation_keeps_v2_schema_reuses_loaded_engine(tmp_path):
    engine=FakeEngine();chat=WebChatSession(tmp_path,lambda:engine);chat.ask('first')
    assert chat.new_chat() and chat.state['conditions']['service_mode']=='unknown'
    assert chat.state['version']=='unified-passenger-v2' and chat._engine is engine and not chat.messages

def test_page_uses_passenger_adapter_without_requests_on_rerun(monkeypatch,tmp_path):
    import hk_taxi.web_chat as old
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(old,'page_status',lambda *a,**k:{'configured':True})
    engine=FakeEngine();chat=WebChatSession(tmp_path,lambda:engine)
    app=AppTest.from_file(str(R/'streamlit_app.py'),default_timeout=20)
    app.session_state['taxi_chat']=chat;app.run()
    assert not app.exception and not engine.calls
    app.segmented_control[0].set_value('English').run();app.run()
    assert not engine.calls
    app.chat_input(key='question').set_value('Can I get a receipt?').run()
    assert not app.exception and len(engine.calls)==1 and engine.calls[0][2]=='en'
    app.run();assert len(engine.calls)==1
    app.button(key='new_chat').click().run()
    assert chat.state['version']=='unified-passenger-v2' and not chat.messages and len(engine.calls)==1

def test_facility_sources_are_not_shown_as_zero_dollar_fares(monkeypatch,tmp_path):
    import hk_taxi.web_chat as old
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(old,'page_status',lambda *a,**k:{'configured':True})
    chat=WebChatSession(tmp_path,lambda:FakeEngine())
    record=dict(answer='西九龙B2候车区 [P1]',citations=[dict(id='P1',type='facility',facility={'name':'西九龙B2候车区'},source_url='https://www.highspeed.mtr.com.hk/tc/guide/wek.html')])
    chat.messages=[dict(role='assistant',content=record['answer'],record=record)]
    app=AppTest.from_file(str(R/'streamlit_app.py'),default_timeout=20);app.session_state['taxi_chat']=chat;app.run()
    assert not app.exception
    captions=[c.value for c in app.caption]
    assert any('P1 · 西九龙B2候车区' in c for c in captions)
    assert not any('P1' in c and 'HK$' in c for c in captions)

