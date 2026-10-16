"""港的通 · local passenger-facing chat. No requests until an explicit submission."""
from pathlib import Path
import sys
import streamlit as st

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from hk_taxi.web_chat import LANGUAGES,page_status,split_answer,safe_source_url,trip_summary
from hk_taxi.web_chat_passenger_v2 import WebChatSession
from hk_taxi.journey_feedback import present_journey_failure

st.set_page_config(page_title='港的通 · 香港的士助手',page_icon=':material/local_taxi:',layout='centered',initial_sidebar_state='auto')

@st.cache_resource(scope='session',max_entries=1,show_spinner=False,on_release=lambda chat:chat.close())
def cached_chat():return WebChatSession(ROOT)

if 'taxi_chat' not in st.session_state:st.session_state.taxi_chat=cached_chat()
chat=st.session_state.taxi_chat
status=page_status(ROOT)

with st.sidebar:
    st.badge('港的通',icon=':material/local_taxi:',color='primary')
    st.caption('HONG KONG TAXI COMPANION')
    st.space('small')
    if st.button('开始新对话',icon=':material/add:',width='stretch',type='primary',key='new_chat'):
        chat.new_chat();st.session_state.question='';st.rerun()
    st.segmented_control('回答语言',list(LANGUAGES),default='跟随提问',required=True,key='reply_language',width='stretch')
    st.divider()
    origin,destination,luggage=trip_summary(chat.state)
    st.markdown('**当前行程**')
    if origin or destination:
        st.caption('起点');st.markdown(origin or '待补充')
        st.caption('终点');st.markdown(destination or '待补充')
        if luggage and luggage!='unknown':st.caption('行李：'+('不带行李' if luggage=='none' else luggage))
    else:st.caption('告诉我起点和终点，再补充具体上落客位置。')
    if chat.messages:
        st.download_button('导出本次对话',data=chat.export_text(),file_name='hk-taxi-chat.txt',mime='text/plain',icon=':material/download:',width='stretch',on_click='ignore',key='export_chat')

with st.container(horizontal=True,vertical_alignment='center'):
    st.badge('香港 · 的士出行',color='primary',icon=':material/local_taxi:')

if not chat.messages:
    st.title('下一程，心里有数。',anchor=False)
    st.markdown('查乘车规则，问支付与行李，估算参考车费。\n\n告诉我你要去哪里，也可以从一个小问题开始。')
    st.space('small')
    cols=st.columns(3,gap='small')
    suggestions=[('乘车与收费',':material/local_taxi:','红的起步多少钱？','市区红的起步多少钱？'),
                 ('支付与收据',':material/credit_card:','每辆车都能刷Visa？','有非二维码支付就保证每辆的士都接受Visa吗？'),
                 ('行李规则',':material/luggage:','小背包要加钱吗？','小背包放在乘客车厢内也一定收6元吗？')]
    for col,(title,icon,label,question) in zip(cols,suggestions):
        with col.container(border=True,height='stretch'):
            st.markdown(icon+' **'+title+'**')
            st.caption(label)
            if st.button('填入问题',key='suggest_'+title,type='tertiary',icon=':material/arrow_forward:',width='stretch'):
                st.session_state.question=question
    st.caption('示例只填入输入框；按发送后才会查询。')
else:
    st.title('港的通',anchor=False)
    st.caption('继续这一程，也可以问一个新问题。')

history=st.container()
for index,message in enumerate(chat.messages):
    with history.chat_message(message['role'],avatar=':material/local_taxi:' if message['role']=='assistant' else None):
        if message['role']=='user':st.markdown(message['content'])
        else:
            record=message.get('record',{'answer':message['content']})
            body,sources=split_answer(record);st.markdown(present_journey_failure(ROOT,record,body))
            citations=record.get('citations') or []
            if sources or citations:
                with st.expander(f'来源与计价依据 · {len(sources)+len(citations)}',icon=':material/library_books:',key='sources_'+str(index)):
                    for source in sources:
                        label=source['source_id']+' · '+source['title'];url=safe_source_url(source.get('source_url'))
                        st.markdown(f'[{label}]({url})' if url else label)
                        st.text(source.get('text',''))
                    for citation in citations:
                        if citation.get('type')=='facility':
                            facility=citation.get('facility') or {}
                            st.caption(citation['id']+' · '+facility.get('name','候车点'))
                            if url:=safe_source_url(citation.get('source_url')):st.markdown(f'[候车点来源]({url})')
                        elif citation.get('type')=='route':
                            st.caption(f"{citation['id']} · 路线距离 {citation.get('distance_m',0)/1000:.2f} km")
                        else:
                            amount=citation.get('fare_cent',0)/100
                            st.caption(f"{citation['id']} · {citation.get('label') or ('已知费用分项合计' if citation.get('components') else '基础距离金额')} HK${amount:.2f}")
                            if url:=safe_source_url(citation.get('source_url')):st.markdown(f'[官方收费依据]({url})')

if chat.last_save_warning:st.warning('回答已生成，但本地记录未保存。请先导出对话，不要为保存记录而重复发送。')
if not status['configured']:st.info('问答服务尚未配置完成，请先检查项目.env中的Key、Base URL和模型名称。',icon=':material/settings:')

prompt=st.chat_input('问一问乘车规则，或输入“从哪里到哪里”',key='question',submit_mode='disable',disabled=not status['configured'])
if prompt:
    with history.chat_message('user'):st.markdown(prompt)
    with history.chat_message('assistant',avatar=':material/local_taxi:'):
        with st.spinner('正在查询，请稍候…',show_time=True):
            chat.ask(prompt,LANGUAGES[st.session_state.reply_language])
    st.rerun()
