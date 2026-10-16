"""Explicit follow-up state and conservative repairs for slot-shaped short replies."""
from copy import deepcopy
import re,json
from .multi_journey import norm

def repair(question,state,plan):
    p=deepcopy(plan);pending=state.get('pending_followup');audit={'changed':False}
    if not pending or pending.get('type')!='journey_access' or p['action'] in {'new','reverse'}:return p,audit
    q=norm(question)
    if not re.search(r'落客|上客|下车|上车|入口|出口|dropoff|pickup',q):return p,audit
    # Regulatory questions are not destination updates merely because they
    # contain the words "drop-off".
    if re.search(r'规定|规则|违法|合法|必须|是否|能不能|可以随便|可不可以|allowed|legal|must',q):
        no_updates=(p['action']=='keep' and not any(p.get(k) for k in ['origin','destination','origin_choice','destination_choice','via','access_update'])
                    and not any(v is not None for v in p['conditions'].values()))
        if no_updates and 'rules' in p['tasks'] and 'journey' in p['tasks'] and not re.search(r'多少钱|费用|估算|重算|fare|cost|price',q):
            p['tasks'].remove('journey');return p,{'changed':True,'reason':'regulatory question without trip update'}
        return p,audit
    side='destination' if re.search(r'落客|下车|dropoff',q) else 'origin' if re.search(r'上客|上车|pickup',q) else None
    if not side:return p,audit
    endpoint=state.get(side)
    if not endpoint:return p,audit
    names=[endpoint['anchor']['query'],endpoint['anchor']['name']]+[c['name'] for c in endpoint['candidates']]
    related=any(len(norm(n))>=2 and (norm(n) in q or q in norm(n)) for n in names)
    if not related and not re.match(r'^(就|那|这个|這個|那个|那個|the|that)',q):return p,audit
    # Keep other genuinely requested intents; remove only the invented generic
    # pickup/drop-off-regulations subquestion from a short location fragment.
    mixed=bool(re.search(r'收据|支付|行李|抽烟|receipt|payment|baggage|smok',q))
    if not mixed:p['tasks']=[t for t in p['tasks'] if t!='rules'];p['rag_query']=''
    if 'journey' not in p['tasks']:p['tasks'].insert(0,'journey')
    p.update(action='update',access_update=side,clarification='')
    p[side]=None  # Current place retained; candidate choice is a separate field.
    if not re.search(r'总站|總站|的士站|路|道|出口|入口|taxistand|road|street|exit|entrance',q):
        p[side+'_choice']=None  # A generic access fragment is not confirmation of a specific stop.
    audit={'changed':p!=plan,'side':side,'reason':'short access reply to active journey'}
    return p,audit

def recover_rule_query(raw,question):
    try:data=json.loads(raw)
    except (ValueError,TypeError):return raw,False
    if isinstance(data,dict) and isinstance(data.get('tasks'),list) and 'rules' in data['tasks'] and isinstance(data.get('rag_query'),str) and not data['rag_query'].strip():
        data['rag_query']=question
        return json.dumps(data,ensure_ascii=False),True
    return raw,False

def update_pending(state,plan,answer,results):
    state['last_assistant_answer']=answer[-1800:]
    if 'journey' in plan['tasks'] and any(r.get('estimate') for r in results):
        state['pending_followup']={'type':'journey_access','sides':['origin','destination'],
            'origin':state['origin']['anchor']['query'],'destination':state['destination']['anchor']['query']}

def access_ack(state,plan,language):
    side=plan.get('access_update')
    if side not in {'origin','destination'}:return ''
    endpoint=state.get(side)
    if not endpoint:return ''
    candidate=next((c for c in endpoint['candidates'] if c['id']==endpoint.get('selected_id')),None)
    location=(candidate or {}).get('name') or endpoint['anchor']['name']+'附近'
    if language=='en':return ('Keeping the current origin; the drop-off reference is ' if side=='destination' else 'Keeping the current destination; the pickup reference is ')+location+'.'
    return ('沿用当前起点，落客点按' if side=='destination' else '沿用当前终点，上客点按')+location+'估算。'
