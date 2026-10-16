"""Unified planning, explicit session patches, frozen RAG and bounded journey tools."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone,timedelta
from pathlib import Path
import hashlib,json,re
from jsonschema import Draft202012Validator
from .generation import BailianGenerator,GenerationSettings,GenerationAPIError
from .multi_journey import MapClient,MapError,discover,estimate_pairs,route_pairs,norm
from .fare_calculator import load_distance_rule
from .rag_answer import FrozenRagSession,select_language,validate_answer,render_answer


def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()


class BudgetError(RuntimeError): pass
class PlanError(ValueError): pass


class Budget:
    """Durable reservation before a request; failures count and cannot silently retry."""
    def __init__(self,path,limits):
        self.path=Path(path);self.limits=limits
        self.counts=json.loads(self.path.read_text(encoding='utf-8'))['counts'] if self.path.exists() else {k:0 for k in limits}
    def reserve(self,kind):
        if self.counts[kind]>=self.limits[kind]: raise BudgetError(kind+' budget exhausted')
        self.counts[kind]+=1
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps({'limits':self.limits,'counts':self.counts},indent=2),encoding='utf-8')


def empty_state():
    return {'version':'unified-session-v1','origin':None,'destination':None,
      'conditions':{'taxi_type':'urban','luggage':'unknown'},'routes':{},'history':[],
      'last_user_question':None,'updated_at':None,'pending_clarification':None,
      'via':[],'pending_followup':None,'last_assistant_answer':None}


def compact_state(state):
    result={'conditions':state['conditions'],'last_user_question':state.get('last_user_question'),'pending_clarification':state.get('pending_clarification')}
    for side in ['origin','destination']:
        endpoint=state.get(side)
        result[side]=None if endpoint is None else {
          'query':endpoint['anchor']['query'],'name':endpoint['anchor']['name'],
          'kind':endpoint['anchor']['kind'],'selected_id':endpoint.get('selected_id'),
          'selection_basis':endpoint.get('selection_basis'),
          'candidates':[{k:c[k] for k in ['id','name','kind']} for c in endpoint['candidates']]}
    result['pending_followup']=state.get('pending_followup')
    result['last_assistant_answer']=(state.get('last_assistant_answer') or '')[-1200:]
    result['recent_turns']=[{'question':h['question'],'answer':h.get('answer','')[:800]} for h in state.get('history',[])[-2:] if 'question' in h]
    result['via']=[{'query':e['anchor']['query'],'name':e['anchor']['name']} for e in state.get('via',[])]
    return result


def validate_plan(raw,schema):
    try: data=json.loads(raw)
    except json.JSONDecodeError: raise PlanError('router_invalid_json') from None
    errors=list(Draft202012Validator(schema).iter_errors(data))
    if errors: raise PlanError('router_schema:'+','.join(e.validator for e in errors))
    if 'rules' in data['tasks'] and not data['rag_query'].strip(): raise PlanError('missing_rule_query')
    if data['action']=='clarify' and not data['clarification'].strip(): raise PlanError('missing_clarification')
    if data['action']=='reverse' and (data['origin'] or data['destination']): raise PlanError('reverse_with_new_endpoints')
    if not data['tasks']:
        valid=(data['action']=='update' and data['origin'] is None and data['destination'] is None
          and data['origin_choice'] is None and data['destination_choice'] is None
          and (data['conditions']['luggage'] is not None or data['conditions'].get('boarding_type') is not None) and data['conditions']['taxi_type'] is None
          and not data['rag_query'] and not data['clarification'])
        if not valid or data.get('via') or data.get('access_update'): raise PlanError('empty_tasks_without_safe_condition_update')
        if data['conditions'].get('boarding_type') is not None: data['tasks']=['journey']
    return data


def choose(endpoint,hint):
    if not hint: return endpoint
    matches=[c for c in endpoint['candidates'] if c['id']==hint or norm(hint) in norm(c['name'])]
    if len(matches)!=1: raise PlanError('候选位置无法唯一确定，请补充道路名称或候选编号。')
    endpoint=deepcopy(endpoint)
    endpoint.update(selected_id=matches[0]['id'],selection_basis='user_explicit')
    return endpoint


def active(endpoint):
    if not endpoint: return endpoint
    if endpoint.get('selected_id'):
        return dict(endpoint,candidates=[c for c in endpoint['candidates'] if c['id']==endpoint['selected_id']])
    return endpoint


def route_key(a,b,as_of,via=None):
    data=[a['id'],a['coordinate'],b['id'],b['coordinate'],as_of,'urban-distance-20240714-v1']
    if via:data.append([(p['id'],p['coordinate']) for p in via])
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def explicit_stand_endpoint(query,stands):
    """Resolve a uniquely named official stand from a compound venue+stand expression."""
    if not re.search(r'的士站|出租[车車]站|taxi\s*stand',query,re.I): return None
    q=norm(re.sub(r'(的士站|出租[车車]站|taxi\s*stand)\s*$','',query,flags=re.I))
    matches=[]
    for stand in stands:
        if '市區' not in stand.get('status_tc','') or '過海' in stand.get('status_tc',''): continue
        names=[norm(stand['location_tc']),norm(stand.get('location_en',''))]
        if any(n and q.endswith(n) for n in names): matches.append(stand)
    if len(matches)!=1: return None
    s=matches[0];identifier='stand-'+str(s['stand_id']);coord=[s['longitude'],s['latitude']]
    return {'anchor':{'query':query,'name':s['location_tc'],'name_en':s.get('location_en',''),
          'coordinate':coord,'kind':'place','source':'official_taxi_stands'},
      'candidates':[{'id':identifier,'name':s['location_tc']+'（官方市区的士站）','coordinate':coord,
          'kind':'official_taxi_stand','source_url':'https://data.gov.hk/en-data/dataset/hk-td-tis_37-taxi-stands',
          'anchor_offset_m':0,'curb_permission_verified':False}],
      'selected_id':identifier,'selection_basis':'user_explicit_stand_name'}


class LiveBackend:
    def __init__(self,root,config,budget=None,resolver=None):
        self.root=Path(root);self.config=config;self.budget=budget
        self.stands=json.loads((self.root/'data/processed/taxi_stands.json').read_text(encoding='utf-8'))
        self.client=MapClient(root,config)
        self.client.place_resolver=resolver
    def discover(self,query,kind):
        return explicit_stand_endpoint(query,self.stands) or discover(self.client,query,kind,self.stands)
    def same_place(self,query,endpoint):
        return norm(query) in {norm(endpoint['anchor']['query']),norm(endpoint['anchor']['name']),norm(endpoint['anchor'].get('name_en',''))}
    def route(self,a,b,via=None):
        if self.budget: self.budget.reserve('ors')
        self.client.ors_calls=0  # Per-request maximum enforced by orchestrator; budget remains durable.
        result=self.client.route(a,b,via=via) if via else self.client.route(a,b)
        return dict(result,source_mode='live',requested_at=now())




def apply_plan(state,plan,backend):
    result=deepcopy(state)
    if not plan['tasks']:
        if not result.get('origin') or not result.get('destination'): raise PlanError('请先提供行程，再更新行李条件。')
        result['conditions']['luggage']=plan['conditions']['luggage']
        return result
    # Rule-only questions cannot silently rewrite active journey slots.
    if 'journey' not in plan['tasks']: return result
    if plan['action']=='new':
        result.update(origin=None,destination=None,via=[],conditions={'taxi_type':'urban','luggage':'unknown'})
    if plan['action']=='reverse':
        for side in ['origin','destination']:
            endpoint=result.get(side)
            if not endpoint or (not endpoint.get('selected_id') and len(endpoint['candidates'])!=1):
                raise PlanError('当前有多个位置候选，请先选定要反转的那一组起终点。')
        result['origin'],result['destination']=result['destination'],result['origin']
        # Boarding exemptions describe the old pickup, not the passenger.
        if result['conditions'].get('boarding_type')=='cross_harbour_stand':
            result['conditions']['boarding_type']='ordinary'
    for side in ['origin','destination']:
        patch=plan[side]
        if patch is not None:
            old=result.get(side)
            same_place=old is not None and backend.same_place(patch['query'],old)
            if old is None or not same_place or (patch['kind']!='auto' and patch['kind']!=old['anchor']['kind']):
                try: result[side]=backend.discover(patch['query'],patch['kind'])
                except (MapError,ValueError) as exc:
                    raise type(exc)(('起点' if side=='origin' else '终点')+'“'+patch['query']+'”：'+str(exc)) from None
                if side=='origin' and old is not None and not same_place and result['conditions'].get('boarding_type')=='cross_harbour_stand':
                    result['conditions']['boarding_type']='ordinary'
        if plan[side+'_choice']:
            if result.get(side) is None: raise PlanError('请先提供'+side+'地点。')
            result[side]=choose(result[side],plan[side+'_choice'])
    if plan.get('via') is not None:
        result['via']=[backend.discover(v['query'],v['kind']) for v in plan['via']]
    for field,value in plan['conditions'].items():
        if value is not None: result['conditions'][field]=value
    return result


def execute_journey(state,backend,root,as_of):
    if state['conditions']['taxi_type']!='urban':
        return [],'当前计算器仅支持市区的士；其他类别可先查询规则，暂不套用市区费率。'
    if not state.get('origin') or not state.get('destination'):
        return [],'请补充'+('起点' if not state.get('origin') else '终点')+'，再提供有依据的费用估算。'
    origin,destination=active(state['origin']),active(state['destination'])
    via=[active(v)['candidates'][0] for v in state.get('via',[])]
    pairs=route_pairs(origin,destination)
    if not pairs: return [],'没有找到可用的道路接入候选，请补充具体地点。'
    rule=load_distance_rule(Path(root)/'data/processed/hk_taxi.duckdb')
    # Check the actual tariff start date; review timestamps are audit metadata only.
    from .fare_calculator import calculate_distance
    calculate_distance(1,rule,as_of)
    results=[]
    cap=min(3,getattr(backend,'turn_limit',3))
    for a,b in pairs[:cap]:
        key=route_key(a,b,as_of,via)
        old=state['routes'].get(key)
        if old:
            result=dict(deepcopy(old),reused_session_result=True)
        else:
            try:
                result=backend.route(a,b,via=via) if via else backend.route(a,b)
                result.pop('estimate',None)
                if not result['review_flags']:
                    from .fare_calculator import calculate_distance
                    result['estimate']=calculate_distance(result['distance_m'],rule,as_of)
                state['routes'][key]=deepcopy(result)
            except (MapError,BudgetError,ValueError) as exc:
                results.append({'origin_id':a['id'],'destination_id':b['id'],'status':'tool_error','error':str(exc)})
                break
        if result.get('estimate'):
            from .harbour_tolls import quote
            result['toll_quote']=quote(root,result,state)
        results.append(result)
    # Carry the exact displayed unchanged endpoint; never expand it merely because the other side was selected.
    for side in ['origin','destination']:
        endpoint=state[side]
        index=0 if side=='origin' else 1
        displayed_ids={pair[index]['id'] for pair in pairs[:cap]}
        if len(displayed_ids)==1 and not endpoint.get('selected_id'):
            endpoint.update(selected_id=next(iter(displayed_ids)),selection_basis='system_displayed_endpoint_assumption')
    return results,''


def journey_text(state,results,note,language='zh-Hans'):
    from .structured_guard import script_text
    quotes=[r for r in results if r.get('estimate')]
    english=language=='en'
    if not quotes:
        if note and note.startswith(('请','当前','没有找到','未找到','同名地点','起点','终点','已找到')):
            message=note
        elif note and 'predates the effective' in note:
            message='The configured tariff does not cover that historical date.' if english else '当前这套收费规则不适用于你查询的历史日期。'
        elif any('HTTP 429' in r.get('error','') for r in results):
            message='The route service is rate-limited. Please try again shortly.' if english else '路线服务暂时限流，请稍后再试。'
        elif any('timeout' in r.get('error','').lower() or 'Network/TLS' in r.get('error','') for r in results):
            message='The map service did not respond. Please try again later.' if english else '地图服务暂时没有响应，请稍后再试。'
        elif any(r.get('review_flags') for r in results):
            message='A route was returned, but its road access needs checking. Please try another entrance.' if english else '已查到路线，但道路接入点存在异常；可改用附近街口或入口再估算。'
        else:
            message='The route could not be found. Please give more specific locations or try again later.' if english else '暂时没查到可用路线，请补充更具体的起终点，或稍后再试。'
        return script_text(message,language),[]
    cents=[r.get('toll_quote',{}).get('subtotal_cent',r['estimate']['fare_cent']) for r in quotes]
    low,high=min(cents)/100,max(cents)/100
    price=f'HK${low:.2f}' if low==high else f'HK${low:.2f}–{high:.2f}'
    lines=[('Estimated fare: ' if english else '参考车费：')+price];registry=[]
    def label(name):
        return name.replace('（官方市区的士站）','的士站').replace(' / 车行通道接入点','附近').replace(' / 路线参考点','附近')
    for i,r in enumerate(quotes,1):
        f=r['estimate'];rid,fid=f'R{i}',f'F{i}'
        tq=r.get('toll_quote') or {};total=tq.get('subtotal_cent',f['fare_cent'])/100
        via_names=[label(active(v)['candidates'][0]['name']) for v in state.get('via',[])]
        path=' → '.join([label(r['origin_name'])]+via_names+[label(r['destination_name'])])
        markers=f'[{fid}]'+(f' [T{i}]' if tq.get('crossings') else '')
        lines.append(f"{path}：{r['distance_m']/1000:.2f} km [{rid}]，HK${total:.2f} {markers}。")
        registry.extend([{'id':rid,'type':'route','source_mode':r['source_mode'],'requested_at':r['requested_at'],
                         'origin':r['origin_coordinate'],'destination':r['destination_coordinate'],'distance_m':r['distance_m']},
                        {'id':fid,'type':'calculation','rule_version':f['rule_version'],'fare_cent':f['fare_cent'],'source_url':f['source_url']}])
    for i,r in enumerate(quotes,1):
        tq=r.get('toll_quote') or {}
        if tq.get('crossings'):
            names='、'.join(t['name'] for t in tq['crossings'])
            lines.append((f"Route {i}: distance HK${tq['distance_cent']/100:.2f} + tunnel HK${tq['outbound_cent']/100:.2f} + return toll HK${tq['return_cent']/100:.2f} = HK${tq['subtotal_cent']/100:.2f}." if english else f"路线{i}：距离计程{tq['distance_cent']/100:.2f}＋隧道费{tq['outbound_cent']/100:.2f}＋回程费{tq['return_cent']/100:.2f}＝{tq['subtotal_cent']/100:.2f}港元；经过{names}。")+f" [T{i}]")
            registry.append({'id':f'T{i}','type':'calculation','label':'过海隧道附加费','fare_cent':tq['outbound_cent']+tq['return_cent'],'source_url':tq['source_url']})
    tq_all=[r.get('toll_quote') or {} for r in quotes]
    if any(t.get('return_cent') for t in tq_all):
        lines.append('Assuming an ordinary one-way hiring, including the return toll.' if english else '按普通单程、非过海的士站上车估算，已计入回程费。')
    if any(t.get('exemption')=='cross_harbour_stand' for t in tq_all):
        lines.append('Return toll waived because boarding is at a cross-harbour taxi stand; the actual tunnel toll remains.' if english else '按从过海的士站上车处理，免回程费，实际隧道费仍须支付；距离暂沿用当前代表点。')
    if any(t.get('exemption')=='same_side_final_destination' for t in tq_all):
        lines.append('The complete route finishes on the starting side of the harbour: return toll waived, with each actual crossing charged.' if english else '完整行程最终回到出发的同一岸，免回程费；实际每次穿隧的隧道费仍计入。')
    if any(t.get('status') in {'unavailable','unresolved_toll'} for t in tq_all):
        lines.append('Tunnel charges could not be confirmed and are not included in this estimate.' if english else '这条路线的隧道费用尚未确认，当前金额未计入该部分。')
    if state.get('via'):
        lines.append(('One continuous hiring via: ' if english else '同一次租用，途经：')+' → '.join(e['anchor']['name'] for e in state['via'])+'。')
    if any(r.get('route_advisories') for r in quotes):
        lines.append('The route may involve road access restrictions.' if english else '路线可能涉及通行限制。')
    lines.append(('Includes identified harbour tolls; waiting, baggage and other road charges are excluded.' if english else '已计入识别到的过海隧道费用；不含等候、行李及其他道路收费，实际以咪表及适用收费为准。') if any(t.get('crossings') for t in tq_all) else ('Distance-based estimate; waiting, baggage and unaccounted road charges are excluded.' if english else '当前为距离计程估算；不含等候、行李及未计入的道路收费，实际以咪表及适用收费为准。'))
    for side,zh,en in [('origin','起点','Origin'),('destination','终点','Destination')]:
        endpoint=state.get(side)
        if endpoint and endpoint['anchor'].get('representative'):
            a=endpoint['anchor']
            lines.append((en+' represented by: ' if english else zh+'代表地点：')+(a.get('name_en') or a['name'] if english else a['name'])+'。')
    broad=any(e and e['anchor']['kind']=='area' for e in [state.get('origin'),state.get('destination')])
    if broad or any(e and e['anchor'].get('representative') for e in [state.get('origin'),state.get('destination')]):lines.append('This rough estimate uses nearby representative locations. Do you have a specific street corner or entrance?' if english else '这是按附近代表地点给出的粗估。你有更具体的街口或入口吗？')
    return script_text('  \n'.join(lines),language),registry

class UnifiedAssistant:
    def __init__(self,root,backend=None,budget=None,as_of=None):
        self.root=Path(root);self.budget=budget
        self.as_of=as_of or datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        self.config=json.loads((self.root/'configs/unified_v1.json').read_text(encoding='utf-8'))
        self.schema=json.loads((self.root/'prompts/unified/router.schema.json').read_text(encoding='utf-8'))
        self.router_system=(self.root/'prompts/unified/router.system.txt').read_text(encoding='utf-8')
        self.mixed_system=(self.root/'prompts/unified/mixed.system.txt').read_text(encoding='utf-8')
        self.rule_system=(self.root/'prompts/unified/rules.system.txt').read_text(encoding='utf-8')
        settings=GenerationSettings.load(self.root)
        self.router=BailianGenerator(replace(settings,max_tokens=512))
        self.answer_generator=BailianGenerator(settings)
        mc=json.loads((self.root/'configs/journey_multi_v1.json').read_text(encoding='utf-8'))
        self.place_resolution_audit=[]
        self.backend=backend or LiveBackend(root,mc,budget,resolver=self.resolve_place)
        self.rag=None
    def resolve_place(self,query,kind,rows):
        from .place_resolution import propose
        if self.budget: self.budget.reserve('router')
        proposal,audit=propose(self.router,query,kind,rows)
        self.place_resolution_audit.append(audit)
        return proposal
    def route(self,question,state):
        if self.budget: self.budget.reserve('router')
        response=self.router.generate(json.dumps({'question':question,'session':compact_state(state)},ensure_ascii=False),self.router_system)
        self.last_router_audit={'raw':response.answer,'usage':response.usage,'requested_model':response.requested_model,'returned_model':response.returned_model}
        if response.truncated or response.unexpected_thinking: raise PlanError('router_truncated_or_thinking')
        from .journey_context import recover_rule_query
        raw,recovered=recover_rule_query(response.answer,question)
        return validate_plan(raw,self.schema),{'rule_query_recovered_from_user':recovered,'raw':response.answer,'usage':response.usage,
          'requested_model':response.requested_model,'returned_model':response.returned_model}
    def get_rag(self):
        if self.rag is None:
            self.rag=FrozenRagSession(self.root,with_generator=False)
            self.rag.generator=self.answer_generator
            self.rag.system=self.rule_system
        return self.rag
    def handle(self,question,state,generate=True,language=None):
        before=deepcopy(state)
        record={'question':question,'state_before':compact_state(before),'started_at':now(),
                'retrieval':None,'generation':None,'journey_results':[],'citations':[]}
        try:
            plan,route_audit=self.route(question,state)
            from .journey_context import repair
            original_plan=deepcopy(plan)
            plan,context_audit=repair(question,state,plan)
            record.update(plan=plan,router=route_audit,router_plan_before_context=original_plan,context_resolution=context_audit)
        except PlanError as exc:
            from .structured_guard import script_text
            record.update(status='planner_error',answer=script_text('无法可靠解析本轮请求，请明确起终点或要查询的规则。',select_language(question,language)),error=str(exc),router=getattr(self,'last_router_audit',None),state_after=compact_state(before))
            return before,record
        selected_language=select_language(question,language)
        resolution_start=len(getattr(self,'place_resolution_audit',[]))
        if plan['action']=='clarify':
            pending=deepcopy(before);pending.update(pending_clarification=plan['clarification'],last_user_question=question,updated_at=now())
            from .structured_guard import script_text
            record.update(status='needs_clarification',answer=script_text(plan['clarification'],selected_language),state_after=compact_state(pending))
            return pending,record
        try:
            working=apply_plan(state,plan,self.backend)
        except (PlanError,MapError,ValueError) as exc:
            # A failed endpoint update is atomic; rule evidence may still be answered.
            working=deepcopy(before);record['journey_note']=str(exc);record['slot_update_failed']=True
        record['place_resolution_audit']=getattr(self,'place_resolution_audit',[])[resolution_start:]
        text=[]
        if not plan['tasks']:
            condition=working['conditions']['luggage']
            condition='不带行李' if condition=='none' else condition
            text.append('已更新行李条件：'+condition+'；起终点保持不变。基础距离金额不因行李条件而重算。')
        if 'journey' in plan['tasks']:
            if record.get('slot_update_failed'): results,note=[],record['journey_note']
            else:
                try: results,note=execute_journey(working,self.backend,self.root,self.as_of)
                except (ValueError,MapError) as exc: results,note=[],str(exc)
            block,registry=journey_text(working,results,note,selected_language)
            from .journey_context import access_ack
            ack=access_ack(working,plan,selected_language)
            text.append((ack+'\n' if ack else '')+block);record.update(journey_results=results,citations=registry)
        rule_text_start=len(text)
        if 'rules' in plan['tasks']:
            rag=self.get_rag()
            query=question if plan['tasks']==['rules'] and not before.get('origin') and not before.get('destination') else plan['rag_query']
            vectors,_=rag.prepare_vectors([query])
            retrieval,sources=rag.retrieve(query,vectors)
            record['retrieval']={'query':query,'chunk_ids':[h['chunk']['chunk_id'] for h in retrieval['hits']],'sources':sources}
            if generate:
                if self.budget: self.budget.reserve('answer')
                if 'journey' not in plan['tasks']:
                    generated=rag.generate_from_sources(query,sources,selected_language)
                else:
                    # Numeric route blocks are rendered by code; model answers all rule subquestions.
                    message=json.dumps({'original_question':question,'rule_question':query,'sources':sources,
                        'journey_already_rendered':text,'language':selected_language},ensure_ascii=False)
                    response=self.answer_generator.generate(message,self.mixed_system)
                    parsed,errors=validate_answer(response.answer,sources,rag.schema)
                    if response.truncated: errors.append('output_truncated')
                    if response.unexpected_thinking: errors.append('unexpected_thinking')
                    generated={'raw_response':response.answer,'parsed':parsed,'validation_errors':errors,
                      'rendered_answer':render_answer(parsed,sources,selected_language) if not errors else None,
                      'usage':response.usage,'returned_model':response.returned_model}
                record['generation']=generated
                if generated.get('parsed') is not None and not generated.get('validation_errors'):
                    from .structured_guard import protect_answer
                    try:
                        guarded=protect_answer(self.root,query,generated['parsed'],sources,selected_language,rag.schema)
                        record['structured_guard']=guarded
                        generated=dict(generated,guarded_rendered_answer=guarded['rendered_answer'])
                        text.append(guarded['rendered_answer'] or '暂时无法确认这项收费，请换个说法或稍后再试。')
                    except Exception as exc:
                        record['structured_guard']={'error_type':type(exc).__name__,'failed_closed':True}
                        text.append('暂时无法确认这项收费，请稍后再试。')
                else:
                    text.append(generated['rendered_answer'] or '这次没能整理出可靠的回答，请换个说法再问一次。')
            else:
                text.append('【本轮测试未安排规则生成；已记录检索证据，不作为完整问答验收。】')
        if 'live_queue' in plan['tasks']:
            text.append('当前没有实时的士排队／空车数据，不能用路线预计时间代替排队时间。')
        # Web is a fallback after local coverage assessment; ordinary questions
        # use the same model with its web tool available, without domain refusal.
        from .web_fallback import needs_web,answer_with_web
        general=bool({'general','out_of_scope'} & set(plan['tasks']))
        gap=needs_web(record.get('generation') or {},record.get('structured_guard')) if record.get('generation') else False
        if generate and (general or gap):
            local='\n\n'.join(text[rule_text_start:]) if 'rules' in plan['tasks'] else None
            try:
                if self.budget: self.budget.reserve('answer')
                # General subquestions must not inherit taxi-only rag_query text.
                web=answer_with_web(self.answer_generator,question,before,selected_language,local)
                record['web_search']=web
                if gap:
                    parsed=(record.get('generation') or {}).get('parsed') or {}
                    if parsed.get('status') in {'insufficient_evidence','out_of_scope'}:
                        del text[rule_text_start:]
                text.append(web['answer'])
            except (GenerationAPIError,BudgetError) as exc:
                record['web_search']={'status':'failed','error':str(exc)}
                text.append(str(exc))
        elif general and not generate:
            text.append('【本轮仅记录一般问题路由，未调用回答或联网工具。】')
        from .journey_context import update_pending
        update_pending(working,plan,'\n\n'.join(text),record['journey_results'])
        working['history'].append({'question':question,'answer':'\n\n'.join(text),'plan':plan,'previous_slots':compact_state(before)})
        working['last_user_question']=question;working['updated_at']=now()
        working['pending_clarification']=record.get('journey_note') if record.get('slot_update_failed') else None
        from .structured_guard import script_text
        text=[script_text(t,selected_language,record.get('retrieval',{}).get('sources',[]) if record.get('retrieval') else []) if not ('来源（官方原文）' in t or '來源（官方原文）' in t or 'Sources (original official text)' in t) else t for t in text]
        record.update(answer='\n\n'.join(text),status='processed',state_after=compact_state(working))
        return working,record
    def close(self):
        self.router.close();self.answer_generator.close()
