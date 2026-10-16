"""Opt-in passenger v2; original UnifiedAssistant and UI defaults remain frozen."""
from pathlib import Path
from copy import deepcopy
import json
from jsonschema import Draft202012Validator
from .unified import (UnifiedAssistant as OriginalAssistant,empty_state as old_empty,apply_plan,execute_journey,
    compact_state,now,PlanError,MapError,BudgetError,GenerationAPIError,select_language,active,
    validate_answer,render_answer)
from .passenger_tools_v2 import upgrade_state,apply_condition_patch,fee_components,find_pickup_facilities,NEW_DEFAULTS
from .structured_guard import script_text

def empty_state():return upgrade_state(old_empty())

def apply_plan_v2(state,plan,backend,root):
    # Legacy endpoint/origin-bound exemption behavior is reused exactly.
    legacy=deepcopy(plan);legacy['tasks']=[t for t in plan['tasks'] if t!='facilities']
    legacy['conditions']={k:v for k,v in plan['conditions'].items() if k not in NEW_DEFAULTS}
    if not legacy['tasks']:
        legacy['tasks']=['rules']  # A facility-only query is not a luggage update.
    updated=apply_plan(upgrade_state(state),legacy,backend)
    actual_update='journey' in plan['tasks']
    if actual_update:
        updated=apply_condition_patch(updated,plan['conditions'])
        if plan['conditions'].get('luggage')=='none' and plan['conditions'].get('bags') is None:updated['conditions']['bags']=[]
        if plan['conditions'].get('luggage') not in {None,'none'} and plan['conditions'].get('bags') is None:
            # Do not leave a previous numeric baggage total attached to new vague luggage.
            updated['conditions']['bags']=None
        old=state.get('origin');new=updated.get('origin')
        changed=plan['action']=='reverse' or (old and new and not backend.same_place(old['anchor']['query'],new))
        if changed and plan['conditions'].get('pickup_facility_kind') is None:updated['conditions']['pickup_facility_kind']='unknown'
    choice=plan.get('facility_choice')
    if choice:
        available=(state.get('pickup_facilities') or {}).get('facilities',[])
        f=next((f for f in available if f['id']==choice),None)
        if f is None:raise PlanError('请从上一轮列出的站点中选择。')
        if not f.get('coordinate'):raise PlanError('该楼层设施暂只有文字指引；请补充已确认的道路入口。')
        candidate=dict(id=f['id'],name=f['name'],kind='official_taxi_stand',coordinate=f['coordinate'],source_url=f['source_url'],boarding_type='cross_harbour_stand' if f['kind']=='cross_harbour_stand' else 'ordinary')
        updated['origin']=dict(anchor=dict(query=f['name'],name=f['name'],kind='place'),candidates=[candidate],selected_id=f['id'],selection_basis='explicit_user_facility_selection')
        updated['conditions'].update(pickup_facility_kind=f['kind'],boarding_type=candidate['boarding_type'])
    return updated

def execute_journey_v2(state,backend,root,as_of):
    routes,note=execute_journey(state,backend,root,as_of)
    for r in routes:
        if r.get('estimate'):r['quote_components']=fee_components(root,r,state)
    return routes,note

def journey_text_v2(state,results,note,language='zh-Hans'):
    from .unified import journey_text
    quotes=[r for r in results if r.get('quote_components')]
    if not quotes:return journey_text(state,results,note,language)
    en=language=='en';totals=[r['quote_components']['total_known_cent'] for r in quotes]
    price=f'HK${min(totals)/100:.2f}'+(f'–{max(totals)/100:.2f}' if min(totals)!=max(totals) else '')
    basis=quotes[0]['quote_components']['price_basis']
    labels={'known_meter_components':('参考车费（已知分项）：','Estimated known components: '),'meter_reference':('同程按表参考：','Same-trip meter reference: '),'platform_agreed_total':('订单约定整笔价：','User-provided agreed total: ')}
    lines=[labels[basis][int(en)]+price];registry=[]
    fields=[('distance_cent','距离计程','Distance'),('outbound_toll_cent','隧道费','Tunnel'),('return_toll_cent','回程费','Return toll'),('luggage_cent','行李','Baggage'),('ordinary_phone_booking_cent','普通电召','Ordinary phone booking'),('animal_cent','普通宠物','Pet'),('fleet_booking_cent','车队约定预约费','Agreed fleet booking')]
    for i,r in enumerate(quotes,1):
        q=r['quote_components'];path=' → '.join([r['origin_name']]+[active(v)['candidates'][0]['name'] for v in state.get('via',[])]+[r['destination_name']])
        lines.append(f'{path}：{r["distance_m"]/1000:.2f} km [R{i}]；HK${q["total_known_cent"]/100:.2f} [F{i}]。')
        if q['price_basis']!='platform_agreed_total':lines.append(' + '.join((b if en else a)+f' HK${q[k]/100:.2f}' for k,a,b in fields if q[k] is not None and (q[k]>0 or k=='distance_cent')))
        registry.extend([dict(id=f'R{i}',type='route',source_mode=r['source_mode'],requested_at=r['requested_at'],distance_m=r['distance_m'],origin=r['origin_coordinate'],destination=r['destination_coordinate']),dict(id=f'F{i}',type='calculation',fare_cent=q['total_known_cent'],source_url=r['estimate']['source_url'],components=q)])
    allq=[r['quote_components'] for r in quotes]
    if any('ordinary_one_way_return_toll' in q['assumptions'] for q in allq):lines.append('Ordinary one-way hiring assumed; return toll included.' if en else '按普通单程、非过海站上车计算，已计回程费。')
    if any('cross_harbour_stand' in q['assumptions'] for q in allq):lines.append('Boarding at the confirmed cross-harbour stand: return toll waived.' if en else '按已确认从过海的士站上车处理，免回程费，实际隧道费仍计入。')
    if any('same_side_final_destination' in q['assumptions'] for q in allq):lines.append('Final destination on the starting side: return toll waived; actual crossings charged.' if en else '最终目的地回到出发同岸，免回程费；实际穿隧仍逐次计入。')
    if basis=='meter_reference':lines.append('The platform may use an agreed booking fee or a lump sum; this is only a meter reference.' if en else '车队预约可能按约定预约费或整笔价收费；以上只是同程按表参考。')
    if basis=='platform_agreed_total':lines.append('The agreed total is shown separately; meter and toll calculations are not added again. Check the order for included items.' if en else '整笔价按你提供的订单金额显示，不再叠加咪表和隧道费；包含项目以订单约定为准。')
    else:
        missing=set(x for q in allq for x in q['not_included'])
        lines.append('Waiting and other unconfirmed extras are excluded; actual fare follows the meter and applicable charges.' if en else '未计等候及其他尚未确定的附加费，实际以咪表及适用收费为准。')
        if 'unresolved_luggage' in missing and (state['conditions'].get('bags') is not None or state['conditions'].get('luggage') not in {None,'unknown','none'}):lines.append('Baggage fees need the item count, storage location or qualifying conditions.' if en else '部分行李条件尚不完整，该部分未计入；可补充件数、摆放位置及适用条件。')
        if 'unconfirmed_harbour_tolls' in missing:lines.append('Unconfirmed harbour tolls are not included.' if en else '隧道费尚未确认，该部分未计入。')
        if 'unresolved_pet_count' in missing:lines.append('Pet count is missing, so the pet surcharge is not included.' if en else '普通宠物数量未说明，该项附加费尚未计入。')
        if 'guide_dog_not_automatically_charged_as_pet' in missing:lines.append('No ordinary-pet fee has been applied to the guide dog; its charging position needs separate evidence.' if en else '没有把导盲犬自动按普通宠物加费；该项需另据适用资料确认。')
    if any(r.get('route_advisories') for r in quotes):lines.append('The route may involve access restrictions.' if en else '路线可能涉及通行限制。')
    if any(e and (e['anchor'].get('representative') or e['anchor']['kind']=='area') for e in [state.get('origin'),state.get('destination')]):lines.append('This uses the named representative pickup points. A specific street corner or entrance can refine it.' if en else '这是按上方列出的代表接入点给出的粗估；补充具体街口或入口可以更准。')
    return script_text('  \n'.join(lines),language),registry

def facility_text(root,state,plan,backend,language,previous_state=None):
    request=plan.get('facility_request');c=state['conditions'];mode=c.get('service_mode','unknown')
    old_mode=((previous_state or {}).get('conditions') or {}).get('service_mode','unknown')
    need_refresh=mode=='fleet_prebooked' or (old_mode in {'fleet_prebooked','ordinary_phone_booking'} and mode=='street_or_stand')
    if request is None and need_refresh and 'journey' in plan['tasks'] and plan['conditions'].get('service_mode') not in {None,'unknown'}:
        if state.get('origin'):request=dict(query=state['origin']['anchor']['query'],kind='fleet_prebooked_pickup' if mode=='fleet_prebooked' else 'ordinary_stand')
    if not request:return '',[]
    if state.get('origin') and backend.same_place(request['query'],state['origin']):endpoint=state['origin']
    else:
        try:endpoint=backend.discover(request['query'],'auto')
        except (ValueError,MapError):return script_text('请补充找站地点；已有行程保持不变。',language),[]
    place=dict(query=endpoint['anchor']['query'],name=endpoint['anchor']['name'],coordinate=active(endpoint)['candidates'][0].get('coordinate'))
    found=find_pickup_facilities(root,place,request['kind'],c['taxi_type'],mode)
    # Independent official WK ground pickup and the fleet-specific listing remain side by side.
    if request['kind']=='fleet_prebooked_pickup':
        extra=find_pickup_facilities(root,place,'prebooked_pickup',c['taxi_type'],mode)
        found['facilities'].extend(extra['facilities'])
    state['pickup_facilities']=found;en=language=='en';lines=[];refs=[]
    for i,f in enumerate(found['facilities'],1):
        lines.append(f'{f["name"]}'+(f'（{f["distance_m"]}m '+('straight line' if en else '直线距离')+'）' if f['distance_m'] is not None else '')+f' [P{i}]')
        # Preserve operating-hour qualifications from source SQL status.
        if '（' in f['selection_reason'] or '(' in f['selection_reason']:lines.append(f['selection_reason'])
        refs.append(dict(id=f'P{i}',type='facility',source_url=f['source_url'],facility=f))
    if request['kind']=='fleet_prebooked_pickup':lines.append('Use the pickup point specified in your booking; listed floors and streets are distinct facilities.' if en else '车队预约请以订单安排的上客区为准；名单中的楼层、路段是不同设施。')
    if not lines:lines.append('No verified facility was found for that place and category.' if en else '暂未找到该地点、该类别的已核验候车设施。')
    return script_text('  \n'.join(lines),language),refs

class ExpandedRag:
    def __init__(self,root,frozen,reranker='none'):
        from .passenger_corpus import PassengerCorpusSession
        self.session=PassengerCorpusSession(root,frozen_session=frozen);self.frozen=frozen;self.scorer=None
        if reranker!='none':
            from .runtime_assets import verify_model
            verify_model(root,'reranker')
            from .reranking import BgePairScorer,FrozenCandidateAdapter
            models={'pretrained':'models/bge-reranker-v2-m3/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'}
            self.scorer=BgePairScorer(Path(root)/models[reranker]);self.adapter=FrozenCandidateAdapter(self.session.retriever)
    def __getattr__(self,name):return getattr(self.frozen,name)
    def retrieve(self,question,vectors):
        if self.scorer is None:return self.session.retrieve(question,vectors)
        from .reranking import unique_pairs,select_candidates
        from .rag_answer import source_cards
        plan=self.adapter.build(question,vectors);scores,timing=self.scorer.score(unique_pairs(plan));result=select_candidates(plan,scores)
        result['reranker_audit']=timing
        return result,source_cards(result['hits'])
    def close(self):self.frozen.close()

class UnifiedAssistant(OriginalAssistant):
    def __init__(self,root,reranker='none',**kwargs):
        if reranker not in {'none','pretrained'}:raise ValueError('Unknown reranker arm')
        self.reranker_arm=reranker
        super().__init__(root,**kwargs)
        self.schema=json.loads((self.root/'prompts/passenger_v2/router.schema.json').read_text(encoding='utf-8'))
        self.router_system=(self.root/'prompts/passenger_v2/router.system.txt').read_text(encoding='utf-8')
    def get_rag(self):
        if self.rag is None:
            frozen=super().get_rag();self.rag=ExpandedRag(self.root,frozen,self.reranker_arm)
        return self.rag
    def route(self,question,state):
        if self.budget:self.budget.reserve('router')
        response=self.router.generate(json.dumps(dict(question=question,session=compact_state(state),pickup_facilities=state.get('pickup_facilities')),ensure_ascii=False),self.router_system)
        if response.truncated or response.unexpected_thinking:raise PlanError('router_truncated_or_thinking')
        from .journey_context import recover_rule_query
        raw,recovered=recover_rule_query(response.answer,question)
        try:
            plan=json.loads(raw);Draft202012Validator(self.schema).validate(plan)
            if 'rules' in plan['tasks'] and not plan['rag_query'].strip():raise ValueError('Missing rule subquery')
            if plan['action']=='clarify' and not plan['clarification'].strip():raise ValueError('Missing clarification')
            if plan['action']=='reverse' and (plan['origin'] or plan['destination']):raise ValueError('Reverse cannot replace endpoints')
            if not plan['tasks']:raise ValueError('Use journey for fee-condition updates')
            if plan.get('facility_choice') and 'journey' not in plan['tasks']:raise ValueError('Selection must update journey')
        except Exception as exc:raise PlanError(type(exc).__name__) from None
        return plan,dict(raw=response.answer,usage=response.usage,rule_query_recovered_from_user=recovered,requested_model=response.requested_model,returned_model=response.returned_model)

    def handle(self,question,state,generate=True,language=None):
        state=upgrade_state(state)
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
            working=apply_plan_v2(state,plan,self.backend,self.root)
        except (PlanError,MapError,ValueError) as exc:
            # A failed endpoint update is atomic; rule evidence may still be answered.
            working=deepcopy(before);record['journey_note']=str(exc);record['slot_update_failed']=True
        facility_block,facility_citations=('',[]) if record.get('slot_update_failed') else facility_text(self.root,working,plan,self.backend,selected_language,before)
        record['facilities']=working.get('pickup_facilities')
        record['place_resolution_audit']=getattr(self,'place_resolution_audit',[])[resolution_start:]
        text=[]
        if not plan['tasks']:
            condition=working['conditions']['luggage']
            condition='不带行李' if condition=='none' else condition
            text.append('已更新行李条件：'+condition+'；起终点保持不变。基础距离金额不因行李条件而重算。')
        if 'journey' in plan['tasks']:
            if record.get('slot_update_failed'): results,note=[],record['journey_note']
            else:
                try: results,note=execute_journey_v2(working,self.backend,self.root,self.as_of)
                except (ValueError,MapError) as exc: results,note=[],str(exc)
            block,registry=journey_text_v2(working,results,note,selected_language)
            from .journey_context import access_ack
            ack=access_ack(working,plan,selected_language)
            text.append((ack+'\n' if ack else '')+block);record.update(journey_results=results,citations=registry)
        if facility_block:
            text.append(facility_block);record['citations'].extend(facility_citations)
        rule_text_start=len(text)
        if 'rules' in plan['tasks']:
            rag=self.get_rag()
            query=question if plan['tasks']==['rules'] and not before.get('origin') and not before.get('destination') else plan['rag_query']
            vectors,_=rag.prepare_vectors([query])
            retrieval,sources=rag.retrieve(query,vectors)
            record['retrieval']={'query':query,'chunk_ids':[h['chunk']['chunk_id'] for h in retrieval['hits']],'sources':sources,'reranker':{'variant':getattr(self,'reranker_arm','none'),'audit':retrieval.get('reranker_audit')}}
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
        super().close()
        if self.rag is not None:self.rag.close()
