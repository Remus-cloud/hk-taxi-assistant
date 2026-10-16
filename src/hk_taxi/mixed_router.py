"""Opt-in mixed-request retrieval; frozen original router remains the fallback."""
import json,re,time
from .expanded import ExpandedPolicyRetriever
from .retrieval import fuse

class MixedIntentRetriever(ExpandedPolicyRetriever):
    def __init__(self,base):
        super().__init__(base)
        self.mixed_config=json.loads((base.root/'configs/mixed_router_v1.json').read_text(encoding='utf-8'))
        self.topic_metadata=json.loads((base.root/'data/curated/topics.json').read_text(encoding='utf-8'))
        assert set(self.topic_metadata)=={c['chunk_id'] for c in base.chunks}

    def detect(self,query):
        text,changes=self.router.normalize(query,True)
        cfg=self.mixed_config;pat=cfg['patterns']
        matches=lambda name,s:bool(re.search(pat[name],s,re.I))
        clauses=[s.strip() for s in re.split(r'[？?。；;，,！!]|另外|同时|还有|如果',text) if s.strip()]
        route_shape=bool(re.search(cfg['route_pattern'],text))
        route_cost=route_shape and bool(re.search(cfg['cost_pattern'],text))
        taxi=bool(re.search(cfg['domain_pattern'],text))
        competing=bool(re.search(cfg['other_transport_pattern'],text))
        context=taxi or (route_cost and not competing)
        m=re.search(cfg['known_origin_pattern'],text)
        origin=('港珠澳大桥香港口岸' if '口岸' in m.group(1) else '香港机场') if m else None
        categories=[cat for cat,s in [('urban','市区的士'),('nt','新界的士'),('lantau','大屿山的士')] if s in text]
        requested=[];parts={}
        def add(topic,part):
            if topic not in requested:requested.append(topic)
            if part not in parts.setdefault(topic,[]):parts[topic].append(part)
        if context:
            if route_cost:
                # Carry only the route clauses, not an unrelated service request.
                route_parts=[s for s in clauses if re.search(cfg['route_pattern'],s)]
                add('route_reference',' '.join(route_parts) or text)
            for clause in clauses:
                if matches('receipt',clause):add('receipt',clause)
                if matches('payment',clause):
                    if re.search(r'信用卡|visa',clause,re.I) and matches('payment_failure',clause):topic='payment_card_fallback'
                    elif '现金' in clause:topic='payment_cash'
                    elif matches('payment_requirement',clause):topic='payment_requirement'
                    elif re.search(r'不适用|不兼容|不同|不付|不畀',clause):topic='payment_mismatch'
                    else:topic='payment_method_check'
                    add(topic,clause)
                if matches('luggage',clause) and matches('luggage_question',clause):
                    add('luggage_conditions',clause)
                    if re.search(r'收费|行李费|费用|多少|钱',clause):add('luggage_amount',clause)
                if matches('operating_scope',clause):add('operating_scope',clause)
                if matches('fare_schedule',clause):add('fare_schedule',clause)
                elif matches('waiting',clause):
                    # Waiting alone is one request, not two artificial intents.
                    add('actual_fare_conditions' if route_cost else 'fare_schedule',clause)
        return dict(normalized_query=text,alias_changes=changes,context_confirmed=context,
                    requested_topics=requested,topic_parts=parts,clauses=clauses,origin=origin,
                    fare_categories=categories,mixed=len(requested)>=2,
                    optional_meter_eligible=route_cost and bool(re.search(pat['waiting'],text)))

    def topic_query(self,decision,topic):
        names={'urban':'市区的士','nt':'新界的士','lantau':'大屿山的士'}
        category=' '.join(names[x] for x in decision['fare_categories'])
        parts=' '.join(decision['topic_parts'].get(topic,[]))
        return ' '.join(x for x in [category,self.mixed_config['hints'][topic],parts] if x)

    def candidates(self,query,vector,indices):
        dense=self.base.vectors@vector
        lexical=self.base.bm25.get_scores(self.base.tokenize(query))
        n=self.base.config['candidate_per_retriever']
        def ranking(scores,positive=False):
            ix=sorted(indices,key=lambda i:(-float(scores[i]),self.base.chunks[i]['chunk_id']))
            if positive:ix=[i for i in ix if scores[i]>0]
            return [dict(chunk=self.base.chunks[i],score=float(scores[i]),rank=j+1) for j,i in enumerate(ix[:n])]
        return fuse(ranking(dense),ranking(lexical,True),self.base.config['rrf_k'])

    def eligible_indices(self,decision,topic=None):
        indices=[];excluded=[]
        for i,c in enumerate(self.base.chunks):
            cid=c['chunk_id'];old=self.tags[cid];meta=self.topic_metadata[cid]
            if decision['origin'] and old['origin'] and old['origin']!=decision['origin']:
                excluded.append(dict(chunk_id=cid,reason='known different origin'));continue
            if topic and topic not in meta['topics']:continue
            # Category constraints apply only to new topic channels, not the global fallback.
            if topic and meta['fare_categories'] and decision['fare_categories']:
                if not set(decision['fare_categories'])<=set(meta['fare_categories']):
                    excluded.append(dict(chunk_id=cid,reason='topic candidate category mismatch'));continue
            indices.append(i)
        return indices,excluded

    def search_mixed(self,query,vector,variant='current_combined',topic_vectors=None):
        if variant not in ['current_combined','mixed_detection_only','topic_protection','topic_protection_meter']:
            raise ValueError('Unknown mixed retrieval variant')
        start=time.perf_counter()
        original=super().search(query,vector,True,True)
        decision=self.detect(query)
        original['mixed']=dict(decision=decision,applied=False,variant=variant,subqueries=[],reservations=[],uncovered_topics=[],optional_meter_added=False)
        if variant=='current_combined' or not decision['mixed']:return original
        indices,excluded=self.eligible_indices(decision)
        global_hits=self.candidates(decision['normalized_query'],vector,indices)
        original['route']=dict(intent='mixed_request',origin=decision['origin'],reason='multiple explicit requested topics',requested_topics=decision['requested_topics'])
        original['excluded']=excluded
        trace=original['mixed'];trace['applied']=True
        if variant=='mixed_detection_only':
            original['hits']=global_hits[:self.base.config['final_top_k']]
            original['channels']={'mode':'global_candidate_ranking_no_fixed_quota'}
            trace['uncovered_topics']=[t for t in decision['requested_topics'] if not any(t in self.topic_metadata[h['chunk']['chunk_id']]['topics'] for h in original['hits'])]
            original['ranking_ms']=(time.perf_counter()-start)*1000
            return original
        selected=[];seen=set();covered=set()
        for topic in decision['requested_topics']:
            if topic in covered:continue
            tq=self.topic_query(decision,topic)
            if topic_vectors is None or tq not in topic_vectors:raise ValueError('Pre-encoded topic query vector required')
            ti,te=self.eligible_indices(decision,topic)
            candidates=self.candidates(tq,topic_vectors[tq],ti)
            trace['subqueries'].append(dict(topic=topic,query=tq,excluded=te,candidates=[dict(chunk_id=h['chunk']['chunk_id'],unit_id=h['chunk']['unit_id'],score=float(h['score'])) for h in candidates]))
            if not candidates:continue
            hit=candidates[0];cid=hit['chunk']['chunk_id']
            if cid not in seen:
                if len(selected)>=self.base.config['final_top_k']:continue
                selected.append(hit);seen.add(cid)
            supported=set(self.topic_metadata[cid]['topics'])&set(decision['requested_topics']);covered|=supported
            trace['reservations'].append(dict(topic=topic,chunk_id=cid,supported_requested_topics=sorted(supported)))
        # Optional meter basis comes after every explicit requested-topic attempt.
        if variant=='topic_protection_meter' and decision['optional_meter_eligible'] and len(selected)<self.base.config['final_top_k']:
            meters=[h for h in global_hits if 'meter_basis' in self.topic_metadata[h['chunk']['chunk_id']]['topics']]
            if not meters:
                ti,_=self.eligible_indices(decision,'meter_basis')
                meters=self.candidates(decision['normalized_query'],vector,ti)
            if meters and meters[0]['chunk']['chunk_id'] not in seen:
                selected.append(meters[0]);seen.add(meters[0]['chunk']['chunk_id']);trace['optional_meter_added']=True
        for h in global_hits:
            if len(selected)>=self.base.config['final_top_k']:break
            cid=h['chunk']['chunk_id']
            if cid not in seen:selected.append(h);seen.add(cid)
        original['hits']=selected
        trace['uncovered_topics']=[t for t in decision['requested_topics'] if not any(t in self.topic_metadata[h['chunk']['chunk_id']]['topics'] for h in selected)]
        original['channels']={'mode':'requested_topic_reservations_then_global_fill','reservation_ids':[r['chunk_id'] for r in trace['reservations']]}
        original['ranking_ms']=(time.perf_counter()-start)*1000
        return original
