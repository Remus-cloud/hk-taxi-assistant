"""Opt-in reranking over frozen candidate channels; never changes the frozen release."""
from copy import deepcopy
from pathlib import Path
import json,math,time


class FrozenCandidateAdapter:
    def __init__(self,frozen_retriever):
        self.r=frozen_retriever
        if self.r.base.config['candidate_per_retriever']!=10 or self.r.base.config['final_top_k']!=5:
            raise ValueError('This adapter is audited for the frozen 10-candidate / top-5 policy only')

    def build(self,question,vectors):
        start=time.perf_counter();r=self.r
        normalized,changes=r.router.normalize(question,True)
        decision=r.detect(question)
        pools={}
        def pool(name,query,indices):
            pools[name]={'query':query,'hits':r.candidates(query,vectors[query],indices)}
        if decision['mixed']:
            eligible,excluded=r.eligible_indices(decision)
            pool('global',normalized,eligible)
            for topic in decision['requested_topics']:
                eligible,_=r.eligible_indices(decision,topic)
                pool('topic:'+topic,r.topic_query(decision,topic),eligible)
            mode='mixed'
        else:
            routing=r.router.route(normalized);eligible=[];excluded=[]
            for i,c in enumerate(r.base.chunks):
                tags=r.tags[c['chunk_id']];reason=None
                if routing['intent']=='general_fare_rule' and 'reference_estimate' in tags['content_types']:
                    reason='numeric reference estimate excluded for explicit general rule question'
                if routing['intent']=='route_estimate' and routing['origin'] and tags['origin'] and tags['origin']!=routing['origin']:
                    reason='known different origin in route reference content'
                if reason:excluded.append({'chunk_id':c['chunk_id'],'reason':reason})
                else:eligible.append(i)
            pool('global',normalized,eligible)
            if routing['intent']=='route_estimate':
                pool('reference',normalized,[i for i in eligible if set(r.tags[r.base.chunks[i]['chunk_id']]['content_types']) & {'reference_estimate','estimate_explanation','passenger_guide'}])
                pool('rules',normalized,[i for i in eligible if 'fare_rule' in r.tags[r.base.chunks[i]['chunk_id']]['content_types']])
                mode='route_reference'
            else:mode='global'
        return {'question':question,'normalized_query':normalized,'alias_changes':changes,'mode':mode,
                'decision':decision,'pools':pools,'excluded':excluded,
                'topic_metadata':r.topic_metadata,'candidate_build_ms':(time.perf_counter()-start)*1000}


def unique_pairs(plan):
    """One model input for each distinct channel-query and chunk, with no gold input."""
    seen=set();items=[]
    for channel in plan['pools'].values():
        for hit in channel['hits']:
            key=(channel['query'],hit['chunk']['chunk_id'])
            if key not in seen:
                seen.add(key);items.append((key,channel['query'],hit['chunk']['text']))
    return items


def select_candidates(plan,scores=None):
    """Keep the frozen quotas, supported-topic reservations and final fill order."""
    pools=deepcopy(plan['pools'])
    if scores is not None:
        for channel in pools.values():
            for hit in channel['hits']:
                key=(channel['query'],hit['chunk']['chunk_id'])
                if key not in scores or not math.isfinite(scores[key]):raise ValueError('Missing/nonfinite reranker score')
                hit['rerank_score']=float(scores[key])
            channel['hits'].sort(key=lambda h:(-h['rerank_score'],h['chunk']['chunk_id']))
    global_hits=pools['global']['hits'];reservations=[]
    if plan['mode']=='global':selected=global_hits[:5]
    elif plan['mode']=='route_reference':
        # Exact old channel allocation: do not silently replace it with global top-5.
        selected=pools['reference']['hits'][:3]+pools['rules']['hits'][:2]
        seen={h['chunk']['chunk_id'] for h in selected}
        for hit in global_hits:
            if len(selected)>=5:break
            if hit['chunk']['chunk_id'] not in seen:selected.append(hit);seen.add(hit['chunk']['chunk_id'])
    elif plan['mode']=='mixed':
        selected=[];seen=set();covered=set();requested=plan['decision']['requested_topics']
        for topic in requested:
            if topic in covered:continue
            hits=pools['topic:'+topic]['hits']
            if not hits:continue
            hit=hits[0];cid=hit['chunk']['chunk_id']
            if cid not in seen:
                if len(selected)>=5:continue
                selected.append(hit);seen.add(cid)
            supported=set(plan['topic_metadata'][cid]['topics'])&set(requested);covered|=supported
            reservations.append({'topic':topic,'chunk_id':cid,'supported_requested_topics':sorted(supported)})
        for hit in global_hits:
            if len(selected)>=5:break
            cid=hit['chunk']['chunk_id']
            if cid not in seen:selected.append(hit);seen.add(cid)
    else:raise ValueError('Unknown candidate plan mode')
    requested=plan['decision']['requested_topics']
    uncovered=[t for t in requested if not any(t in plan['topic_metadata'][h['chunk']['chunk_id']]['topics'] for h in selected)]
    return {'hits':selected,'reservations':reservations,'uncovered_topics':uncovered,
            'reranker_enabled':scores is not None,'mode':plan['mode']}


class BgePairScorer:
    def __init__(self,model_path,max_length=1024,batch_size=2):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        start=time.perf_counter();self.torch=torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU fallback')
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(str(model_path),local_files_only=True,
                    trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
        self.max_length=max_length;self.batch_size=batch_size
        torch.cuda.synchronize();self.load_ms=(time.perf_counter()-start)*1000
        self.info={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'capability':list(torch.cuda.get_device_capability(0)),'arch_list':torch.cuda.get_arch_list(),
            'dtype':'float32','batch_size':batch_size,'max_length':max_length,'model_load_ms':self.load_ms,
            'attention':'eager','quantization':None,'optimizer_created':False,'training':False}

    def score(self,items):
        torch=self.torch;torch.cuda.synchronize();start=time.perf_counter()
        out={};max_tokens=0
        for i in range(0,len(items),self.batch_size):
            batch=items[i:i+self.batch_size]
            tokenized=self.tokenizer([q for _,q,_ in batch],[d for _,_,d in batch],padding=True,
                                     truncation=False,return_tensors='pt')
            lengths=tokenized['attention_mask'].sum(dim=1).tolist();max_tokens=max([max_tokens]+lengths)
            if max_tokens>self.max_length:raise ValueError(f'Pair exceeds approved limit ({max_tokens}>{self.max_length}); not silently truncated')
            with torch.inference_mode():
                logits=self.model(**{k:v.to('cuda') for k,v in tokenized.items()}).logits.reshape(-1).float().cpu().tolist()
            for (key,_,_),value in zip(batch,logits):
                if not math.isfinite(value):raise ValueError('Nonfinite model output')
                out[key]=float(value)
        torch.cuda.synchronize()
        return out,{'rerank_ms':(time.perf_counter()-start)*1000,'pair_count':len(items),'max_pair_tokens':max_tokens,
                    'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2}


class RerankedRagSession:
    """Composition over the old session. Old CLI and default Streamlit stay intact."""
    def __init__(self,frozen_session,scorer):
        self.frozen=frozen_session;self.scorer=scorer;self.adapter=FrozenCandidateAdapter(frozen_session.retriever)
    def __getattr__(self,name):
        # Keep schema/prompt/generator access compatible with UnifiedAssistant.
        return getattr(self.frozen,name)
    def prepare_vectors(self,questions):return self.frozen.prepare_vectors(questions)
    def retrieve(self,question,vectors):
        from .rag_answer import source_cards
        plan=self.adapter.build(question,vectors)
        scores,timing=self.scorer.score(unique_pairs(plan));result=select_candidates(plan,scores)
        result.update(reranker_audit=timing,candidate_build_ms=plan['candidate_build_ms'])
        return result,source_cards(result['hits'])
    def generate_from_sources(self,*a,**kw):return self.frozen.generate_from_sources(*a,**kw)
    def close(self):self.frozen.close()
