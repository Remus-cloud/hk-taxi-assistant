"""Rule query router behind a replaceable interface; baseline module remains unchanged."""
import json,re,time
from opencc import OpenCC
from .retrieval import ranked,fuse

class QueryRouter:
    def __init__(self, policy):
        self.policy=policy
        self.cc=OpenCC("t2s")

    def normalize(self,query,aliases=False):
        text=self.cc.convert(query)
        changes=[]
        if not aliases:return text,changes
        context=bool(re.search(r"的士|出租车|车费|计费|费率|跳表|起步|乘车|载客|营运|过海",text))
        pairs=self.policy["aliases"]
        pattern="|".join(re.escape(x) for x in sorted(pairs,key=len,reverse=True))
        def replace(m):
            token=m.group()
            if token in {"红的","绿的","蓝的"} and not context:return token
            canonical=pairs[token];changes.append(dict(original=token,canonical=canonical));return canonical
        return re.sub(pattern,replace,text),changes

    def route(self,query):
        text=self.cc.convert(query)
        rule_terms=re.findall(r"起步价|起步距离|起步|跳表|计费单位|每公里|每分钟|每200|费率|阈值|附加费|回程费|行李费|后续计费|后续收费",text)
        origin=None
        match=re.search(r"(?:从|由)?(港珠澳大桥香港口岸|香港国际机场|香港机场|机场|香港口岸)(?:出发)?(?:到|至|去|前往)",text)
        if match:
            origin='港珠澳大桥香港口岸' if '口岸' in match.group(1) else '香港机场'
        # Detect other explicit routes as well; never assume an origin label for unknown places.
        route_shape=bool(match or re.search(r"从.{1,20}(?:到|去|至).{1,20}",text))
        cost_terms=re.findall(r"多少钱|车费|费用|估价|报价|总价|大约|估算",text)
        if route_shape and cost_terms and rule_terms:
            return dict(intent='mixed_or_uncertain',origin=None,reason='route cost and explicit rule question coexist',matched_terms=rule_terms+cost_terms)
        if route_shape and cost_terms:
            return dict(intent='route_estimate',origin=origin,reason='explicit origin/destination route with cost request',matched_terms=cost_terms)
        if rule_terms and not route_shape:
            return dict(intent='general_fare_rule',origin=None,reason='explicit rule terminology without route request',matched_terms=rule_terms)
        return dict(intent='mixed_or_uncertain',origin=None,reason='no unambiguous fare intent',matched_terms=[])


class PolicyRetriever:
    def __init__(self,base):
        self.base=base
        self.tags=json.loads((base.root/'configs/content_types_v2.json').read_text(encoding='utf-8'))
        self.policy=json.loads((base.root/'configs/router_v2.json').read_text(encoding='utf-8'))
        self.router=QueryRouter(self.policy)
        assert set(self.tags)=={c['chunk_id'] for c in base.chunks}

    def search(self,query,vector,aliases=False,routing=False):
        normalized,changes=self.router.normalize(query,aliases)
        decision=self.router.route(normalized) if routing else dict(intent='baseline',origin=None,reason='router disabled')
        if not aliases and not routing:
            result=self.base.search(query,'hybrid',5,vector)
            result.update(normalized_query=query,alias_changes=[],route=decision)
            return result
        start=time.perf_counter()
        dense_scores=self.base.vectors@vector
        lexical_scores=self.base.bm25.get_scores(self.base.tokenize(normalized))
        eligible=[];excluded=[]
        for i,c in enumerate(self.base.chunks):
            tags=self.tags[c['chunk_id']]; types=tags['content_types'];reason=None
            if decision['intent']=='general_fare_rule' and 'reference_estimate' in types:
                reason='numeric reference estimate excluded for explicit general rule question'
            if decision['intent']=='route_estimate' and decision['origin'] and tags['origin'] and tags['origin']!=decision['origin']:
                reason='known different origin in route reference content'
            if reason:excluded.append(dict(chunk_id=c['chunk_id'],reason=reason))
            else:eligible.append(i)
        def candidates(scores,positive=False):
            ix=sorted(eligible,key=lambda i:(-float(scores[i]),self.base.chunks[i]['chunk_id']))
            if positive:ix=[i for i in ix if scores[i]>0]
            return [dict(chunk=self.base.chunks[i],score=float(scores[i]),rank=j+1) for j,i in enumerate(ix[:10])]
        all_hits=fuse(candidates(dense_scores),candidates(lexical_scores,True),60)
        channels={}
        if decision['intent']=='route_estimate':
            # Run independently ranked channels with the same corpus BM25 statistics; filtering does not refit IDF.
            full_eligible=eligible
            def channel(types):
                nonlocal eligible
                eligible=[i for i in full_eligible if set(self.tags[self.base.chunks[i]['chunk_id']]['content_types']) & set(types)]
                return fuse(candidates(dense_scores),candidates(lexical_scores,True),60)
            reference=channel(['reference_estimate','estimate_explanation','passenger_guide'])
            rules=channel(['fare_rule'])
            eligible=full_eligible
            selected=reference[:3]+rules[:2]
            channels=dict(reference_ids=[h['chunk']['chunk_id'] for h in reference[:3]],rule_ids=[h['chunk']['chunk_id'] for h in rules[:2]])
            seen={h['chunk']['chunk_id'] for h in selected}
            for h in all_hits:
                if len(selected)>=5:break
                if h['chunk']['chunk_id'] not in seen:selected.append(h);seen.add(h['chunk']['chunk_id'])
        else:selected=all_hits[:5]
        return dict(query=query,normalized_query=normalized,alias_changes=changes,route=decision,excluded=excluded,channels=channels,hits=selected,ranking_ms=(time.perf_counter()-start)*1000)
