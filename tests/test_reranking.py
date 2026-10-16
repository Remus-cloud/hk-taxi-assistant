from copy import deepcopy
import math
import pytest
from hk_taxi.reranking import select_candidates,unique_pairs

def hit(cid):return {'chunk':{'chunk_id':cid,'text':'text '+cid},'score':1,'rank':1}
def pool(*ids,q='query'):return {'query':q,'hits':[hit(x) for x in ids]}
def global_plan():return {'mode':'global','pools':{'global':pool('a','b','c','d','e','f')},
                         'decision':{'requested_topics':[]},'topic_metadata':{}}
def scores(p,values):return {(c['query'],h['chunk']['chunk_id']):values[h['chunk']['chunk_id']] for c in p['pools'].values() for h in c['hits']}

def test_can_promote_sixth_candidate_without_mutating_plan():
    p=global_plan();before=deepcopy(p);s=scores(p,dict(a=1,b=2,c=3,d=4,e=5,f=6))
    assert select_candidates(p,s)['hits'][0]['chunk']['chunk_id']=='f'
    assert p==before

def test_tie_break_is_stable_chunk_id_and_no_model_top5_unchanged():
    p=global_plan()
    assert [h['chunk']['chunk_id'] for h in select_candidates(p)['hits']]==list('abcde')
    assert [h['chunk']['chunk_id'] for h in select_candidates(p,scores(p,{x:0 for x in 'abcdef'}))['hits']]==list('abcde')

def test_mixed_global_scores_cannot_displace_receipt_payment_and_luggage():
    p={'mode':'mixed','pools':{'global':pool('fare1','fare2','fare3','fare4','fare5'),
         'topic:receipt':pool('receipt'), 'topic:payment':pool('payment'), 'topic:luggage':pool('luggage')},
       'decision':{'requested_topics':['receipt','payment','luggage']},
       'topic_metadata':{**{f'fare{i}':{'topics':[]} for i in range(1,6)},
                         **{t:{'topics':[t]} for t in ['receipt','payment','luggage']}}}
    s=scores(p,{**{f'fare{i}':100 for i in range(1,6)},'receipt':-4,'payment':-5,'luggage':-6})
    r=select_candidates(p,s);ids=[h['chunk']['chunk_id'] for h in r['hits']]
    assert ids[:3]==['receipt','payment','luggage'] and len(ids)==5 and not r['uncovered_topics']

def test_one_chunk_supporting_two_topics_is_reserved_once():
    p={'mode':'mixed','pools':{'global':pool('a','b','c','d','e'),
          'topic:receipt':pool('both'), 'topic:payment':pool('different')},
       'decision':{'requested_topics':['receipt','payment']},
       'topic_metadata':{**{x:{'topics':[]} for x in 'abcde'},'both':{'topics':['receipt','payment']},'different':{'topics':['payment']}}}
    r=select_candidates(p)
    assert [x['topic'] for x in r['reservations']]==['receipt']
    assert len({h['chunk']['chunk_id'] for h in r['hits']})==5

def test_route_reference_keeps_three_reference_two_rules():
    p=global_plan();p['mode']='route_reference';p['pools'].update(reference=pool('a','b','c','d'),rules=pool('e','f','g'))
    s=scores(p,{x:i for i,x in enumerate('abcdefg')})
    assert [h['chunk']['chunk_id'] for h in select_candidates(p,s)['hits']]==list('dcbgf')

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_reranker_scores_never_silently_enter_ranking(value):
    p=global_plan();s=scores(p,{x:1 for x in 'abcdef'});s[('query','b')]=value
    with pytest.raises(ValueError):select_candidates(p,s)

def test_missing_scores_fail_instead_of_dropping_candidate():
    with pytest.raises(ValueError):select_candidates(global_plan(),{})

def test_pair_dedup_uses_query_and_chunk_not_just_chunk():
    p=global_plan();p['pools']['same']=pool('a');p['pools']['other']=pool('a',q='topic')
    assert len(unique_pairs(p))==7

def test_optional_session_preserves_unified_schema_and_generator_access():
    from types import SimpleNamespace
    from hk_taxi.reranking import RerankedRagSession
    wrapped=object.__new__(RerankedRagSession)
    wrapped.frozen=SimpleNamespace(schema={'required':['answer_parts']},generator=object(),system='unchanged prompt')
    assert wrapped.schema is wrapped.frozen.schema
    assert wrapped.generator is wrapped.frozen.generator and wrapped.system=='unchanged prompt'
