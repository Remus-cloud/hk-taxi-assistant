"""Offline retrieval regression on fixed fact-group labels. No answer-model calls."""
from pathlib import Path
import argparse,json,os,socket,statistics,sys,time
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def metric(question,ids):
    groups=question['groups']
    if not groups:return None
    covered=sum(bool(set(g['acceptable_chunk_ids'])&set(ids)) for g in groups)
    return {'recall':covered/len(groups),'complete':int(covered==len(groups))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-cache',type=Path,help='Explicit fixed query-vector cache for migration comparison; normally encode locally')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/retrieval.json')
    args=parser.parse_args()
    def blocked(*a,**k):raise RuntimeError('Network disabled during offline retrieval evaluation')
    socket.socket.connect=blocked;socket.create_connection=blocked
    import numpy as np
    from hk_taxi.rag_answer import FrozenRagSession
    from hk_taxi.reranking import FrozenCandidateAdapter,select_candidates,unique_pairs,BgePairScorer
    questions=[json.loads(s) for s in (ROOT/'evaluation/questions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    reference=json.loads((ROOT/'evaluation/reference_top5.json').read_text(encoding='utf-8'))
    session=FrozenRagSession(ROOT,with_generator=False)
    if args.query_cache:
        vectors=dict(zip(json.loads((args.query_cache/'texts.json').read_text(encoding='utf-8')),np.load(args.query_cache/'vectors.npy',allow_pickle=False)))
        encoding_ms=None
    else:vectors,encoding_ms=session.prepare_vectors([q['question'] for q in questions])
    adapter=FrozenCandidateAdapter(session.retriever)
    models=json.loads((ROOT/'configs/models.json').read_text(encoding='utf-8'))
    scorer=BgePairScorer(ROOT/models['reranker']['local_path'])
    rows=[];timings=[]
    for index,q in enumerate(questions,1):
        plan=adapter.build(q['question'],vectors)
        a=[h['chunk']['chunk_id'] for h in select_candidates(plan)['hits']]
        scores,timing=scorer.score(unique_pairs(plan));timings.append(timing['rerank_ms'])
        b=[h['chunk']['chunk_id'] for h in select_candidates(plan,scores)['hits']]
        old=reference[q['id']]
        rows.append({'id':q['id'],'question':q['question'],'A':a,'B':b,
                     'A_matches_reference':a==old['A'],'B_matches_reference':b==old['B'],
                     'A_at5':metric(q,a),'B_at5':metric(q,b),'rerank_ms':timing['rerank_ms']})
        if index%20==0:print(str(index)+'/'+str(len(questions))+' evaluated',flush=True)
    primary_ids={q['id'] for q in questions if q['cohort']=='new_full15' or (q['cohort'].startswith('old_') and q.get('answerability','answerable')=='answerable')}
    primary=[r for r in rows if r['id'] in primary_ids and r['A_at5'] is not None]
    summary={'questions':len(rows),'primary_questions':len(primary),'query_vectors':'explicit_fixed_cache' if args.query_cache else 'fresh_local_encoding',
             'A_rankings_identical':sum(r['A_matches_reference'] for r in rows),'B_rankings_identical':sum(r['B_matches_reference'] for r in rows),
             'changed_questions':[r['id'] for r in rows if not r['A_matches_reference'] or not r['B_matches_reference']],
             'embedding_ms':encoding_ms,'rerank_median_ms':statistics.median(timings),'cloud_calls':0}
    for arm in ['A','B']:
        summary[arm]={'recall5':statistics.mean(r[arm+'_at5']['recall'] for r in primary),
                      'complete5':sum(r[arm+'_at5']['complete'] for r in primary)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));session.close()

if __name__=='__main__':main()
