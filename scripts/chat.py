"""Unified taxi questions. Old RAG and journey CLIs remain available."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hk_taxi.unified_passenger_v2 import UnifiedAssistant,empty_state
from hk_taxi.unified import BudgetError
from hk_taxi.generation import GenerationAPIError,GenerationConfigError

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('question',nargs='?')
    p.add_argument('--session',required=True)
    p.add_argument('--interactive',action='store_true')
    p.add_argument('--reranker',choices=['none','pretrained'],default='pretrained')
    p.add_argument('--language',choices=['zh-Hans','zh-Hant','en'])
    args=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,60}',args.session):p.error('Invalid session ID.')
    if not args.question and not args.interactive:p.error('Provide a question or --interactive.')
    folder=ROOT/'artifacts/passenger_sessions_v2';folder.mkdir(parents=True,exist_ok=True)
    file=folder/(args.session+'.json')
    state=json.loads(file.read_text(encoding='utf-8')) if file.exists() else empty_state()
    assistant=None
    try:
        assistant=UnifiedAssistant(ROOT,reranker=args.reranker)
        question=args.question
        while True:
            if question is None:
                question=input('你（输入 /exit 结束）：').strip()
            if question in {'/exit','/quit'}:break
            if not question: question=None;continue
            state,record=assistant.handle(question,state,language=args.language)
            file.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
            with (folder/(args.session+'.turns.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(record['answer'])
            print('\n会话已保存。自由文本路由会调用Qwen；行程工具每轮最多3次ORS，规则回答最多另调用1次Qwen。')
            if not args.interactive:break
            question=None
    except (GenerationAPIError,GenerationConfigError,BudgetError,ValueError) as exc:
        print(str(exc),file=sys.stderr);return 1
    finally:
        if assistant:assistant.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
