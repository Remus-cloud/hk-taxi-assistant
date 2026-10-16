"""Session-local web adapter. UI reruns never automatically issue model requests."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta,date
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from uuid import uuid4
import json,re
from .unified import UnifiedAssistant,empty_state
from .generation import GenerationSettings,GenerationConfigError,GenerationAPIError

LANGUAGES={'跟随提问':None,'简体':'zh-Hans','繁體':'zh-Hant','English':'en'}
def now():return datetime.now(timezone(timedelta(hours=8)))

def page_status(root,today=None):
    root=Path(root);today=today or now().date()
    try:GenerationSettings.load(root);configured=True
    except (GenerationConfigError,OSError,ValueError):configured=False
    return {'configured':configured}


def split_answer(record):
    """Keep cited prose; collapse only the program-generated source bibliography."""
    answer=record.get('answer','')
    # Remove only bibliography lines; never truncate later mixed-answer blocks.
    heading=r'^(?:来源（官方原文）|來源（官方原文）|Sources \(original official text\)|网页来源|網頁來源|Web sources)[:：]\s*$'
    lines=[];bibliography=False
    for line in answer.splitlines():
        if re.match(heading,line.strip()):bibliography=True;continue
        if bibliography:
            if not line.strip() or re.match(r'^- \[[SW]\d+ · ',line):continue
            bibliography=False
        lines.append(line)
    body='\n'.join(lines).rstrip()
    # Currency symbols must not start Markdown inline-math across fare lines.
    body=re.sub(r'(?<!\\)HK\$(?=\d)',r'HK\\$',body)
    used=set(re.findall(r'\[([SW]\d+)\]',body))
    guard=record.get('structured_guard') or {}
    sources=list(guard.get('sources') or (record.get('retrieval') or {}).get('sources',[]))
    web_sources=(record.get('web_search') or {}).get('sources',[])
    # Bibliography-only sources still belong in the expander.
    return body,[deepcopy(s) for s in sources if s.get('source_id') in used]+[deepcopy(s) for s in web_sources]


def safe_source_url(url):
    u=urlparse(url or '')
    return url if u.scheme=='https' and u.hostname and not u.username and not u.password else None

class WebChatSession:
    def __init__(self,root,engine_factory=None):
        self.root=Path(root);self._factory=engine_factory or (lambda:UnifiedAssistant(self.root))
        self._engine=None;self._lock=Lock();self.state=empty_state();self.messages=[]
        self.session_id='web-'+uuid4().hex;self.last_save_warning=False
    def close(self):
        if self._engine is not None:self._engine.close();self._engine=None
    def new_chat(self):
        if self._lock.locked():return False
        self.state=empty_state();self.messages=[];self.session_id='web-'+uuid4().hex
        self.last_save_warning=False;return True
    def ask(self,question,language=None):
        question=question.strip()
        if not question:raise ValueError('请输入问题。')
        if language not in {None,'zh-Hans','zh-Hant','en'}:raise ValueError('不支持的回答语言。')
        if not self._lock.acquire(blocking=False):return {'success':False,'error':'上一条问题仍在处理，请稍候。'}
        started=now();self.last_save_warning=False
        try:
            if self._engine is None:self._engine=self._factory()
            new,record=self._engine.handle(question,deepcopy(self.state),language=language)
            self.state=new
            self.messages.extend([{'role':'user','content':question},
              {'role':'assistant','content':record['answer'],'record':record,
               'elapsed_seconds':round((now()-started).total_seconds(),1)}])
            self._save(record)
            return {'success':True,'record':record}
        except (GenerationAPIError,GenerationConfigError) as exc:
            # These exception classes deliberately contain only sanitized public messages.
            return self._failure(question,str(exc),type(exc).__name__)
        except Exception as exc:
            return self._failure(question,'本次请求未完成，请稍后重新提交。没有自动重试。',type(exc).__name__)
        finally:self._lock.release()
    def _failure(self,question,message,error_type):
        record={'question':question,'answer':message,'status':'web_error','error_type':error_type,'created_at':now().isoformat()}
        self.messages.extend([{'role':'user','content':question},{'role':'assistant','content':message,'record':record}])
        self._save(record);return {'success':False,'error':message,'record':record}
    def _save(self,record):
        try:
            folder=self.root/'artifacts/web_sessions';folder.mkdir(parents=True,exist_ok=True)
            payload={'session_id':self.session_id,'state':self.state,'messages':self.messages,'updated_at':now().isoformat()}
            temp=folder/(self.session_id+'.tmp')
            temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(folder/(self.session_id+'.json'))
            with (folder/(self.session_id+'.turns.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        except (OSError,TypeError):
            # A completed paid call must not be automatically retried just because logging failed.
            self.last_save_warning=True
    def export_text(self):
        return '\n\n'.join(('你：' if m['role']=='user' else '港的通：')+'\n'+m['content'] for m in self.messages)

def trip_summary(state):
    def name(key):
        item=state.get(key)
        if not item:return None
        anchor=item.get('anchor',{});selected=item.get('selected_id')
        candidate=next((c for c in item.get('candidates',[]) if c.get('id')==selected),None)
        return (candidate or {}).get('name') or anchor.get('name') or anchor.get('query')
    return name('origin'),name('destination'),(state.get('conditions') or {}).get('luggage')
