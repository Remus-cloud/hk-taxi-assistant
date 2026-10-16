"""Portable retrieval + reviewed multilingual generation + local citation validation."""
from pathlib import Path
import importlib,importlib.util,json,hashlib,re,sys
from opencc import OpenCC
from jsonschema import Draft202012Validator
from .generation import BailianGenerator,GenerationSettings

LANGUAGES={'zh-Hans':'简体中文','zh-Hant':'繁體中文','en':'English'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def select_language(question,requested=None):
    if requested is not None:
        if requested not in LANGUAGES:raise ValueError('Supported languages: zh-Hans, zh-Hant, en')
        return requested
    explicit=[(r'请用.{0,3}(?:英文|英语)|請用.{0,3}(?:英文|英語)|answer in english','en'),
              (r'请用.{0,3}繁体|請用.{0,3}繁體|traditional chinese','zh-Hant'),
              (r'请用.{0,3}简体|請用.{0,3}簡體|simplified chinese','zh-Hans')]
    for pattern,language in explicit:
        if re.search(pattern,question,re.I):return language
    if re.search(r'[\u3400-\u9fff]',question):
        return 'zh-Hant' if OpenCC('t2s').convert(question)!=question else 'zh-Hans'
    return 'en' if re.search(r'[A-Za-z]',question) else 'zh-Hans'

def source_cards(hits):
    fields=['source_locator','page','verified_as_of','document_version','effective_from','estimate_updated_at','content_kind']
    cards=[]
    for i,h in enumerate(hits,1):
        c=h['chunk']
        cards.append(dict(source_id=f'S{i}',chunk_id=c['chunk_id'],title=c['title'],text=c['text'],
                          source_url=c['source_url'],**{field:c.get(field) for field in fields}))
    return cards

def build_user_message(template,question,sources,language):
    # Internal chunk IDs, retrieval metrics and gold answers never enter model context.
    public=[{k:v for k,v in s.items() if k!='chunk_id'} for s in sources]
    return template.replace('{{question}}',question).replace('{{sources_json}}',json.dumps(public,ensure_ascii=False,indent=2))+f'\n本次指定回答语言 / Requested answer language: {LANGUAGES[language]} ({language}).\n'

def validate_answer(raw,sources,schema):
    errors=[]
    try:data=json.loads(raw)
    except json.JSONDecodeError:return None,['invalid_json']
    for error in Draft202012Validator(schema).iter_errors(data):
        errors.append('schema:'+'.'.join(str(x) for x in error.absolute_path)+':'+error.validator)
    if errors:return data,errors
    known={s['source_id'] for s in sources}
    for field in ['answer_parts','limitations']:
        for i,part in enumerate(data[field]):
            for sid in part['source_ids']:
                if sid not in known:errors.append(f'unknown_source:{sid}')
            if re.search(r'https?://|www\.',part['text'],re.I):errors.append(f'generated_url:{field}.{i}')
    return data,errors

def render_answer(data,sources,language):
    mapping={s['source_id']:s for s in sources};used=[];lines=[]
    labels={'zh-Hans':('需要确认','说明与限制','来源（官方原文）'),
            'zh-Hant':('需要確認','說明與限制','來源（官方原文）'),
            'en':('Clarifying questions','Limitations','Sources (original official text)')}[language]
    def render_part(part):
        markers=[]
        for sid in part['source_ids']:
            if sid not in mapping:raise ValueError('Unvalidated source identifier')
            if sid not in used:used.append(sid)
            markers.append(f"[{sid}]({mapping[sid]['source_url']})")
        return part['text']+(' '+' '.join(markers) if markers else '')
    for part in data['answer_parts']:lines.append(render_part(part))
    if data['clarifying_questions']:
        lines.append(labels[0]+'：');lines.extend('- '+q for q in data['clarifying_questions'])
    if data['limitations']:
        lines.append(labels[1]+'：');lines.extend(render_part(part) for part in data['limitations'])
    if used:
        lines.append(labels[2]+'：')
        for sid in used:
            s=mapping[sid]
            lines.append(f"- [{sid} · {s['title']}]({s['source_url']})")
    return '\n\n'.join(lines)

class FrozenRagSession:
    def __init__(self,root,with_generator=True):
        self.root=Path(root)
        self.config=json.loads((self.root/'configs/rag_generation_v1.json').read_text(encoding='utf-8'))
        from .runtime_retrieval import RuntimeRetriever
        from .mixed_router import MixedIntentRetriever
        self.base=RuntimeRetriever(self.root)
        self.retriever=MixedIntentRetriever(self.base)
        self.system=(self.root/self.config['system_prompt']).read_text(encoding='utf-8')
        self.template=(self.root/self.config['user_template']).read_text(encoding='utf-8')
        self.schema=json.loads((self.root/self.config['schema']).read_text(encoding='utf-8'))
        self.generator=BailianGenerator(GenerationSettings.load(self.root)) if with_generator else None

    def prepare_vectors(self,questions):
        texts=[]
        for question in questions:
            d=self.retriever.detect(question)
            for t in [d['normalized_query']]+([self.retriever.topic_query(d,topic) for topic in d['requested_topics']] if d['mixed'] else []):
                if t not in texts:texts.append(t)
        vectors,ms,_=self.base.encode_queries(texts)
        return dict(zip(texts,vectors)),ms

    def retrieve(self,question,vectors):
        d=self.retriever.detect(question)
        result=self.retriever.search_mixed(question,vectors[d['normalized_query']],self.config['retrieval_variant'],vectors)
        return result,source_cards(result['hits'])

    def generate_from_sources(self,question,sources,requested_language=None):
        language=select_language(question,requested_language)
        if self.generator is None:raise RuntimeError('Generator not enabled')
        message=build_user_message(self.template,question,sources,language)
        response=self.generator.generate(message,self.system)
        data,errors=validate_answer(response.answer,sources,self.schema)
        if response.truncated:errors.append('output_truncated')
        if response.unexpected_thinking:errors.append('unexpected_thinking')
        return dict(language=language,raw_response=response.answer,parsed=data,validation_errors=errors,
             rendered_answer=render_answer(data,sources,language) if not errors else None,
             requested_model=response.requested_model,returned_model=response.returned_model,
             finish_reason=response.finish_reason,usage=response.usage)

    def close(self):
        if self.generator is not None:self.generator.close()
