"""Approved finite rule ownership and script normalization for the unified entry only.

Not a general natural-language entailment checker. No network clients here.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib,json,re
import duckdb
from .runtime_assets import verify_asset
from opencc import OpenCC
from .rag_answer import validate_answer,render_answer

_S=OpenCC('t2s');_T=OpenCC('s2t')
def norm(text):return _S.convert(text).lower()
BAG=r'行李|背包|140|141|轮椅|拐杖|婴儿车|luggage|baggage|backpack|wheelchair|crutch|stroller'
BASE=r'起步|落旗|首.?两.?公里|首.?2.?公里|flag.?fall|initial fare|base fare'
DATE=r'生效|哪天|同一天|同日|开始|起始|日期|effective|start|since|same day'

class RuleStore:
    def __init__(self,root):
        self.root=Path(root)
        cfg=json.loads((self.root/'configs/structured_guard_v1.json').read_text(encoding='utf-8'))
        if cfg['approval_status']!='approved':raise ValueError('Unapproved structured rules')
        self.db=self.root/cfg['database'];self.version=cfg['version']
        verify_asset(self.root,cfg['database'])
        with duckdb.connect(str(self.db),read_only=True) as c:
            self.sources={r[0]:json.loads(r[1]) for r in c.execute('SELECT id,payload FROM guard_sources').fetchall()}
            self.evidence={r[0]:json.loads(r[1]) for r in c.execute('SELECT id,payload FROM guard_evidence').fetchall()}
        for source in self.sources.values():
            for path_key,hash_key in [('extracted_path','extracted_sha256')]:
                if hashlib.sha256((self.root/source[path_key]).read_bytes()).hexdigest()!=source[hash_key]:
                    raise ValueError('Rule evidence snapshot hash mismatch')
    def get(self,rid):
        with duckdb.connect(str(self.db),read_only=True) as c:
            row=c.execute('SELECT payload FROM guard_rules WHERE id=? AND approval_status=?',[rid,'approved']).fetchone()
        if row is None:raise ValueError('No approved rule: '+rid)
        return json.loads(row[0])
    def coverage(self,cid):
        with duckdb.connect(str(self.db),read_only=True) as c:
            row=c.execute('SELECT payload FROM guard_coverage WHERE id=?',[cid]).fetchone()
        if row is None:raise ValueError('Unknown coverage record')
        return json.loads(row[0])

def evaluate_baggage(store,location=None,total_cm=None,item_type=None,mobility_use=None,light_status=None):
    """Evaluate approved predicates only; false criterion is not a legal definition."""
    if item_type in store.get('R06')['predicate']['all'][0]['value']:
        if mobility_use is True:return {'outcome':'exempt','rule_ids':['R06'],'surcharge_cent':0}
        if mobility_use is None:return {'outcome':'needs_clarification','missing':['mobility_aid_use']}
    if location is None:return {'outcome':'needs_clarification','missing':['storage_location']}
    if location=='trunk':return {'outcome':'general_charge_condition_met','rule_ids':['R03'],'subject_to_exemptions':True}
    if location!='passenger_compartment':return {'outcome':'needs_clarification','missing':['recognized_storage_location']}
    size=None
    if total_cm is not None:
        if isinstance(total_cm,bool):raise ValueError('Invalid size')
        try:size=Decimal(str(total_cm))
        except InvalidOperation:raise ValueError('Invalid size') from None
        if not size.is_finite() or size<0:raise ValueError('Invalid size')
    threshold=Decimal(store.get('R04')['predicate']['all'][1]['value'])
    if light_status is True and size is not None and size>threshold:
        return {'outcome':'needs_clarification','missing':['conflicting_light_status_and_size']}
    if light_status is True:return {'outcome':'exempt','rule_ids':['R05'],'surcharge_cent':0}
    if size is None:return {'outcome':'needs_clarification','missing':['dimension_sum_or_confirmed_light_status']}
    return {'outcome':'general_charge_condition_met' if size>threshold else 'dimension_condition_not_met',
            'rule_ids':['R04'],'comparison':'gt','threshold_cm':str(threshold),
            'complete_legal_definition_established':False,'subject_to_exemptions':True}

def script_text(text,language,sources=()):
    if language=='en':return text
    converter=_S if language=='zh-Hans' else _T
    originals=[s.get('text','') for s in sources]
    # Preserve exact quoted official wording and URLs; source cards never get converted.
    pattern=r'https?://[^\s<>]+|“[^”]+”|「[^」]+」'
    result=[];pos=0
    for m in re.finditer(pattern,text):
        result.append(converter.convert(text[pos:m.start()]));token=m.group()
        quoted=token[1:-1]
        result.append(token if token.startswith('http') or any(quoted in s for s in originals) else converter.convert(token))
        pos=m.end()
    result.append(converter.convert(text[pos:]));return ''.join(result)

def normalize_answer(data,language,sources):
    new=deepcopy(data);changes=[]
    for field in ['answer_parts','limitations']:
        for i,part in enumerate(new[field]):
            text=script_text(part['text'],language,sources)
            if text!=part['text']:changes.append(field+'.'+str(i))
            part['text']=text
    for i,q in enumerate(new['clarifying_questions']):
        text=script_text(q,language,sources)
        if text!=q:changes.append('clarifying_questions.'+str(i))
        new['clarifying_questions'][i]=text
    return new,changes

def protected_topics(question):
    q=norm(question)
    # Only reviewed urban base fare is owned; leave explicit other-category base fare to RAG.
    other=bool(re.search(r'新界|绿的|大屿山|蓝的|green|lantau|new territories',q))
    baggage=bool(re.search(BAG,q))
    # Mentions in a receipt/availability question do not establish an owned fee subquestion.
    if re.search(r'收据|receipt|可以带|能不能带|能带|allowed to (?:bring|carry)',q) and not re.search(r'多少钱|几元|140|141|总和|免费|豁免|轻便|定义|生效|起始|哪天|公斤|weight|how much|exempt|definition|effective',q):
        baggage=False
    return {'baggage':baggage, 'base':bool(re.search(BASE,q)) and not other}

def protect_answer(root,question,parsed,sources,language,schema,store=None):
    """Replace owned subanswers with evidence-backed facts, preserve other intents.

    Audit flags describe ownership/replacement, not universal error detection.
    """
    q=norm(question);topics=protected_topics(question);original=deepcopy(parsed);new=deepcopy(parsed)
    cards=deepcopy(sources);audit={'version':'guard-v1','topics':topics,'replacement_policy':'owned_rule_fragments_not_generic_semantic_checker','removed_fragments':[],'rule_ids':[],'coverage_ids':[],'script_changes':[]}
    if any(topics.values()):
        unprotected_requested=bool(re.search(r'visa|mastercard|支付|现金|收据|打印|投诉|隧道|道路费|动物|预约|营运|地铁|payment|cash|receipt|complaint|toll|animal|booking|operating',q))
        def owned(s):
            t=norm(s)
            # In a purely owned subquestion, no free-form pronoun continuation is allowed.
            if not unprotected_requested:return True
            if topics['baggage'] and re.search(BAG+r'|轻便|公斤|重量|尺寸|厘米|起始|维持|豁免',t):return True
            if topics['base'] and re.search(BASE+r'|首.?2.?公里|首.?两.?公里|该费率|这次费率',t):return True
            # Keep a date-only continuation from laundering an unsupported claim.
            if (topics['baggage'] or topics['base']) and re.search(r'同日起|同一天|该日起|same day',t):return True
            return False
        for field in ['answer_parts','limitations']:
            kept=[]
            for part in new[field]:
                # Split only at sentence boundaries/newlines, never decimal points.
                pieces=[x for x in re.split(r'(?<=[。！？])|\n+|(?<=[.!?])\s+(?!\d)',part['text']) if x.strip()]
                keep=[]
                for text in pieces:
                    # Separate independently phrased payment/receipt clauses before ownership.
                    if owned(text) and re.search(r'visa|支付|现金|收据|打印|payment|cash|receipt',norm(text)):
                        subclauses=re.split(r'(?<=[，；;,])',text)
                    else:subclauses=[text]
                    for clause in subclauses:
                        if owned(clause):audit['removed_fragments'].append({'field':field,'text':clause,'source_ids':part['source_ids']})
                        else:keep.append(clause)
                if keep:kept.append(dict(part,text=''.join(keep)))
            new[field]=kept
        # Remove orphan headings left when all their fact sentences were owned.
        headings=r'^(?:必要条件与收费细节|收费条件与豁免|收费条件|不用收行李费的条件包括|结论)[:：]?$'
        new['answer_parts']=[p for p in new['answer_parts'] if not re.fullmatch(headings,norm(p['text']).strip())]
        # Drop redundant clarification about fields that the owned answer explicitly addresses.
        new['clarifying_questions']=[s for s in new['clarifying_questions'] if not owned(s)]
        try:
            store=store or RuleStore(root)
        except (OSError,ValueError,duckdb.Error) as exc:
            # Fail only owned fragments; retain independently supported other subanswers.
            audit.update(error_type=type(exc).__name__,failed_closed=True)
            has_other=bool(new['answer_parts'])
            message=('I cannot confirm this charge right now. Please try again later.' if language=='en' else '暂时无法确认这项收费，请稍后再试。')
            new['answer_parts'].append({'text':message,'source_ids':[]})
            new['status']='partial_answer' if has_other else 'insufficient_evidence'
            new,changes=normalize_answer(new,language,cards);audit['script_changes']=changes
            checked,errors=validate_answer(json.dumps(new,ensure_ascii=False),cards,schema);audit['validation_errors']=errors
            return {'parsed':checked,'sources':cards,'audit':audit,'rendered_answer':render_answer(checked,cards,language) if not errors else None}
        by_source={};next_id=max([int(s['source_id'][1:]) for s in cards]+[0])+1
        def references(rids,extra=()):
            nonlocal next_id
            eids=list(extra)
            for rid in rids:
                r=store.get(rid);audit['rule_ids'].append(rid)
                eids+=r['evidence_ids']+r.get('date_evidence_ids',[])
            result=[]
            for eid in dict.fromkeys(eids):
                e=store.evidence[eid];sid=e['source_id'];s=store.sources[sid]
                if sid not in by_source:
                    title='的士收費表及乘客須知' if sid=='TD' else '的士新收費七月十四日起生效'
                    card={'source_id':'S'+str(next_id),'chunk_id':'structured-guard-'+sid,'title':title,'text':'','source_url':s['source_url'],'verified_as_of':s['verified_as_of'],'evidence_ids':[],'rule_version':store.version}
                    next_id+=1;cards.append(card);by_source[sid]=card
                card=by_source[sid]
                if eid not in card['evidence_ids']:card['evidence_ids'].append(eid);card['text']+=e['quote_original']+'\n'
                if card['source_id'] not in result:result.append(card['source_id'])
            return result
        def add(zh,en,rids=(),extra=()):
            new['answer_parts'].append({'text':en if language=='en' else zh,'source_ids':references(rids,extra)})
        unknown=False
        if topics['base']:
            r=store.get('R01');v=r['value'];amount=v['amount_cent']/100;metres=v['base_distance_m'];date=r['effective_from']
            # No silent urban assumption for a truly unspecified taxi category.
            if re.search(r'市区|红的|红色|urban|red taxi',q):
                zh=f'市区的士首{metres//1000}公里或其任何部分为{amount:g}港元。'
                en=f'Urban taxi flagfall is HK${amount:g} for the first {metres//1000} km or any part.'
                if re.search(DATE,q):
                    zh+=f'这次费率自{date}起生效。';en+=f' This rate took effect on {date}.'
                add(zh,en,['R01'])
            else:
                add('请确认的士类别；本规则库首轮仅整理市区起步价，不把它套到其他类别。','Please specify the taxi category; this rule set covers urban flagfall only.')
                new['clarifying_questions'].append('Which taxi category?' if language=='en' else '你问的是市区、新界还是大屿山的士？');unknown=True
        if topics['baggage']:
            if re.search(DATE,q):
                c=store.coverage('C01');audit['coverage_ids'].append('C01')
                add(c['allowed_answer'],'The reviewed sources do not establish when the HK$6 baggage charge first took effect; the announcement says it remained HK$6.',['R02'],c['basis_evidence_ids']);unknown=True
            weight=bool(re.search(r'公斤|重量|几斤|kg|kilogram|weight',q))
            definition=bool(re.search(r'定义|界定|definition|define',q))
            if weight:
                c=store.coverage('C03');audit['coverage_ids'].append('C03')
                add(c['allowed_answer'],'The reviewed passages do not specify a baggage weight limit in kilograms.',[],c['basis_evidence_ids'])
            if definition:
                c=store.coverage('C02');audit['coverage_ids'].append('C02')
                add(c['allowed_answer'],'These sources state charging and exemption conditions, not a complete legal definition of light baggage.',[],c['basis_evidence_ids'])
            if re.search(r'司机|driver',q) and re.search(r'决定|裁量|decid|discretion',q):
                add('本次证据给出客观收费条件，不支持“是否收费完全由司机决定”的说法。','The reviewed evidence provides objective charging criteria; it does not support unrestricted driver discretion.',['R03','R04','R05'])
            # A date-only subquestion should not trigger a generic baggage lecture.
            date_only=bool(re.search(DATE,q)) and not re.search(r'140|尺寸|车厢|尾箱|背包|轮椅|拐杖|婴儿车|多少钱|收费条件|weight|trunk|cabin',q)
            if not date_only:
                threshold=store.get('R04')['predicate']['all'][1]['value']
                add(f'一般来说，司机可就放在车尾行李厢内的普通行李，或放在车厢内且长、阔、高总和超过{threshold}厘米的行李收取附加费。未超过这一数值，只表示未触发该尺寸条件，不据此认定完整的轻便行李定义。',f'Generally, baggage in the boot, or in the passenger compartment with length + width + height strictly greater than {threshold} cm, may attract the charge. Not exceeding this threshold means this size condition is not met; it is not a complete definition of light baggage.',['R03','R04'])
                if re.search(r'总和|长阔高|长宽高|dimension.?sum|sum of|combined',q):
                    values=re.findall(r'(\d+(?:\.\d+)?)\s*(?:厘米|cm)',q)
                    for value in dict.fromkeys(values):
                        result=evaluate_baggage(store,'passenger_compartment',value)
                        hit=result['outcome']=='general_charge_condition_met'
                        add(f'若已确认放在乘客车厢，尺寸总和{value}厘米：'+('触发' if hit else '未触发')+'“超过140厘米”这一收费条件；仍须核对明确豁免。',f'For baggage confirmed in the passenger compartment, a sum of {value} cm '+('meets' if hit else 'does not meet')+' the strictly-greater-than-140-cm condition; applicable exemptions still matter.',['R04'])
                cents=store.get('R02')['value']['amount_cent'];amount=cents/100
                add(f'符合收费条件且不属豁免的普通行李，每件{amount:g}港元；摆放在乘客车厢内的轻便行李豁免。',f'Chargeable, non-exempt ordinary baggage is HK${amount:g} per piece; light baggage in the passenger compartment is exempt.',['R02','R05'])
                if re.search(r'两件|2件|two (?:bags|pieces)',q):
                    add(f'若两件均属普通应收费行李且无豁免，合计{2*cents/100:g}港元；这不是行程最终总价。',f'If both pieces are ordinary chargeable baggage without exemptions, the surcharge is HK${2*cents/100:g}; this is not a final trip total.',['R02'])
                add('残疾乘客赖以行动的轮椅及拐杖免收行李附加费，明确豁免优先于一般存放或尺寸条件。','Wheelchairs and crutches used as mobility aids by disabled passengers are exempt; this explicit exemption takes precedence over general location or size conditions.',['R06'])
                if re.search(r'婴儿车|stroller',q):
                    add('婴儿车按照上述准则处理，不一概收费或免费，也不自动等同轮椅。','Strollers follow the same criteria; they are not automatically charged, exempt or treated as wheelchairs.',['R07'])
            if unknown:new['status']='partial_answer' if len(new['answer_parts'])>1 else 'insufficient_evidence'
        audit['rule_ids']=list(dict.fromkeys(audit['rule_ids']))
    new,changes=normalize_answer(new,language,cards);audit['script_changes']=changes
    checked,errors=validate_answer(json.dumps(new,ensure_ascii=False),cards,schema)
    audit['validation_errors']=errors;audit['original_preserved']=True
    return {'parsed':checked,'sources':cards,'audit':audit,'rendered_answer':render_answer(checked,cards,language) if not errors else None}
