"""User-facing journey failure messages; no change to tariff validity policy."""
from pathlib import Path
import json,re
from .rag_answer import select_language
from .structured_guard import script_text

TARIFF_INTERVAL_ERROR='Requested date falls outside the verified tariff interval; reverify source.'

def fare_date_feedback(root,language='zh-Hans',state=None):
    text='This earlier reply did not include a fare. Estimates are available again; please resend the journey.' if language=='en' else '这条旧回答没有算出车费。现在已恢复估算，请重新发送行程。'
    return script_text(text,language)


def present_journey_failure(root,record,body=None):
    body=record.get('answer','') if body is None else body
    if TARIFF_INTERVAL_ERROR not in body:return body
    language=record.get('answer_language') or select_language(record.get('question',''))
    result=fare_date_feedback(root,language,record.get('state_after'))
    # A historical mixed answer can contain a failed trip and a valid rule response.
    tail=((record.get('structured_guard') or {}).get('rendered_answer')
          or (record.get('generation') or {}).get('rendered_answer'))
    if tail:
        tail=re.split(r'\n(?:来源（官方原文）|來源（官方原文）|Sources \(original official text\))[:：]\s*\n',tail,maxsplit=1)[0].rstrip()
        if tail and tail in body:result+='\n\n'+tail
    return result
