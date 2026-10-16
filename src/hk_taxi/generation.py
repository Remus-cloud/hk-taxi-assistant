"""Bailian text API adapter. Credentials stay local; no automatic retries/fallback."""
from dataclasses import dataclass,field
from pathlib import Path
from urllib.parse import urlparse
import json,os
from dotenv import dotenv_values
from openai import OpenAI,APIConnectionError,APITimeoutError,APIStatusError,OpenAIError

class GenerationConfigError(ValueError): pass
class GenerationAPIError(RuntimeError): pass

@dataclass(frozen=True)
class GenerationSettings:
    api_key:str=field(repr=False)
    base_url:str
    model:str
    temperature:float=0.1
    max_tokens:int=1500
    enable_thinking:bool=False
    timeout_seconds:float=60.0

    @classmethod
    def load(cls,root,environ=None):
        root=Path(root)
        cfg=json.loads((root/'configs/generation_v1.json').read_text(encoding='utf-8'))
        # No upward directory search; only this project's .env is read.
        local=dotenv_values(root/'.env',encoding='utf-8-sig',interpolate=False) if (root/'.env').is_file() else {}
        supplied=dict(os.environ if environ is None else environ)
        def setting(name,default=''):
            value=supplied.get(name,local.get(name,default))
            return str(value or '').strip()
        key=setting('DASHSCOPE_API_KEY')
        if not key or key.upper() in {'YOUR_API_KEY','YOUR_DASHSCOPE_API_KEY','PASTE_YOUR_KEY_HERE'}:
            raise GenerationConfigError('请在项目 .env 填写 DASHSCOPE_API_KEY，或设置同名环境变量。')
        url=setting('DASHSCOPE_BASE_URL')
        if not url or any(s.lower() in url.lower() for s in ['YOUR_WORKSPACE_ID','{WorkspaceId}','填写']):
            raise GenerationConfigError('请填写百炼控制台对应地域的完整 Base URL，并替换业务空间 ID。')
        parsed=urlparse(url)
        host=parsed.hostname or ''
        allowed=host in {'dashscope.aliyuncs.com','dashscope-intl.aliyuncs.com'} or host.endswith('.maas.aliyuncs.com')
        if parsed.scheme!='https' or not allowed or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise GenerationConfigError('Base URL 应为官方百炼 HTTPS 地址，不包含密钥、查询参数或用户名。')
        if not parsed.path.rstrip('/').endswith('/compatible-mode/v1'):
            raise GenerationConfigError('Base URL 应以 /compatible-mode/v1 结尾，不要追加 /chat/completions。')
        model=setting('GENERATION_MODEL',cfg['model'])
        if not model:raise GenerationConfigError('GENERATION_MODEL 不能为空；请使用控制台的模型 ID。')
        return cls(api_key=key,base_url=url.rstrip('/'),model=model,temperature=cfg['temperature'],
                   max_tokens=cfg['max_tokens'],enable_thinking=cfg['enable_thinking'],timeout_seconds=cfg['timeout_seconds'])

@dataclass(frozen=True)
class GenerationResult:
    answer:str
    requested_model:str
    returned_model:str
    finish_reason:str
    usage:dict
    truncated:bool
    unexpected_thinking:bool

def request_parameters(prompt,settings,system_prompt='请用简体中文简洁回复。'):
    if not prompt.strip():raise GenerationConfigError('问题不能为空。')
    return dict(model=settings.model,messages=[{'role':'system','content':system_prompt},{'role':'user','content':prompt}],
                temperature=settings.temperature,max_tokens=settings.max_tokens,stream=False,
                extra_body={'enable_thinking':settings.enable_thinking})

class BailianGenerator:
    def __init__(self,settings,http_client=None):
        self.settings=settings
        self.client=OpenAI(api_key=settings.api_key,base_url=settings.base_url,timeout=settings.timeout_seconds,
                           max_retries=0,http_client=http_client)

    def close(self):self.client.close()

    def generate(self,prompt,system_prompt='请用简体中文简洁回复。'):
        payload=request_parameters(prompt,self.settings,system_prompt)
        try:
            response=self.client.chat.completions.create(**payload)
        except APITimeoutError:
            raise GenerationAPIError('请求超时，未自动重试。你可以检查网络后手动重试。') from None
        except APIConnectionError:
            raise GenerationAPIError('连接百炼失败，未自动重试。请检查网络和 Base URL。') from None
        except APIStatusError as exc:
            hints={400:'请求参数不被当前模型接受，请核对模型 ID 与思考参数；没有自动改参数或换模型。',
                   401:'认证失败，请核对 Key 与 endpoint 地域是否匹配。',
                   403:'服务权限不足，请核对业务空间和模型权限。',
                   404:'模型或 endpoint 不存在，请复制控制台准确的模型 ID 与 Base URL。',
                   429:'额度、限流或账户状态限制，请检查百炼控制台；未自动重试。'}
            # Do not expose provider response bodies, which may contain credential-like values.
            raise GenerationAPIError(f'百炼请求失败（HTTP {exc.status_code}）：'+hints.get(exc.status_code,'服务暂时不可用，未自动重试。')) from None
        except OpenAIError:
            raise GenerationAPIError('SDK未能处理服务响应，未自动重试；请检查接口兼容性。') from None
        if not response.choices:
            raise GenerationAPIError('服务未返回候选答案；未自动追加请求。')
        choice=response.choices[0]
        answer=choice.message.content
        if not isinstance(answer,str) or not answer.strip():
            raise GenerationAPIError('服务没有返回文字答案；未自动追加请求。')
        usage=response.usage.model_dump(exclude_none=True) if response.usage else {}
        reasoning=choice.message.model_extra.get('reasoning_content') if choice.message.model_extra else None
        reasoning_count=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0) or 0
        return GenerationResult(answer=answer,requested_model=self.settings.model,returned_model=response.model,
             finish_reason=choice.finish_reason,usage=usage,truncated=choice.finish_reason=='length',
             unexpected_thinking=not self.settings.enable_thinking and bool(reasoning or reasoning_count))
