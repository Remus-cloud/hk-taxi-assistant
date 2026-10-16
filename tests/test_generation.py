import json,subprocess,shutil
from pathlib import Path
import httpx2,pytest
from hk_taxi.generation import GenerationSettings,GenerationConfigError,GenerationAPIError,BailianGenerator
R=Path(__file__).resolve().parents[1]
FAKE_KEY='sk-test-only-not-a-real-key'
URL='https://test-space.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'

def make_settings():return GenerationSettings(api_key=FAKE_KEY,base_url=URL,model='qwen3.7-plus-2026-05-26')

def make_client(handler):
    transport=httpx2.MockTransport(handler)
    return BailianGenerator(make_settings(),http_client=httpx2.Client(transport=transport))

def completion(reason='stop',content='API连接成功。'):
    return {'id':'chatcmpl-mock','object':'chat.completion','created':1,'model':'qwen3.7-plus-2026-05-26',
            'choices':[{'index':0,'finish_reason':reason,'message':{'role':'assistant','content':content}}],
            'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}

def prepare_root(tmp_path,env):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/generation_v1.json').write_text((R/'configs/generation_v1.json').read_text(encoding='utf-8'),encoding='utf-8')
    (tmp_path/'.env').write_text(env,encoding='utf-8-sig')
    return tmp_path

def test_exact_payload_and_single_request():
    requests=[]
    def handler(request):
        requests.append(request)
        assert str(request.url)==URL+'/chat/completions'
        body=json.loads(request.content)
        assert body['model']=='qwen3.7-plus-2026-05-26'
        assert body['enable_thinking'] is False and body['temperature']==0.1
        assert body['max_tokens']==1500 and body['stream'] is False
        assert 'tools' not in body and FAKE_KEY not in request.content.decode()
        assert request.headers['authorization']=='Bearer '+FAKE_KEY
        return httpx2.Response(200,json=completion())
    client=make_client(handler)
    try:
        result=client.generate('请回复连接成功')
        assert result.answer=='API连接成功。'
        assert result.usage['total_tokens']==15 and not result.truncated
        assert len(requests)==1
    finally:client.close()

@pytest.mark.parametrize('status',[400,401,403,404,429,500])
def test_provider_failure_no_retry_no_secret_output(status):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx2.Response(status,json={'error':{'message':FAKE_KEY,'type':'mock_error'}})
    client=make_client(handler)
    try:
        with pytest.raises(GenerationAPIError) as error:client.generate('test')
        assert len(requests)==1
        assert FAKE_KEY not in str(error.value)
    finally:client.close()

def test_local_dotenv_and_explicit_environment_precedence(tmp_path):
    root=prepare_root(tmp_path,f'DASHSCOPE_API_KEY={FAKE_KEY}\nDASHSCOPE_BASE_URL={URL}\nGENERATION_MODEL=qwen3.7-plus\n')
    settings=GenerationSettings.load(root,environ={})
    assert settings.model=='qwen3.7-plus' and settings.api_key==FAKE_KEY
    assert FAKE_KEY not in repr(settings)
    overridden=GenerationSettings.load(root,environ={'GENERATION_MODEL':'qwen3.7-plus-2026-05-26'})
    assert overridden.model=='qwen3.7-plus-2026-05-26'

@pytest.mark.parametrize('env',[f'DASHSCOPE_BASE_URL={URL}\n',f'DASHSCOPE_API_KEY={FAKE_KEY}\nDASHSCOPE_BASE_URL=https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1\n',f'DASHSCOPE_API_KEY={FAKE_KEY}\nDASHSCOPE_BASE_URL=https://unrelated.example/compatible-mode/v1\n'])
def test_invalid_config_blocks_request_before_network(tmp_path,env):
    root=prepare_root(tmp_path,env)
    with pytest.raises(GenerationConfigError):GenerationSettings.load(root,environ={})

def test_truncation_reported_without_second_call():
    requests=[]
    def handler(request):requests.append(request);return httpx2.Response(200,json=completion(reason='length'))
    client=make_client(handler)
    try:
        result=client.generate('test');assert result.truncated and len(requests)==1
    finally:client.close()

def test_timeout_no_retry():
    requests=[]
    def handler(request):
        requests.append(request);raise httpx2.ReadTimeout('mock timeout',request=request)
    client=make_client(handler)
    try:
        with pytest.raises(GenerationAPIError,match='超时'):client.generate('test')
        assert len(requests)==1
    finally:client.close()

def test_empty_prompt_never_calls_transport():
    def handler(request):raise AssertionError('Must not send an empty prompt')
    client=make_client(handler)
    try:
        with pytest.raises(GenerationConfigError):client.generate('  ')
    finally:client.close()

def test_gitignore_real_git_check(tmp_path):
    git=shutil.which('git')
    if not git:pytest.skip('git unavailable')
    subprocess.run([git,'init','-q',str(tmp_path)],check=True,capture_output=True)
    (tmp_path/'.gitignore').write_text((R/'.gitignore').read_text(encoding='utf-8'),encoding='utf-8')
    inputs=['.env','.env.local','.env.production','private.env','secrets/token.txt','.env.example']
    result=subprocess.run([git,'-C',str(tmp_path),'check-ignore','--no-index','--stdin','-z'],input=('\0'.join(inputs)+'\0').encode(),capture_output=True)
    ignored={x.decode() for x in result.stdout.split(b'\0') if x}
    assert set(inputs[:-1])<=ignored
    assert '.env.example' not in ignored
