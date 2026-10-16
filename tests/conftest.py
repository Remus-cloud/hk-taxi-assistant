"""Tests use fake credentials and reject accidental external network access."""
import socket
import pytest

@pytest.fixture(autouse=True)
def offline_services(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY','offline-test-placeholder-not-a-real-key')
    monkeypatch.setenv('DASHSCOPE_BASE_URL','https://offline-test.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1')
    monkeypatch.setenv('GENERATION_MODEL','qwen3.7-plus-2026-05-26')
    original_connect=socket.socket.connect
    original_create=socket.create_connection
    def local(address):
        return isinstance(address,tuple) and address[0] in {'127.0.0.1','::1','localhost'}
    def guarded_connect(sock,address):
        if not local(address):raise RuntimeError('External network is disabled in tests')
        return original_connect(sock,address)
    def guarded_create(address,*args,**kwargs):
        if not local(address):raise RuntimeError('External network is disabled in tests')
        return original_create(address,*args,**kwargs)
    monkeypatch.setattr(socket.socket,'connect',guarded_connect)
    monkeypatch.setattr(socket,'create_connection',guarded_create)
