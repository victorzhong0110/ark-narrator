"""推理网关：OpenAI 兼容形状、鉴权、models、流式。"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, *, tokens: str | None = None):
    monkeypatch.setenv("GW_BACKEND", "scripted")
    if tokens:
        monkeypatch.setenv("GW_TOKENS", tokens)
    else:
        monkeypatch.delenv("GW_TOKENS", raising=False)
    monkeypatch.setenv("GW_TOKENS_FILE", "gateway/__nonexistent__.yaml")  # 避免读到真 tokens 文件
    import gateway.app as gw
    importlib.reload(gw)
    return gw


@pytest.fixture
def client(monkeypatch):
    gw = _client(monkeypatch)
    with TestClient(gw.app) as c:
        yield c


def test_chat_completions_openai_shape(client):
    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [
            {"role": "system", "content": "你是能天使"},
            {"role": "user", "content": "你好"}]}).json()
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"]["role"] == "assistant"
    assert "你好" in r["choices"][0]["message"]["content"]   # scripted 回显
    assert "usage" in r


def test_models_endpoint(client):
    r = client.get("/v1/models").json()
    assert r["data"][0]["id"]


def test_healthz(client):
    assert client.get("/healthz").json()["backend"] == "scripted"


def test_auth_required_when_tokens_set(monkeypatch):
    gw = _client(monkeypatch, tokens="tok-abc")
    with TestClient(gw.app) as c:
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert c.post("/v1/chat/completions", json=body).status_code == 401
        ok = c.post("/v1/chat/completions", json=body,
                    headers={"Authorization": "Bearer tok-abc"})
        assert ok.status_code == 200


def test_streaming_openai_chunks(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "嗨"}], "stream": True}) as s:
        body = "".join(s.iter_text())
    assert "data:" in body
    assert "chat.completion.chunk" in body
    assert "[DONE]" in body
