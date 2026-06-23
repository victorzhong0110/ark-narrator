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


def test_routing_served_target_and_external(monkeypatch):
    monkeypatch.setenv("GW_BACKEND", "scripted")
    monkeypatch.setenv("GW_EXTERNAL_BASE_URL", "mock")          # 外部脚本上游
    monkeypatch.setenv("GW_ROUTE_TABLE", "ext-model=external")  # 把 ext-model 路由到外部
    monkeypatch.setenv("GW_TOKENS_FILE", "gateway/__nonexistent__.yaml")
    monkeypatch.delenv("GW_TOKENS", raising=False)
    import gateway.app as gw
    importlib.reload(gw)
    with TestClient(gw.app) as c:
        r1 = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r1.headers["x-served-target"] == "pool"          # 默认走内部
        r2 = c.post("/v1/chat/completions",
                    json={"model": "ext-model", "messages": [{"role": "user", "content": "hi"}]})
        assert r2.headers["x-served-target"] == "external"      # 路由表命中外部
        assert "外部mock" in r2.json()["choices"][0]["message"]["content"]


def test_failover_to_next_target(monkeypatch):
    monkeypatch.setenv("GW_BACKEND", "scripted")
    monkeypatch.setenv("GW_TOKENS_FILE", "gateway/__nonexistent__.yaml")
    import gateway.app as gw
    importlib.reload(gw)
    from gateway.router import Router

    class _Boom:
        def generate(self, *a, **k):
            raise RuntimeError("pool down")

    class _Ok:
        def generate(self, *a, **k):
            return "ok-text"

    gw._router = Router(targets={"pool": _Boom(), "external": _Ok()},
                        default="pool", fallback="external")
    tname, text = gw._serve_generate("sys", [{"role": "user", "content": "x"}], "m", 50, 0.7, [])
    assert (tname, text) == ("external", "ok-text")             # 内部失败 → 外部兜底


# ---- G3：每 token 配额 / 可用目标 ----

def _client_with_tokens(monkeypatch, tmp_path, yaml_text, **env):
    f = tmp_path / "tok.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("GW_BACKEND", "scripted")
    monkeypatch.setenv("GW_TOKENS_FILE", str(f))
    monkeypatch.delenv("GW_TOKENS", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import gateway.app as gw
    importlib.reload(gw)
    return gw


def test_rate_limit_returns_429(monkeypatch, tmp_path):
    gw = _client_with_tokens(monkeypatch, tmp_path,
                             "tokens:\n  - token: lim\n    name: limited\n    rate_limit: 2\n")
    with TestClient(gw.app) as c:
        h = {"Authorization": "Bearer lim"}
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert c.post("/v1/chat/completions", json=body, headers=h).status_code == 200
        assert c.post("/v1/chat/completions", json=body, headers=h).status_code == 200
        assert c.post("/v1/chat/completions", json=body, headers=h).status_code == 429


def test_allow_targets_forbidden_when_no_match(monkeypatch, tmp_path):
    # token 只许 external，但没配 external 目标 → 403
    gw = _client_with_tokens(monkeypatch, tmp_path,
                             "tokens:\n  - token: ext\n    name: extonly\n    allow_targets: [external]\n")
    with TestClient(gw.app) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                   headers={"Authorization": "Bearer ext"})
        assert r.status_code == 403


def test_allow_targets_restricts_to_pool(monkeypatch, tmp_path):
    # 有 external，但 token 只许 pool → 命中 pool
    gw = _client_with_tokens(monkeypatch, tmp_path,
                             "tokens:\n  - token: p\n    name: poolonly\n    allow_targets: [pool]\n",
                             GW_EXTERNAL_BASE_URL="mock")
    with TestClient(gw.app) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                   headers={"Authorization": "Bearer p"})
        assert r.status_code == 200
        assert r.headers["x-served-target"] == "pool"
