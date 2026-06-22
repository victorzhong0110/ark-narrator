"""生产服务：探针 / 指标 / 鉴权 / 对话（scripted 后端，不下模型）。"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, *, auth_key: str | None = None):
    monkeypatch.setenv("ARK_BACKEND", "scripted")
    monkeypatch.setenv("ARK_STORE", "memory")
    monkeypatch.setenv("ARK_CLOUD_AUDIT", "false")
    monkeypatch.setenv("ARK_SCENE_TAGGER", "heuristic")
    monkeypatch.delenv("ARK_PROFILE", raising=False)
    if auth_key:
        monkeypatch.setenv("ARK_API_AUTH_KEY", auth_key)
    else:
        monkeypatch.delenv("ARK_API_AUTH_KEY", raising=False)
    import app.server as srv
    importlib.reload(srv)               # 让 settings 重新读环境
    return srv


@pytest.fixture
def client(monkeypatch):
    srv = _client(monkeypatch)
    with TestClient(srv.app) as c:
        yield c


def test_livez_and_readyz(client):
    assert client.get("/livez").json()["status"] == "alive"
    assert client.get("/readyz").json()["status"] == "ready"


def test_metrics_endpoint(client):
    client.post("/chat", json={"character": "能天使", "message": "你好"})
    body = client.get("/metrics").text
    assert "ark_requests_total" in body
    assert "ark_request_seconds" in body


def test_chat_works_and_labels(client):
    r = client.post("/chat", json={"character": "能天使", "message": "你好"}).json()
    assert r["response"]
    assert r["ai_label"]
    assert r["session_id"]


def test_unknown_character_400(client):
    assert client.post("/chat", json={"character": "查无此人", "message": "hi"}).status_code == 400


def test_request_id_header(client):
    assert "X-Request-ID" in client.get("/livez").headers


def test_auth_required_when_key_set(monkeypatch):
    srv = _client(monkeypatch, auth_key="secret123")
    with TestClient(srv.app) as c:
        # 无 key → 401
        assert c.post("/chat", json={"character": "能天使", "message": "hi"}).status_code == 401
        # 正确 key → 放行
        ok = c.post("/chat", json={"character": "能天使", "message": "hi"},
                    headers={"X-API-Key": "secret123"})
        assert ok.status_code == 200
