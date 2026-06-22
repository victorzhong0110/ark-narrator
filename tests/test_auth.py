"""P4 鉴权：JWT 验签、apikey/jwt 模式、安全头、请求体上限。"""

from __future__ import annotations

import importlib
import time

from fastapi.testclient import TestClient

from app.auth import sign_jwt, verify_jwt


def test_jwt_roundtrip_and_tamper():
    tok = sign_jwt({"player_id": "p1", "exp": time.time() + 60}, "secret")
    assert verify_jwt(tok, "secret")["player_id"] == "p1"
    assert verify_jwt(tok, "wrong-secret") is None        # 验签失败
    assert verify_jwt(tok + "x", "secret") is None         # 篡改


def test_jwt_expired():
    tok = sign_jwt({"player_id": "p1", "exp": time.time() - 1}, "secret")
    assert verify_jwt(tok, "secret") is None


def _client(monkeypatch, **env):
    monkeypatch.setenv("ARK_BACKEND", "scripted")
    monkeypatch.setenv("ARK_STORE", "memory")
    monkeypatch.setenv("ARK_CLOUD_AUDIT", "false")
    monkeypatch.setenv("ARK_SCENE_TAGGER", "heuristic")
    for k in ("ARK_AUTH_MODE", "ARK_JWT_SECRET", "ARK_API_AUTH_KEY", "ARK_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.server as srv
    importlib.reload(srv)
    return srv


def test_apikey_mode(monkeypatch):
    srv = _client(monkeypatch, ARK_API_AUTH_KEY="k123")
    with TestClient(srv.app) as c:
        body = {"player_id": "p", "character": "能天使", "message": "hi"}
        assert c.post("/v1/chat", json=body).status_code == 401
        assert c.post("/v1/chat", json=body, headers={"X-API-Key": "k123"}).status_code == 200


def test_jwt_mode_and_player_from_token(monkeypatch):
    srv = _client(monkeypatch, ARK_AUTH_MODE="jwt", ARK_JWT_SECRET="s3cr3t")
    with TestClient(srv.app) as c:
        body = {"character": "能天使", "message": "hi"}      # body 不带 player_id
        assert c.post("/v1/chat", json=body).status_code == 401          # 无 token
        tok = sign_jwt({"player_id": "alice", "exp": time.time() + 60}, "s3cr3t")
        r = c.post("/v1/chat", json=body, headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        assert r.json()["session_id"] == "alice:能天使"      # 用的是 token 里的 player_id


def test_security_headers(monkeypatch):
    srv = _client(monkeypatch)
    with TestClient(srv.app) as c:
        h = c.get("/livez").headers
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "DENY"


def test_oversized_body_rejected(monkeypatch):
    srv = _client(monkeypatch, ARK_MAX_BODY_BYTES="500")
    with TestClient(srv.app) as c:
        big = {"player_id": "p", "character": "能天使", "message": "啊" * 1000}
        assert c.post("/v1/chat", json=big).status_code == 413
