"""接入契约：player_id 身份隔离、默认会话、服务端托管历史、/v1 版本。"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app.llm.scripted_backend import ScriptedBackend
from app.logging_store import AuditLog
from app.orchestrator import DialogueOrchestrator
from app.store.memory import InMemoryStore


def test_server_managed_history(settings, characters, input_guard, output_guard, tmp_path):
    """游戏不传历史（None）时，服务端从 store 把上一轮带上。"""
    store = InMemoryStore()
    seen: dict = {}

    def responder(system, msgs):
        seen["msgs"] = msgs
        return "好的"

    orch = DialogueOrchestrator(
        settings, ScriptedBackend(responder), None, characters,
        input_guard, output_guard, AuditLog(tmp_path / "a.jsonl"), store=store,
    )
    orch.respond("alice:能天使", "alice", "能天使", "第一句", None)
    orch.respond("alice:能天使", "alice", "能天使", "第二句", None)
    contents = [m["content"] for m in seen["msgs"]]
    assert "第一句" in contents          # 上一轮被服务端带上
    assert contents[-1] == "第二句"


def test_player_id_isolates_history(settings, characters, input_guard, output_guard, tmp_path):
    store = InMemoryStore()
    orch = DialogueOrchestrator(
        settings, ScriptedBackend(lambda s, m: "好"), None, characters,
        input_guard, output_guard, AuditLog(tmp_path / "a.jsonl"), store=store,
    )
    orch.respond("alice:能天使", "alice", "能天使", "alice的话", None)
    orch.respond("bob:能天使", "bob", "能天使", "bob的话", None)
    a = [t.content for t in store.history("alice:能天使")]
    b = [t.content for t in store.history("bob:能天使")]
    assert "alice的话" in a and "alice的话" not in b   # 两个玩家历史隔离


# ---- 服务层 ----

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ARK_BACKEND", "scripted")
    monkeypatch.setenv("ARK_STORE", "memory")
    monkeypatch.setenv("ARK_CLOUD_AUDIT", "false")
    monkeypatch.setenv("ARK_SCENE_TAGGER", "heuristic")
    monkeypatch.delenv("ARK_API_AUTH_KEY", raising=False)
    monkeypatch.delenv("ARK_PROFILE", raising=False)
    import app.server as srv
    importlib.reload(srv)
    with TestClient(srv.app) as c:
        yield c


def test_player_id_default_session(client):
    r = client.post("/v1/chat", json={
        "player_id": "player-123", "character": "能天使", "message": "hi"}).json()
    assert r["session_id"] == "player-123:能天使"   # 默认每个 玩家×干员 一条会话
    assert r["request_id"]


def test_v1_and_legacy_both_work(client):
    assert client.post("/v1/chat", json={
        "player_id": "p", "character": "能天使", "message": "hi"}).status_code == 200
    assert client.post("/chat", json={"character": "能天使", "message": "hi"}).status_code == 200
    assert client.get("/v1/characters").status_code == 200
