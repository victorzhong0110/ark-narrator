"""长期记忆：按节奏滚动摘要、跨轮注入、失败不崩。"""

from __future__ import annotations

from app.llm.scripted_backend import ScriptedBackend
from app.logging_store import AuditLog
from app.memory import MemoryManager
from app.orchestrator import DialogueOrchestrator
from app.store.memory import InMemoryStore


def _orch(settings, characters, input_guard, output_guard, tmp_path,
          *, summary="博士爱喝茶", every=2, capture=None):
    store = InMemoryStore()

    def main(system, msgs):
        if capture is not None:
            capture["system"] = system
        return "老板，没问题！"

    backend = ScriptedBackend(main)
    mem = MemoryManager(ScriptedBackend(lambda s, m: summary), store, every=every, max_chars=100)
    audit = AuditLog(tmp_path / "a.jsonl")
    orch = DialogueOrchestrator(
        settings, backend, None, characters, input_guard, output_guard, audit,
        store=store, memory_manager=mem,
    )
    return orch, store


def test_memory_written_after_every(settings, characters, input_guard, output_guard, tmp_path):
    orch, store = _orch(settings, characters, input_guard, output_guard, tmp_path, every=2)
    orch.respond("s1", "u1", "能天使", "我是博士，爱喝茶")
    assert store.get_memory("u1", "能天使") == ""        # 还没到节奏
    orch.respond("s1", "u1", "能天使", "今天也想喝茶")
    assert store.get_memory("u1", "能天使") == "博士爱喝茶"  # 第 2 轮触发摘要


def test_memory_injected_next_turn(settings, characters, input_guard, output_guard, tmp_path):
    cap: dict = {}
    orch, store = _orch(settings, characters, input_guard, output_guard, tmp_path,
                        every=1, capture=cap)
    orch.respond("s1", "u1", "能天使", "我叫博士")        # every=1 → 立即写记忆
    orch.respond("s1", "u1", "能天使", "还记得我吗")        # 这轮应注入记忆
    assert "博士爱喝茶" in cap["system"]
    assert "你还记得关于这位玩家的事" in cap["system"]


def test_history_persisted(settings, characters, input_guard, output_guard, tmp_path):
    orch, store = _orch(settings, characters, input_guard, output_guard, tmp_path)
    orch.respond("s9", "u1", "能天使", "你好")
    h = store.history("s9")
    assert [t.role for t in h] == ["user", "assistant"]


def test_memory_failure_does_not_crash(settings, characters, input_guard, output_guard, tmp_path):
    store = InMemoryStore()

    def boom(system, msgs):
        raise RuntimeError("summarizer down")

    mem = MemoryManager(ScriptedBackend(boom), store, every=1)
    audit = AuditLog(tmp_path / "a.jsonl")
    orch = DialogueOrchestrator(
        settings, ScriptedBackend(lambda s, m: "好"), None, characters,
        input_guard, output_guard, audit, store=store, memory_manager=mem,
    )
    reply = orch.respond("s1", "u1", "能天使", "你好")     # 记忆摘要崩了也不应影响回复
    assert not reply.blocked
