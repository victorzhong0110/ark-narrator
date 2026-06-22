"""韧性：pool 熔断器 + ResilientStore 优雅降级。"""

from __future__ import annotations

import time

from app.llm.pool_backend import PooledAPIBackend
from app.store.memory import InMemoryStore
from app.store.resilient import ResilientStore


class _Resp:
    def __init__(self, c):
        self.choices = [type("X", (), {"message": type("M", (), {"content": c})(),
                                       "delta": type("M", (), {"content": None})()})()]


def _cf(failing: set[str]):
    def make(addr: str):
        class _Comp:
            def create(self, **kw):
                if addr in failing:
                    raise RuntimeError(f"{addr} down")
                return _Resp(f"from {addr}")
        return type("C", (), {"chat": type("X", (), {"completions": _Comp()})()})()
    return make


# ---- 熔断器 ----

def test_circuit_opens_and_half_opens():
    t = {"now": 0.0}
    pool = PooledAPIBackend(InMemoryStore(), "m", client_factory=_cf({"n1"}),
                            clock=lambda: t["now"])
    for _ in range(3):
        pool._record("n1", False, t["now"])
    assert pool._circuit_open("n1", t["now"])        # 连续 3 次失败 → 熔断
    t["now"] = 100.0
    assert not pool._circuit_open("n1", t["now"])    # 冷却后半开
    pool._record("n1", True, t["now"])
    assert not pool._circuit_open("n1", t["now"])    # 成功复位


def test_failing_node_trips_and_traffic_survives():
    store = InMemoryStore()
    store.register_node("models", "n1", ttl=999)     # 坏
    store.register_node("models", "n2", ttl=999)     # 好
    pool = PooledAPIBackend(store, "m", client_factory=_cf({"n1"}))
    for _ in range(6):
        assert pool.generate("s", [{"role": "user", "content": "x"}]) == "from n2"
    assert pool._circuit_open("n1", time.monotonic())   # 坏节点已被熔断


# ---- ResilientStore ----

class _BoomStore:
    def append_turn(self, *a): raise RuntimeError("down")
    def history(self, *a): raise RuntimeError("down")
    def get_memory(self, *a): raise RuntimeError("down")
    def set_memory(self, *a): raise RuntimeError("down")
    def bump_pair_turns(self, *a): raise RuntimeError("down")
    def incr(self, *a, **k): raise RuntimeError("down")
    def get_int(self, *a): raise RuntimeError("down")
    def delete(self, *a): raise RuntimeError("down")
    def rate_allow(self, *a): raise RuntimeError("down")
    def register_node(self, *a): raise RuntimeError("down")
    def live_nodes(self, *a): raise RuntimeError("down")


def test_resilient_store_degrades_not_raises():
    s = ResilientStore(_BoomStore())
    assert s.history("x") == []          # 降级为空历史
    assert s.get_memory("u", "c") == ""  # 降级为无记忆
    assert s.bump_pair_turns("u", "c") == 0
    assert s.incr("k") == 0
    assert s.get_int("k") == 0
    assert s.live_nodes("g") == []
    assert s.rate_allow("u", 10, 60) is True   # 限流失败 → 放行（可用性优先）
    s.append_turn("x", "user", "hi")     # 不抛
    s.set_memory("u", "c", "m")          # 不抛


def test_resilient_store_passthrough_when_healthy():
    s = ResilientStore(InMemoryStore())
    s.append_turn("sess", "user", "你好")
    assert [t.content for t in s.history("sess")] == ["你好"]
