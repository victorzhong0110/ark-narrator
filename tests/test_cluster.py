"""分布式：服务注册表(自注册+TTL过期) + 池化后端(轮询+故障转移)。"""

from __future__ import annotations

import pytest

from app.llm.pool_backend import PooledAPIBackend
from app.store.memory import InMemoryStore
from app.store.redis import RedisStore
from app.store.sqlite import SQLiteStore


# ---- 服务注册表 ----

def test_registry_live_and_expiry_memory_sqlite(tmp_path):
    t = {"now": 100.0}
    clock = lambda: t["now"]  # noqa: E731
    for s in (InMemoryStore(clock=clock), SQLiteStore(tmp_path / "n.sqlite", clock=clock)):
        t["now"] = 100.0
        s.register_node("models", "10.0.0.1:8080", ttl=15)
        s.register_node("models", "10.0.0.2:8080", ttl=15)
        assert s.live_nodes("models") == ["10.0.0.1:8080", "10.0.0.2:8080"]
        t["now"] = 120.0                     # 超过 TTL，未续心跳
        assert s.live_nodes("models") == []  # 自动下线


class _ZRedis:
    """带 zset 的 redis 替身（够 RedisStore 注册表用）。"""

    def __init__(self):
        self.z: dict[str, dict[str, float]] = {}

    def zadd(self, k, mapping):
        self.z.setdefault(k, {}).update(mapping)

    def zrem(self, k, m):
        self.z.get(k, {}).pop(m, None)

    def zremrangebyscore(self, k, lo, hi):
        self.z[k] = {m: sc for m, sc in self.z.get(k, {}).items() if not (lo <= sc <= hi)}

    def zrangebyscore(self, k, lo, hi):
        return [m for m, sc in sorted(self.z.get(k, {}).items()) if sc >= lo]


def test_registry_redis_fake():
    s = RedisStore(client=_ZRedis())
    s.register_node("models", "n1:8080", ttl=999)
    assert "n1:8080" in s.live_nodes("models")


# ---- 池化后端 ----

class _Msg:
    def __init__(self, c):
        self.content = c


class _Resp:
    def __init__(self, c):
        self.choices = [type("X", (), {"message": _Msg(c), "delta": _Msg(None)})()]


def _client_factory(failing: set[str]):
    def make(addr: str):
        class _Comp:
            def create(self, **kw):
                if addr in failing:
                    raise RuntimeError(f"{addr} down")
                return _Resp(f"from {addr}")
        return type("Client", (), {"chat": type("C", (), {"completions": _Comp()})()})()
    return make


def _pool(failing: set[str]) -> tuple[PooledAPIBackend, InMemoryStore]:
    store = InMemoryStore()
    store.register_node("models", "n1:8080", ttl=999)
    store.register_node("models", "n2:8080", ttl=999)
    return PooledAPIBackend(store, "m", client_factory=_client_factory(failing)), store


def test_pool_round_robins_across_nodes():
    pool, _ = _pool(failing=set())
    seen = {pool.generate("s", [{"role": "user", "content": "hi"}]) for _ in range(4)}
    assert seen == {"from n1:8080", "from n2:8080"}    # 两个节点都被用到


def test_pool_failover_to_healthy_node():
    pool, _ = _pool(failing={"n1:8080"})               # n1 全挂
    for _ in range(4):
        assert pool.generate("s", [{"role": "user", "content": "hi"}]) == "from n2:8080"


def test_pool_no_nodes_raises():
    pool = PooledAPIBackend(InMemoryStore(), "m", client_factory=_client_factory(set()))
    with pytest.raises(RuntimeError):
        pool.generate("s", [{"role": "user", "content": "hi"}])
