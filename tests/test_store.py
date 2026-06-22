"""存储层：内存 / SQLite / Redis(假 client) 共测 + TTL / 限流。"""

from __future__ import annotations

from app.store.memory import InMemoryStore
from app.store.redis import RedisStore
from app.store.sqlite import SQLiteStore


class FakeRedis:
    """最小可用的 redis 替身，覆盖 RedisStore 用到的命令。"""

    def __init__(self):
        self.kv: dict = {}
        self.lists: dict[str, list] = {}

    def rpush(self, k, v):
        self.lists.setdefault(k, []).append(v)

    def lrange(self, k, start, end):
        lst = self.lists.get(k, [])
        return lst[start:] if start < 0 else lst[start : (len(lst) if end == -1 else end + 1)]

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v):
        self.kv[k] = v

    def incr(self, k):
        return self.incrby(k, 1)

    def incrby(self, k, d):
        self.kv[k] = int(self.kv.get(k, 0)) + d
        return self.kv[k]

    def expire(self, k, t):
        pass

    def delete(self, k):
        self.kv.pop(k, None)
        self.lists.pop(k, None)


def _stores(tmp_path):
    return [
        InMemoryStore(),
        SQLiteStore(tmp_path / "s.sqlite"),
        RedisStore(client=FakeRedis()),
    ]


def test_history_roundtrip(tmp_path):
    for s in _stores(tmp_path):
        s.append_turn("sess", "user", "你好")
        s.append_turn("sess", "assistant", "老板好")
        h = s.history("sess", limit=10)
        assert [t.role for t in h] == ["user", "assistant"]
        assert h[1].content == "老板好"


def test_memory_roundtrip(tmp_path):
    for s in _stores(tmp_path):
        assert s.get_memory("u", "能天使") == ""
        s.set_memory("u", "能天使", "记得博士爱喝茶")
        assert s.get_memory("u", "能天使") == "记得博士爱喝茶"


def test_pair_turns_increment(tmp_path):
    for s in _stores(tmp_path):
        assert s.bump_pair_turns("u", "c") == 1
        assert s.bump_pair_turns("u", "c") == 2


def test_counters(tmp_path):
    for s in _stores(tmp_path):
        assert s.incr("k") == 1
        assert s.incr("k", 2) == 3
        assert s.get_int("k") == 3
        s.delete("k")
        assert s.get_int("k") == 0


def test_ttl_expiry_memory_and_sqlite(tmp_path):
    t = {"now": 100.0}
    clock = lambda: t["now"]  # noqa: E731
    for s in (InMemoryStore(clock=clock), SQLiteStore(tmp_path / "ttl.sqlite", clock=clock)):
        t["now"] = 100.0                # 每个 store 重置时钟
        s.incr("k", 1, ttl=10)
        assert s.get_int("k") == 1
        t["now"] = 200.0
        assert s.get_int("k") == 0      # 过期归零


def test_rate_allow_fixed_window(tmp_path):
    t = {"now": 0.0}
    s = InMemoryStore(clock=lambda: t["now"])
    assert all(s.rate_allow("u", limit=3, window_s=60) for _ in range(3))
    assert not s.rate_allow("u", limit=3, window_s=60)   # 第 4 次超限
    t["now"] = 61.0                                       # 进入下一窗口
    assert s.rate_allow("u", limit=3, window_s=60)
