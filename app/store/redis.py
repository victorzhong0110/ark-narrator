"""RedisStore：多 worker 共享状态（standard/flagship 档、Mac mini 集群）。

redis 为可选依赖：未安装时构造报清晰错误。client 可注入，便于测试不连真 Redis。
"""

from __future__ import annotations

import json

from app.store.base import Turn
from app.store.memory import fixed_window_allow


def _s(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


class RedisStore:
    def __init__(self, url: str = "redis://localhost:6379/0", *, client=None):
        if client is not None:
            self._r = client
        else:
            try:
                import redis  # 惰性导入
            except ImportError as exc:
                raise RuntimeError("RedisStore 需要 `pip install redis`") from exc
            self._r = redis.Redis.from_url(url)

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        self._r.rpush(f"hist:{session_id}", json.dumps({"role": role, "content": content}))

    def history(self, session_id: str, limit: int = 50) -> list[Turn]:
        start = -limit if limit > 0 else 0
        raw = self._r.lrange(f"hist:{session_id}", start, -1)
        out: list[Turn] = []
        for item in raw:
            try:
                d = json.loads(_s(item))
                out.append(Turn(d["role"], d["content"]))
            except Exception:  # noqa: BLE001
                continue
        return out

    def get_memory(self, user_id: str, character: str) -> str:
        v = self._r.get(f"mem:{user_id}:{character}")
        return _s(v) if v is not None else ""

    def set_memory(self, user_id: str, character: str, text: str) -> None:
        self._r.set(f"mem:{user_id}:{character}", text)

    def bump_pair_turns(self, user_id: str, character: str) -> int:
        return int(self._r.incr(f"pt:{user_id}:{character}"))

    def incr(self, key: str, delta: int = 1, ttl: float | None = None) -> int:
        val = int(self._r.incrby(key, delta))
        if ttl and val == delta:        # 首次创建时设过期
            self._r.expire(key, int(ttl) or 1)
        return val

    def get_int(self, key: str) -> int:
        v = self._r.get(key)
        return int(_s(v)) if v is not None else 0

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def rate_allow(self, user_id: str, limit: int, window_s: float) -> bool:
        import time
        return fixed_window_allow(self, user_id, limit, window_s, time.time())
