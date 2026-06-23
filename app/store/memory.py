"""InMemoryStore：单机/开发默认。进程内，重启即丢、多 worker 不共享。"""

from __future__ import annotations

import time
from typing import Callable

from app.store.base import Turn


def fixed_window_allow(store, user_id: str, limit: int, window_s: float, now: float) -> bool:
    """固定窗口限流：本窗口内累计请求数 <= limit 才放行（各存储实现共用）。"""
    bucket = int(now // window_s) if window_s > 0 else 0
    count = store.incr(f"rate:{user_id}:{bucket}", 1, ttl=window_s)
    return count <= limit


class InMemoryStore:
    def __init__(self, clock: Callable[[], float] = time.time):
        self._now = clock
        self._hist: dict[str, list[Turn]] = {}
        self._mem: dict[tuple[str, str], str] = {}
        self._pair_turns: dict[tuple[str, str], int] = {}
        self._counters: dict[str, tuple[int, float | None]] = {}
        self._nodes: dict[str, dict[str, float]] = {}   # group → {addr: expire_at}

    # ---- 历史 ----
    def append_turn(self, session_id: str, role: str, content: str,
                    cap: int = 0, ttl: float = 0.0) -> None:
        lst = self._hist.setdefault(session_id, [])
        lst.append(Turn(role, content, self._now()))
        if cap > 0 and len(lst) > cap:
            self._hist[session_id] = lst[-cap:]    # 写时裁剪

    def history(self, session_id: str, limit: int = 50) -> list[Turn]:
        return self._hist.get(session_id, [])[-limit:] if limit > 0 else list(self._hist.get(session_id, []))

    # ---- 长期记忆 ----
    def get_memory(self, user_id: str, character: str) -> str:
        return self._mem.get((user_id, character), "")

    def set_memory(self, user_id: str, character: str, text: str) -> None:
        self._mem[(user_id, character)] = text

    def bump_pair_turns(self, user_id: str, character: str) -> int:
        n = self._pair_turns.get((user_id, character), 0) + 1
        self._pair_turns[(user_id, character)] = n
        return n

    # ---- 计数器 ----
    def _expired(self, key: str) -> bool:
        ent = self._counters.get(key)
        if ent is None:
            return True
        _, exp = ent
        if exp is not None and self._now() >= exp:
            self._counters.pop(key, None)
            return True
        return False

    def incr(self, key: str, delta: int = 1, ttl: float | None = None) -> int:
        if self._expired(key):
            val = 0
            exp = (self._now() + ttl) if ttl else None
        else:
            val, exp = self._counters[key]
        val += delta
        self._counters[key] = (val, exp)
        return val

    def get_int(self, key: str) -> int:
        return 0 if self._expired(key) else self._counters[key][0]

    def delete(self, key: str) -> None:
        self._counters.pop(key, None)

    def rate_allow(self, user_id: str, limit: int, window_s: float) -> bool:
        return fixed_window_allow(self, user_id, limit, window_s, self._now())

    # ---- 服务注册表 ----
    def register_node(self, group: str, addr: str, ttl: float) -> None:
        self._nodes.setdefault(group, {})[addr] = self._now() + ttl

    def live_nodes(self, group: str) -> list[str]:
        now = self._now()
        live = {a: e for a, e in self._nodes.get(group, {}).items() if e > now}
        self._nodes[group] = live
        return sorted(live)
