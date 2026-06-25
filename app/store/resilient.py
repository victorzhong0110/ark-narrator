"""ResilientStore：给任意 Store 包一层「失败不抛、优雅降级」。

Redis 抖动/瞬断时，记忆/历史/限流这些非关键操作不应让整个请求 500——而是这一轮
跳过（返回安全默认值），对话照常进行。失败计入 ark_store_errors_total 便于告警。
"""

from __future__ import annotations

import logging
import threading
import time

from app.metrics import METRICS
from app.store.base import Store, Turn

logger = logging.getLogger(__name__)


def _safe(op: str, default):
    """装饰：调用失败时记日志+指标并返回 default，不抛。"""
    def deco(fn):
        def wrap(self, *a, **kw):
            try:
                return fn(self, *a, **kw)
            except Exception as exc:  # noqa: BLE001 — 存储抖动不该打断请求
                METRICS.inc("ark_store_errors_total", {"op": op})
                logger.warning("store.%s 失败，降级：%s", op, exc)
                return default() if callable(default) else default
        return wrap
    return deco


class ResilientStore:
    def __init__(self, inner: Store):
        self._inner = inner
        self._lock = threading.Lock()
        self._local_rl: dict[tuple[str, int], int] = {}   # 限流降级用的进程内计数

    @_safe("append_turn", None)
    def append_turn(self, session_id: str, role: str, content: str,
                    cap: int = 0, ttl: float = 0.0) -> None:
        self._inner.append_turn(session_id, role, content, cap, ttl)

    @_safe("history", list)
    def history(self, session_id: str, limit: int = 50) -> list[Turn]:
        return self._inner.history(session_id, limit)

    @_safe("get_memory", "")
    def get_memory(self, user_id: str, character: str) -> str:
        return self._inner.get_memory(user_id, character)

    @_safe("set_memory", None)
    def set_memory(self, user_id: str, character: str, text: str) -> None:
        self._inner.set_memory(user_id, character, text)

    @_safe("bump_pair_turns", 0)
    def bump_pair_turns(self, user_id: str, character: str) -> int:
        return self._inner.bump_pair_turns(user_id, character)

    @_safe("incr", 0)
    def incr(self, key: str, delta: int = 1, ttl: float | None = None) -> int:
        return self._inner.incr(key, delta, ttl)

    @_safe("get_int", 0)
    def get_int(self, key: str) -> int:
        return self._inner.get_int(key)

    @_safe("delete", None)
    def delete(self, key: str) -> None:
        self._inner.delete(key)

    def rate_allow(self, user_id: str, limit: int, window_s: float) -> bool:
        # 限流是安全控制：Redis 抖断时不能直接放行（fail-open），降级为进程内本地限流
        # （跨副本不精确，但每实例仍兜底，防 Redis 故障期被滥用刷穿）。
        try:
            return self._inner.rate_allow(user_id, limit, window_s)
        except Exception as exc:  # noqa: BLE001
            METRICS.inc("ark_store_errors_total", {"op": "rate_allow"})
            logger.warning("store.rate_allow 失败，降级进程内本地限流：%s", exc)
            return self._local_rate_allow(user_id, limit, window_s)

    def _local_rate_allow(self, user_id: str, limit: int, window_s: float) -> bool:
        if limit <= 0:
            return True
        bucket = int(time.time() // max(1.0, window_s))
        key = (user_id, bucket)
        with self._lock:
            n = self._local_rl.get(key, 0) + 1
            self._local_rl[key] = n
            if len(self._local_rl) > 10000:    # 清理过期桶，防无界增长
                self._local_rl = {k: v for k, v in self._local_rl.items() if k[1] >= bucket}
            return n <= limit

    @_safe("register_node", None)
    def register_node(self, group: str, addr: str, ttl: float) -> None:
        self._inner.register_node(group, addr, ttl)

    @_safe("live_nodes", list)
    def live_nodes(self, group: str) -> list[str]:
        return self._inner.live_nodes(group)
