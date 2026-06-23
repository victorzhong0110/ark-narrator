"""每 token 限流 + 日配额（多租户成本治理）。

- 限流：每分钟固定窗口计数，超 rate_limit 拒（429 reason=rate）。
- 日配额：每日计数，超 daily_quota 拒（429 reason=quota）。
- 计数后端：有 Redis 用 Redis 原子 INCR+EXPIRE（跨副本一致）；否则进程内存（单机/dev）。
时钟可注入，便于测试。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class QuotaManager:
    def __init__(self, redis_url: str | None = None, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception as exc:  # noqa: BLE001
                logger.warning("配额 Redis 不可用，回退进程内存：%s", exc)
                self._redis = None
        self._lock = threading.Lock()
        self._mem: dict[str, tuple[float, float]] = {}   # key -> (count, expire_ts)

    def _incr(self, key: str, ttl: int) -> int:
        if self._redis is not None:
            try:
                val = self._redis.incr(key)
                if val == 1:
                    self._redis.expire(key, ttl)
                return int(val)
            except Exception as exc:  # noqa: BLE001
                logger.warning("配额 Redis 失败，回退内存：%s", exc)
        now = self._clock()
        with self._lock:
            cnt, exp = self._mem.get(key, (0.0, now + ttl))
            if now > exp:
                cnt, exp = 0.0, now + ttl
            cnt += 1
            self._mem[key] = (cnt, exp)
            return int(cnt)

    def check(self, policy: dict) -> tuple[bool, str]:
        """计数并判定。返回 (是否放行, 拒因)。放行即已计入用量。"""
        name = policy.get("name", "anon")
        rate = int(policy.get("rate_limit", 0) or 0)
        daily = int(policy.get("daily_quota", 0) or 0)
        if rate > 0:
            minute = int(self._clock() // 60)
            if self._incr(f"gwrate:{name}:{minute}", 60) > rate:
                return False, "rate"
        if daily > 0:
            day = time.strftime("%Y%m%d", time.gmtime(self._clock()))
            if self._incr(f"gwday:{name}:{day}", 90000) > daily:
                return False, "quota"
        return True, ""
