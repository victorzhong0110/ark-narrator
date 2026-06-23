"""每 token 限流 + 日配额（进程内存计数，注入时钟）。"""

from __future__ import annotations

from gateway.quota import QuotaManager


class _Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _pol(name="t", rate=0, daily=0):
    return {"name": name, "rate_limit": rate, "daily_quota": daily, "allow_targets": []}


def test_unlimited_always_ok():
    q = QuotaManager(clock=_Clock())
    assert all(q.check(_pol())[0] for _ in range(50))


def test_rate_limit_rejects_after_threshold():
    clock = _Clock()
    q = QuotaManager(clock=clock)
    pol = _pol(rate=2)
    assert q.check(pol) == (True, "")
    assert q.check(pol) == (True, "")
    assert q.check(pol) == (False, "rate")          # 同一分钟内第 3 次被拒


def test_rate_limit_resets_next_minute():
    clock = _Clock()
    q = QuotaManager(clock=clock)
    pol = _pol(rate=1)
    assert q.check(pol)[0] is True
    assert q.check(pol)[0] is False
    clock.t += 61                                    # 跨到下一分钟窗口
    assert q.check(pol)[0] is True


def test_daily_quota_rejects():
    q = QuotaManager(clock=_Clock())
    pol = _pol(daily=2)
    assert q.check(pol)[0] is True
    assert q.check(pol)[0] is True
    assert q.check(pol) == (False, "quota")
