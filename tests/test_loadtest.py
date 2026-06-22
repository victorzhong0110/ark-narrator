"""压测器纯函数：分位数 + 摘要。"""

from __future__ import annotations

from bench.loadtest import _pctl, summarize


def test_pctl():
    xs = [float(i) for i in range(1, 101)]   # 1..100
    assert _pctl(xs, 0.50) == 51.0
    assert _pctl(xs, 0.95) == 96.0
    assert _pctl([], 0.5) == 0.0


def test_summarize_has_key_fields():
    res = {
        "n": 100, "concurrency": 10, "wall_s": 1.0, "throughput_rps": 100.0,
        "ok": 98, "status": {200: 98, 429: 2},
        "p50_ms": 5.0, "p95_ms": 9.0, "p99_ms": 10.0,
        "served_by": {"a": 50, "b": 48},
    }
    out = summarize(res)
    assert "吞吐 100.0 req/s" in out
    assert "LB 分散" in out
    assert "p95 9.0ms" in out
