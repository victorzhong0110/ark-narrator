"""指标模块：计数器 / 直方图 / Prometheus 文本渲染。"""

from __future__ import annotations

from app.metrics import Metrics


def test_counter_renders():
    m = Metrics()
    m.inc("ark_requests_total", {"path": "/chat", "status": "200"})
    m.inc("ark_requests_total", {"path": "/chat", "status": "200"})
    out = m.render()
    assert "# TYPE ark_requests_total counter" in out
    assert 'ark_requests_total{path="/chat",status="200"} 2' in out


def test_histogram_renders():
    m = Metrics()
    m.observe("ark_request_seconds", 0.2, {"path": "/chat"})
    m.observe("ark_request_seconds", 3.0, {"path": "/chat"})
    out = m.render()
    assert "# TYPE ark_request_seconds histogram" in out
    assert "ark_request_seconds_count" in out
    assert "ark_request_seconds_sum" in out
    assert 'le="+Inf"' in out


def test_histogram_bucket_cumulative():
    m = Metrics()
    m.observe("lat", 0.04)      # 落入 0.05 桶
    out = m.render()
    assert 'lat_bucket{le="0.05"} 1' in out
    assert "lat_count 1" in out
