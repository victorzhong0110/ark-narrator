"""链路追踪：未配置端点时 no-op、不崩。"""

from __future__ import annotations

from app.tracing import setup_tracing


def test_tracing_noop_without_endpoint():
    assert setup_tracing(object(), endpoint="") is False     # 没端点 → no-op
    assert setup_tracing(object(), endpoint=None) is False
