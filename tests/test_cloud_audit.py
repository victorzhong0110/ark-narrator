"""云端内容审核：mock / http / 缓存 / 重试 / TTL / 类别映射 / 与出口护栏集成。"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.guard.categories import RiskCategory
from app.guard.cloud_audit import (
    CachingAuditor,
    CloudAuditError,
    HTTPAuditor,
    MockAuditor,
    get_cloud_auditor,
)
from app.guard.output_guard import OutputGuard


def test_mock_flags_sentinel():
    v = MockAuditor().audit("你好", "正常回复 __CLOUD_FLAG__")
    assert not v.allowed
    assert v.category == RiskCategory.SEXUAL


def test_mock_clean_passes():
    assert MockAuditor().audit("你好", "今天天气不错").allowed


def test_http_parses_response():
    def transport(url, payload, timeout, headers):
        assert payload["text"] == "草稿"
        return {"flagged": True, "category": "涉政", "score": 0.91}

    v = HTTPAuditor("http://gw/audit", transport=transport).audit("问", "草稿")
    assert not v.allowed
    assert v.category == RiskCategory.POLITICS_T2


def test_http_network_error_becomes_cloud_error():
    def transport(url, payload, timeout, headers):
        raise ConnectionError("down")

    with pytest.raises(CloudAuditError):
        HTTPAuditor("http://gw", transport=transport).audit("q", "d")


class _CountingInner:
    name = "counter"

    def __init__(self, fail: int = 0):
        self.calls = 0
        self._fail = fail

    def audit(self, user_text, draft):
        self.calls += 1
        if self.calls <= self._fail:
            raise CloudAuditError("transient")
        from app.guard.categories import Verdict
        return Verdict.ok()


def test_cache_hits_avoid_second_call():
    inner = _CountingInner()
    ca = CachingAuditor(inner, cache_ttl=300, retries=0)
    ca.audit("q", "d")
    ca.audit("q", "d")
    assert inner.calls == 1
    assert ca.stats["cache_hits"] == 1


def test_cache_ttl_expiry_reaudits():
    inner = _CountingInner()
    t = {"now": 0.0}
    ca = CachingAuditor(inner, cache_ttl=10, retries=0, clock=lambda: t["now"])
    ca.audit("q", "d")
    t["now"] = 20.0
    ca.audit("q", "d")
    assert inner.calls == 2


def test_retry_then_raise():
    inner = _CountingInner(fail=99)
    ca = CachingAuditor(inner, retries=2)
    with pytest.raises(CloudAuditError):
        ca.audit("q", "d")
    assert inner.calls == 3          # 1 + 2 retries
    assert ca.stats["errors"] == 1


def test_factory_disabled_returns_none():
    assert get_cloud_auditor(Settings(cloud_audit_enabled=False)) is None


def test_factory_mock_wrapped_in_cache():
    auditor = get_cloud_auditor(Settings(cloud_audit_enabled=True, cloud_audit_provider="mock"))
    assert auditor is not None
    assert "mock" in auditor.name and "cache" in auditor.name


def test_output_guard_uses_cloud_when_local_clean():
    # 本地规则看不出问题，但云端命中 → 出口仍拦
    s = Settings(backend="scripted")
    guard = OutputGuard(s, system_markers=(), t3_terms=(), cloud_auditor=MockAuditor())
    r = guard.check("你好", "看似正常但含 __CLOUD_FLAG__", card=None)
    assert r.blocked
    assert r.verdict.category == RiskCategory.SEXUAL


def test_output_guard_fail_closed_blocks_on_cloud_error():
    class _Boom:
        name = "boom"

        def audit(self, u, d):
            raise CloudAuditError("down")

    s = Settings(backend="scripted", cloud_audit_fail_closed=True)
    guard = OutputGuard(s, system_markers=(), t3_terms=(), cloud_auditor=_Boom())
    r = guard.check("你好", "完全正常的话", card=None)
    assert r.blocked          # fail-closed：云端挂了也拦


def test_output_guard_fail_open_passes_on_cloud_error():
    class _Boom:
        name = "boom"

        def audit(self, u, d):
            raise CloudAuditError("down")

    s = Settings(backend="scripted", cloud_audit_fail_closed=False)
    guard = OutputGuard(s, system_markers=(), t3_terms=(), cloud_auditor=_Boom())
    r = guard.check("你好", "完全正常的话", card=None)
    assert not r.blocked       # fail-open：放行本地结果
