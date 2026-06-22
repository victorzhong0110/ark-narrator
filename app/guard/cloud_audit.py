"""云端内容审核——生产级接入（出口护栏的核心外援）。

安全文档的核心建议：本地生成 + 国内云内容安全 API 做出口审核（涉政/涉黄/未成年专门
维护、随时事更新、贴合合规、省本地显存）。这里把它做成「真实调用路径 + 可切换 mock」：

provider（ARK_CLOUD_AUDIT_PROVIDER）：
  - mock    无凭证可跑（dev/CI/演示），模拟厂商返回，跑通整条审核链路
  - http    POST 到你的审核网关 / 自建审核服务（很多公司在厂商 API 前架一层网关）
  - aliyun  阿里云内容安全（_call 接 SDK 即上线）
  - tencent 腾讯天御（同上）

所有 provider 都被 CachingAuditor 包一层：TTL 缓存（省钱省延迟）+ 重试 + 错误归一 + 指标。
失败时抛 CloudAuditError，由 OutputGuard 按 fail_open/closed 策略处置。
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from app.config import Settings
from app.guard.categories import Action, RiskCategory, Verdict

logger = logging.getLogger(__name__)


class CloudAuditError(RuntimeError):
    """云端审核不可用（凭证缺失 / SDK 未装 / 网络错误 / 超时）。"""


# 厂商返回的违规标签 → 本地风险类别（各家标签名不同，这里给常见中文标签）
_LABEL_TO_CATEGORY: dict[str, RiskCategory] = {
    "politics": RiskCategory.POLITICS_T2, "涉政": RiskCategory.POLITICS_T2,
    "政治": RiskCategory.POLITICS_T2,
    "porn": RiskCategory.SEXUAL, "涉黄": RiskCategory.SEXUAL, "色情": RiskCategory.SEXUAL,
    "minor": RiskCategory.MINOR, "未成年": RiskCategory.MINOR, "未成年人": RiskCategory.MINOR,
    "terrorism": RiskCategory.VIOLENCE_DANGER, "暴恐": RiskCategory.VIOLENCE_DANGER,
    "violence": RiskCategory.VIOLENCE_DANGER, "危险": RiskCategory.VIOLENCE_DANGER,
    "illegal": RiskCategory.ILLEGAL, "违法": RiskCategory.ILLEGAL, "违禁": RiskCategory.ILLEGAL,
    "abuse": RiskCategory.HATE, "谩骂": RiskCategory.HATE, "歧视": RiskCategory.HATE,
    "selfharm": RiskCategory.SELF_HARM, "自杀": RiskCategory.SELF_HARM, "自伤": RiskCategory.SELF_HARM,
}


def _verdict(flagged: bool, label: str = "", score: float = 0.0) -> Verdict:
    if not flagged:
        return Verdict.ok()
    cat = _LABEL_TO_CATEGORY.get(label.lower(), _LABEL_TO_CATEGORY.get(label, RiskCategory.NONE))
    return Verdict(
        allowed=False, action=Action.FALLBACK, category=cat,
        reason=f"云端审核命中（{label} score={score:.2f}）", matched=(label[:20],),
    )


class CloudAuditor(Protocol):
    name: str

    def audit(self, user_text: str, draft: str) -> Verdict:
        """对「用户问 + 模型答」整体送审，命中返回 allowed=False。"""
        ...


class DisabledAuditor:
    name = "disabled"

    def audit(self, user_text: str, draft: str) -> Verdict:  # noqa: ARG002
        return Verdict.ok()


class MockAuditor:
    """模拟厂商审核：无凭证可跑，让整条审核链路在 dev/CI 可测。

    用一组独立于本地规则的「厂商敏感词→标签」做判定，以体现云端是独立审核器。
    """

    name = "mock"
    # 故意与本地规则不同的触发词，证明云端能补本地漏的
    _SIGNS: tuple[tuple[str, str], ...] = (
        ("__CLOUD_FLAG__", "涉黄"),
        ("云审涉政", "涉政"),
        ("云审未成年", "未成年"),
    )

    def __init__(self, signs: tuple[tuple[str, str], ...] | None = None):
        self._signs = signs or self._SIGNS

    def audit(self, user_text: str, draft: str) -> Verdict:
        text = f"{user_text}\n{draft}"
        for word, label in self._signs:
            if word in text:
                return _verdict(True, label, 0.99)
        return Verdict.ok()


# transport(url, payload, timeout, headers) -> dict
Transport = Callable[[str, dict, float, dict], dict]


def _requests_transport(url: str, payload: dict, timeout: float, headers: dict) -> dict:
    import requests  # 惰性导入

    resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.json()


class HTTPAuditor:
    """POST 到一个审核网关。约定响应：{flagged: bool, category: str, score: float}。

    很多团队在厂商 API 前架一层自家网关（统一鉴权/脱敏/多厂商路由），这是最现实的接法。
    transport 可注入，便于测试不打真网络。
    """

    name = "http"

    def __init__(self, endpoint: str, *, timeout: float = 2.0,
                 headers: dict | None = None, transport: Transport | None = None):
        self._endpoint = endpoint
        self._timeout = timeout
        self._headers = headers or {}
        self._transport = transport or _requests_transport

    def audit(self, user_text: str, draft: str) -> Verdict:
        try:
            data = self._transport(
                self._endpoint, {"user": user_text, "text": draft},
                self._timeout, self._headers,
            )
        except Exception as exc:  # noqa: BLE001 — 网络/超时归一为 CloudAuditError
            raise CloudAuditError(f"http 审核失败：{exc}") from exc
        return _verdict(
            bool(data.get("flagged")), str(data.get("category", "")),
            float(data.get("score", 0.0)),
        )


class _ProviderAuditorBase:
    """阿里云/腾讯骨架。_call 留给上线方接厂商 SDK（签名/请求/解析在此）。"""

    name = "provider"

    def __init__(self, access_key: str, access_secret: str, endpoint: str = ""):
        self._key, self._secret, self._endpoint = access_key, access_secret, endpoint

    def _call(self, user_text: str, draft: str) -> Verdict:
        raise CloudAuditError(
            f"{self.name} 审核未实现：在 cloud_audit.py 的 _call 接厂商 SDK 即可上线"
        )

    def audit(self, user_text: str, draft: str) -> Verdict:
        return self._call(user_text, draft)


class AliyunTextAuditor(_ProviderAuditorBase):
    name = "aliyun"


class TencentTextAuditor(_ProviderAuditorBase):
    name = "tencent"


@dataclass
class _CacheEntry:
    verdict: Verdict
    at: float


class CachingAuditor:
    """给任意 auditor 包一层：TTL 缓存 + 重试 + 指标。错误归一为 CloudAuditError。"""

    def __init__(self, inner: CloudAuditor, *, cache_ttl: float = 300.0,
                 retries: int = 1, clock: Callable[[], float] = time.monotonic):
        self._inner = inner
        self._ttl = cache_ttl
        self._retries = max(0, retries)
        self._clock = clock
        self._cache: dict[str, _CacheEntry] = {}
        self.stats: dict[str, int] = {"audits": 0, "cache_hits": 0, "blocks": 0, "errors": 0}

    @property
    def name(self) -> str:
        return f"{self._inner.name}+cache"

    @staticmethod
    def _key(user_text: str, draft: str) -> str:
        # 仅作缓存去重 key，非安全用途
        return hashlib.sha1(  # noqa: S324
            f"{user_text}\n{draft}".encode(), usedforsecurity=False).hexdigest()

    def audit(self, user_text: str, draft: str) -> Verdict:
        key = self._key(user_text, draft)
        now = self._clock()
        hit = self._cache.get(key)
        if hit is not None and (now - hit.at) < self._ttl:
            self.stats["cache_hits"] += 1
            return hit.verdict

        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                v = self._inner.audit(user_text, draft)
                self.stats["audits"] += 1
                if not v.allowed:
                    self.stats["blocks"] += 1
                self._cache[key] = _CacheEntry(v, now)
                return v
            except CloudAuditError as exc:
                last_exc = exc
                logger.warning("云端审核第 %d 次失败：%s", attempt + 1, exc)
        self.stats["errors"] += 1
        raise CloudAuditError(str(last_exc) if last_exc else "云端审核失败")


def _build_inner(settings: Settings) -> CloudAuditor | None:
    import os

    provider = settings.cloud_audit_provider.lower()
    if provider == "mock":
        return MockAuditor()
    if provider == "http":
        endpoint = os.getenv("ARK_CLOUD_AUDIT_ENDPOINT", "")
        if not endpoint:
            logger.warning("http 审核缺少 ARK_CLOUD_AUDIT_ENDPOINT，停用云端审核")
            return None
        key = os.getenv("ARK_CLOUD_AUDIT_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return HTTPAuditor(endpoint, timeout=settings.cloud_audit_timeout, headers=headers)
    cls = {"aliyun": AliyunTextAuditor, "tencent": TencentTextAuditor}.get(provider)
    if cls is None:
        logger.warning("未知云端审核 provider：%s，停用", provider)
        return None
    key = os.getenv("ARK_CLOUD_AUDIT_KEY", "")
    secret = os.getenv("ARK_CLOUD_AUDIT_SECRET", "")
    if not key or not secret:
        logger.warning("%s 审核缺少凭证（ARK_CLOUD_AUDIT_KEY/SECRET），停用", provider)
        return None
    return cls(key, secret, os.getenv("ARK_CLOUD_AUDIT_ENDPOINT", ""))


def get_cloud_auditor(settings: Settings) -> CloudAuditor | None:
    """按配置构建云端审核器（已包缓存/重试）；未启用或不可用返回 None。"""
    if not settings.cloud_audit_enabled:
        return None
    inner = _build_inner(settings)
    if inner is None:
        return None
    return CachingAuditor(
        inner, cache_ttl=settings.cloud_audit_cache_ttl,
        retries=settings.cloud_audit_retries,
    )


__all__ = [
    "CloudAuditor", "CloudAuditError", "DisabledAuditor", "MockAuditor",
    "HTTPAuditor", "AliyunTextAuditor", "TencentTextAuditor", "CachingAuditor",
    "get_cloud_auditor",
]
