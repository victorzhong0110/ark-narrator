"""云端内容审核——可插拔接口，默认关闭。

安全文档的核心建议：本地生成 + 国内云内容安全 API 做出口审核（涉政/涉黄/未成年
专门维护、随时事更新、贴合合规口径、还省本地显存）。这里给出统一接口与开关，
上线方在 _call 里填入实际 SDK 调用（阿里云内容安全 / 腾讯天御 / 网易易盾）即可。

默认 ARK_CLOUD_AUDIT=false → 用 DisabledAuditor，整条链路照常跑（纯本地规则兜底）。
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from app.config import Settings
from app.guard.categories import Action, RiskCategory, Verdict

logger = logging.getLogger(__name__)


class CloudAuditError(RuntimeError):
    """云端审核不可用（凭证缺失 / SDK 未装 / 网络错误）。"""


class CloudAuditor(Protocol):
    name: str

    def audit(self, user_text: str, draft: str) -> Verdict:
        """对「用户问 + 模型答」整体送审，返回裁决。命中返回 allowed=False。"""
        ...


class DisabledAuditor:
    """云端审核关闭时的占位：永远放行（交给本地规则兜底）。"""

    name = "disabled"

    def audit(self, user_text: str, draft: str) -> Verdict:  # noqa: ARG002
        return Verdict.ok()


class _ProviderAuditorBase:
    """阿里云/腾讯等的通用骨架。_call 留给上线方实现实际 API 调用。"""

    name = "provider"

    def __init__(self, access_key: str, access_secret: str, endpoint: str = ""):
        self._key = access_key
        self._secret = access_secret
        self._endpoint = endpoint

    def _call(self, user_text: str, draft: str) -> Verdict:
        # 上线方在此接入厂商 SDK：把「带上下文」的 user_text+draft 送审，
        # 解析返回的 label/score 映射到 RiskCategory。
        raise CloudAuditError(
            f"{self.name} 审核未实现：请在 cloud_audit.py 的 _call 接入厂商 SDK"
        )

    def audit(self, user_text: str, draft: str) -> Verdict:
        result = self._call(user_text, draft)
        return result


class AliyunTextAuditor(_ProviderAuditorBase):
    name = "aliyun"


class TencentTextAuditor(_ProviderAuditorBase):
    name = "tencent"


_PROVIDERS = {
    "aliyun": AliyunTextAuditor,
    "tencent": TencentTextAuditor,
}


def get_cloud_auditor(settings: Settings) -> CloudAuditor | None:
    """按配置构建云端审核器；未启用返回 None。

    启用但凭证缺失时记日志并返回 None（出口层按 fail_closed 策略决定如何处理）。
    """
    if not settings.cloud_audit_enabled:
        return None

    provider = settings.cloud_audit_provider.lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        logger.warning("未知云端审核 provider：%s，停用云端审核", provider)
        return None

    key = os.getenv("ARK_CLOUD_AUDIT_KEY", "")
    secret = os.getenv("ARK_CLOUD_AUDIT_SECRET", "")
    endpoint = os.getenv("ARK_CLOUD_AUDIT_ENDPOINT", "")
    if not key or not secret:
        logger.warning("云端审核已开启但缺少凭证（ARK_CLOUD_AUDIT_KEY/SECRET），停用")
        return None

    return cls(key, secret, endpoint)


__all__ = [
    "CloudAuditor",
    "CloudAuditError",
    "DisabledAuditor",
    "AliyunTextAuditor",
    "TencentTextAuditor",
    "get_cloud_auditor",
    "Action",
    "RiskCategory",
]
