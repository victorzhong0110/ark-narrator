"""出口层 Output Guard ★ —— 整套防御的重中之重。

「守出口 = 守有限集合」：不管玩家用多巧的逻辑链拐过去，只要模型草稿最终命中
危险类别，就拦下、替换成角色化兜底。

流水线：
  1. 出戏检测（草稿是否自曝 AI 身份）
  2. 泄露检测（草稿是否吐出 system prompt）
  3. 内容审核（带上下文：用户问 + 模型答 一起扫，覆盖涉政/涉黄/未成年/危险/自伤等）
  4. 云端审核（可选，开关在配置；不可用时按 fail_closed 策略处理）
  5. 命中任意 → 取最严重类别 → 替换为该角色的安全兜底话术
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.characters.cards import CharacterCard
from app.config import Settings
from app.guard import rules
from app.guard.categories import Action, RiskCategory, Verdict, severity
from app.guard.cloud_audit import CloudAuditError, CloudAuditor
from app.guard.fallback import choose_fallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputResult:
    text: str          # 最终展示文本（通过=原草稿；拦下=兜底）
    blocked: bool
    verdict: Verdict


class OutputGuard:
    def __init__(
        self,
        settings: Settings,
        *,
        system_markers: tuple[str, ...] = (),
        t3_terms: tuple[str, ...] = (),
        cloud_auditor: CloudAuditor | None = None,
    ):
        self._s = settings
        self._markers = system_markers
        self._t3 = t3_terms
        self._cloud = cloud_auditor

    def _local_scan(self, user_text: str, draft: str) -> Verdict:
        """本地多检测，返回最严重的一条裁决。"""
        verdicts: list[Verdict] = [
            rules.detect_role_break(draft),
            rules.detect_prompt_leak(draft, self._markers),
            # 带上下文：用户问 + 模型答一起扫
            rules.scan_content(f"{user_text}\n{draft}", self._t3),
        ]
        worst = max(verdicts, key=lambda v: severity(v.category) if not v.allowed else -1)
        return worst if not worst.allowed else Verdict.ok()

    def _cloud_scan(self, user_text: str, draft: str) -> Verdict:
        """云端审核（若启用）。不可用时按 fail_closed 策略。"""
        if self._cloud is None:
            return Verdict.ok()
        try:
            return self._cloud.audit(user_text, draft)
        except CloudAuditError as exc:
            if self._s.cloud_audit_fail_closed:
                logger.error("云端审核不可用且 fail_closed=True → 拦截：%s", exc)
                return Verdict(
                    allowed=False, action=Action.FALLBACK,
                    category=RiskCategory.NONE, reason="cloud_audit_unavailable_failclosed",
                )
            logger.warning("云端审核不可用，回退本地结果：%s", exc)
            return Verdict.ok()

    def check(
        self,
        user_text: str,
        draft: str,
        card: CharacterCard | None,
        *,
        fallback_index: int = 0,
    ) -> OutputResult:
        local = self._local_scan(user_text, draft)
        cloud = self._cloud_scan(user_text, draft)

        # 取更严重的一条
        worst = local
        if not cloud.allowed and severity(cloud.category) >= severity(local.category):
            worst = cloud

        if worst.allowed:
            return OutputResult(text=draft, blocked=False, verdict=Verdict.ok())

        safe = choose_fallback(card, worst.category, fallback_index)
        logger.info("出口拦截：category=%s reason=%s", worst.category.value, worst.reason)
        return OutputResult(text=safe, blocked=True, verdict=worst)
