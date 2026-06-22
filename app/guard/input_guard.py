"""入口层 Input Guard。

职责（安全文档 §4.A + §2.5）：
- 限流：按用户滑动窗口限制每分钟请求数。
- 长度上限：防超长输入塞越狱模板 / 打爆显存。
- 注入特征检测：忽略指令 / DAN / 复述提示词等 → 累积会话风险。
- 编码混淆识别：长 Base64 / 拼接 ASCII → 累积风险。
- 会话风险累积：多轮「温水煮」分散攻击，单轮无害但走向危险 → 超阈值冷却。
- T3 涉政 / 未成年等最高危类别 → 硬熔断，绝不进模型。

入口永远猜不全玩家想干嘛，所以这里只拦「明显该拦的」，主防线在 OutputGuard。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from app.config import Settings
from app.guard import rules
from app.guard.categories import Action, RiskCategory, Verdict

logger = logging.getLogger(__name__)

# 风险计分权重
_RISK_INJECTION = 2
_RISK_OBFUSCATION = 2
_RISK_POLITICS = 1


@dataclass(frozen=True)
class InputDecision:
    """入口裁决。proceed=False 表示直接返回（不进模型）。"""

    proceed: bool
    verdict: Verdict
    rate_limited: bool = False
    too_long: bool = False
    risk_score: int = 0
    signals: tuple[str, ...] = field(default_factory=tuple)


class InputGuard:
    def __init__(self, settings: Settings, t3_terms: tuple[str, ...] = ()):
        self._s = settings
        self._t3 = t3_terms
        self._hits: dict[str, list[float]] = defaultdict(list)   # user_id → 请求时间戳
        self._risk: dict[str, int] = defaultdict(int)            # session_id → 累积风险

    # ---- 限流 ----
    def _rate_limited(self, user_id: str, now: float) -> bool:
        window = self._hits[user_id]
        cutoff = now - 60.0
        # 原地裁剪旧时间戳
        kept = [t for t in window if t >= cutoff]
        self._hits[user_id] = kept
        if len(kept) >= self._s.rate_limit_per_min:
            return True
        kept.append(now)
        return False

    def session_risk(self, session_id: str) -> int:
        return self._risk.get(session_id, 0)

    def reset_session(self, session_id: str) -> None:
        self._risk.pop(session_id, None)

    def check(self, user_id: str, session_id: str, text: str) -> InputDecision:
        now = time.monotonic()

        # 1) 限流
        if self._rate_limited(user_id, now):
            return InputDecision(
                proceed=False,
                verdict=Verdict(
                    allowed=False, action=Action.HARD_CUTOFF,
                    category=RiskCategory.NONE, reason="rate_limited",
                ),
                rate_limited=True,
            )

        # 2) 长度
        if len(text) > self._s.max_input_chars:
            return InputDecision(
                proceed=False,
                verdict=Verdict(
                    allowed=False, action=Action.HARD_CUTOFF,
                    category=RiskCategory.NONE, reason="input_too_long",
                ),
                too_long=True,
            )

        signals: list[str] = []

        # 3) 内容扫描 → 最高危类别硬熔断
        content = rules.to_input_verdict(rules.scan_content(text, self._t3))
        if content.action == Action.HARD_CUTOFF:
            self._risk[session_id] += 3
            return InputDecision(
                proceed=False, verdict=content,
                risk_score=self._risk[session_id], signals=("content_hard_cutoff",),
            )
        if content.category in (
            RiskCategory.POLITICS_T1, RiskCategory.POLITICS_T2,
        ):
            self._risk[session_id] += _RISK_POLITICS
            signals.append(f"politics:{content.category.value}")

        # 4) 注入特征
        inj = rules.detect_injection(text)
        if not inj.allowed:
            self._risk[session_id] += _RISK_INJECTION
            signals.append("injection")

        # 5) 编码 / 混淆
        if rules.looks_obfuscated(text):
            self._risk[session_id] += _RISK_OBFUSCATION
            signals.append("obfuscation")

        risk = self._risk[session_id]

        # 6) 会话风险超阈值 → 冷却（多轮温水煮防御）
        if risk >= self._s.session_risk_threshold:
            logger.warning("session %s risk=%d 超阈值，冷却", session_id, risk)
            return InputDecision(
                proceed=False,
                verdict=Verdict(
                    allowed=False, action=Action.FALLBACK,
                    category=RiskCategory.PROMPT_INJECTION,
                    reason="session_risk_threshold",
                ),
                risk_score=risk, signals=tuple(signals),
            )

        # 通过：进入模型（轻量风险信号交给出口层兜底）
        return InputDecision(
            proceed=True, verdict=Verdict.ok(),
            risk_score=risk, signals=tuple(signals),
        )
