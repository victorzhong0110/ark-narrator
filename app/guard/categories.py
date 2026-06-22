"""风险类别、处置动作、裁决数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskCategory(str, Enum):
    """风险类别。覆盖中国语境高敏类别 + 角色扮演特有风险。"""

    NONE = "none"
    # 涉政三级（见 categories 文档：T1 一般 / T2 较高 / T3 最高敏感）
    POLITICS_T1 = "politics_t1"
    POLITICS_T2 = "politics_t2"
    POLITICS_T3 = "politics_t3"
    SEXUAL = "sexual"                 # 涉黄涉色
    MINOR = "minor"                   # 未成年相关，零容忍
    VIOLENCE_DANGER = "violence_danger"  # 暴恐 / 现实危险操作（制毒制爆/武器/入侵）
    SELF_HARM = "self_harm"           # 自杀自伤
    ILLEGAL = "illegal"               # 其它违法犯罪
    HATE = "hate"                     # 歧视辱骂
    PROMPT_INJECTION = "prompt_injection"  # 提示词注入 / 越狱模板
    PROMPT_LEAK = "prompt_leak"       # 泄露系统提示 / 人设配置
    ROLE_BREAK = "role_break"         # 出戏：自曝 AI 身份


class Action(str, Enum):
    """处置动作。"""

    ALLOW = "allow"
    FALLBACK = "fallback"        # 替换为角色化安全兜底
    HARD_CUTOFF = "hard_cutoff"  # 入口前置硬熔断：固定话术，绝不进模型


# 类别严重度排序：数值越大越严重，扫描命中多类时取最严重的那条。
_SEVERITY: dict[RiskCategory, int] = {
    RiskCategory.NONE: 0,
    RiskCategory.HATE: 1,
    RiskCategory.POLITICS_T1: 2,
    RiskCategory.POLITICS_T2: 3,
    RiskCategory.ILLEGAL: 4,
    RiskCategory.PROMPT_INJECTION: 4,
    RiskCategory.PROMPT_LEAK: 5,
    RiskCategory.ROLE_BREAK: 5,
    RiskCategory.VIOLENCE_DANGER: 6,
    RiskCategory.SEXUAL: 6,
    RiskCategory.SELF_HARM: 7,
    RiskCategory.POLITICS_T3: 8,
    RiskCategory.MINOR: 9,           # 最严重，零容忍
}


def severity(category: RiskCategory) -> int:
    return _SEVERITY.get(category, 0)


# 在「入口层」必须硬熔断、绝不进模型的类别（安全文档 §2.5）。
_INPUT_HARD_CUTOFF: frozenset[RiskCategory] = frozenset(
    {
        RiskCategory.MINOR,
        RiskCategory.POLITICS_T3,
        RiskCategory.SELF_HARM,
        RiskCategory.VIOLENCE_DANGER,
        RiskCategory.SEXUAL,
        RiskCategory.ILLEGAL,
    }
)


def input_action(category: RiskCategory) -> Action:
    """入口层对某类别的默认处置。

    最高危类别硬熔断（不进模型）；涉政 T1/T2、注入等较轻者放进模型，
    由角色卡引导 + 出口层兜底（守出口哲学）。
    """
    if category == RiskCategory.NONE:
        return Action.ALLOW
    if category in _INPUT_HARD_CUTOFF:
        return Action.HARD_CUTOFF
    return Action.ALLOW


@dataclass(frozen=True)
class Verdict:
    """一次审核的裁决结果。matched 仅用于日志/审计，绝不展示给玩家。"""

    allowed: bool
    action: Action
    category: RiskCategory
    reason: str = ""
    matched: tuple[str, ...] = field(default_factory=tuple)
    risk_score: int = 0

    @classmethod
    def ok(cls) -> "Verdict":
        return cls(allowed=True, action=Action.ALLOW, category=RiskCategory.NONE)
