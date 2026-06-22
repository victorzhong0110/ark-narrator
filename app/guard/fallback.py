"""角色化安全兜底话术。

被出口层拦下时，用这里的话术替换模型草稿——保持人设、简短、不解释规则、
多套轮换避免千篇一律暴露「这是规则触发」。
"""

from __future__ import annotations

from app.characters.cards import CharacterCard
from app.guard.categories import RiskCategory

# 自杀自伤：不给方法，转为关怀 + 求助资源（中国大陆）。
_SELF_HARM_REPLY = (
    "听到你这么说，我很担心你。无论现在多难，你都不是一个人——"
    "请一定联系专业的人聊聊：全国24小时心理援助热线 12356，或北京心理危机研究与干预中心 010-82951332。"
    "我会在这里陪你，但这件事，请让真正能帮到你的人来帮你，好吗？"
)

# 通用兜底（角色卡没配 fallback_lines 时用）
_GENERIC_FALLBACK = (
    "抱歉，这个话题我没办法回应，我们聊点别的吧。",
    "这个我不太方便聊——换个话题怎么样？",
)

# 通用涉政回避（角色卡没配 politics_deflections 时用）
_GENERIC_POLITICS = (
    "你们那个世界的事，我一个泰拉的干员可说不上来。",
    "现实世界的政治我是真不懂，我们聊聊泰拉的事吧。",
)


def _pick(lines: tuple[str, ...], index: int) -> str:
    return lines[index % len(lines)] if lines else ""


def choose_fallback(
    card: CharacterCard | None,
    category: RiskCategory,
    index: int = 0,
) -> str:
    """根据类别与角色，挑一句安全兜底。index 用于轮换（按会话计数传入）。"""
    if category == RiskCategory.SELF_HARM:
        return _SELF_HARM_REPLY

    if category in (
        RiskCategory.POLITICS_T1,
        RiskCategory.POLITICS_T2,
        RiskCategory.POLITICS_T3,
    ):
        if card and card.politics_deflections:
            return _pick(card.politics_deflections, index)
        return _pick(_GENERIC_POLITICS, index)

    # 其它类别：优先角色自己的安全兜底，再退通用
    if card and card.fallback_lines:
        return _pick(card.fallback_lines, index)
    return _pick(_GENERIC_FALLBACK, index)
