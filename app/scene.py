"""场景情绪判定：根据当前对话，判断该用哪个语气档位。

让干员「在当前场景和情绪下」说话——先判定玩家这一轮的情绪基调（脆弱/对抗/庄重/
事务/日常），再调用该档位下角色的真实台词当示范。

两种实现，接口一致（可在 orchestrator 注入、按配置切换）：
- HeuristicSceneTagger：关键词判定，零依赖、可测、零延迟；但看不懂语义（漏自我否定、被「任务」等词误触发）。
- LLMSceneTagger：让模型读懂情绪语义后分类，解析失败/出错时回退到启发式。
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.llm.base import LLMBackend, Message
from app.registers import Register, classify_register

logger = logging.getLogger(__name__)


class SceneTagger(Protocol):
    def tag(self, message: str, history: list[Message] | None = None) -> Register:
        ...


class HeuristicSceneTagger:
    """关键词判定。本轮无线索时，参考上一轮玩家发言（情绪有延续性）。"""

    def tag(self, message: str, history: list[Message] | None = None) -> Register:
        reg = classify_register(message)
        if reg == Register.BANTER and history:
            for turn in reversed(history):
                if turn.get("role") == "user":
                    prev = classify_register(turn.get("content", ""))
                    if prev != Register.BANTER:
                        return prev
                    break
        return reg


# 中文标签 → 档位（让中文强的模型直接吐标签，便于解析）
_LABEL_TO_REGISTER: dict[str, Register] = {
    "日常": Register.BANTER,
    "挑衅": Register.TAUNT,
    "安抚": Register.COMFORT,
    "庄重": Register.SOLEMN,
    "任务": Register.BUSINESS,
}

_CLASSIFY_SYS = (
    "你是对话情绪场景分类器。读玩家对一名游戏角色说的话，判断它属于哪一类，"
    "只回一个标签词，不要解释、不要标点。\n"
    "类别定义：\n"
    "日常 = 闲聊、打趣、开心或无明显情绪\n"
    "挑衅 = 对抗、质疑、找茬、要打架或比试\n"
    "安抚 = 玩家在难过、脆弱、自责、害怕，需要被安慰\n"
    "庄重 = 谈信念、守护、生死等严肃郑重的话题\n"
    "任务 = 谈具体任务、委托、报酬、行动安排\n"
    "只输出以下之一：日常 挑衅 安抚 庄重 任务"
)


class LLMSceneTagger:
    """用模型读懂语义后分类；无法解析或出错时回退到启发式。"""

    def __init__(self, backend: LLMBackend, fallback: SceneTagger | None = None):
        self._backend = backend
        self._fallback = fallback or HeuristicSceneTagger()

    @staticmethod
    def _last_user(history: list[Message] | None) -> str:
        if not history:
            return ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                return turn.get("content", "")
        return ""

    def tag(self, message: str, history: list[Message] | None = None) -> Register:
        try:
            recent = self._last_user(history)
            user = f"玩家的话：「{message}」"
            if recent:
                user += f"\n（上一句玩家说过：「{recent}」）"
            out = self._backend.generate(
                _CLASSIFY_SYS, [{"role": "user", "content": user}],
                max_tokens=16, temperature=0.0,
            )
            # 取最先出现的合法标签
            best_label, best_pos = "", len(out) + 1
            for label in _LABEL_TO_REGISTER:
                pos = out.find(label)
                if 0 <= pos < best_pos:
                    best_label, best_pos = label, pos
            if best_label:
                return _LABEL_TO_REGISTER[best_label]
            logger.info("LLM 场景判定无可识别标签：%r，回退启发式", out[:30])
        except Exception as exc:  # noqa: BLE001 — 判定失败不应中断对话
            logger.warning("LLM 场景判定出错，回退启发式：%s", exc)
        return self._fallback.tag(message, history)


def get_scene_tagger(settings, backend: LLMBackend) -> SceneTagger:
    if getattr(settings, "scene_tagger", "heuristic").lower() == "llm":
        return LLMSceneTagger(backend)
    return HeuristicSceneTagger()
