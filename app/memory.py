"""长期记忆：让角色「跨会话记得你」——竞品级体验的核心。

每隔 N 轮 用户×角色 对话，把「已有记忆 + 最近几轮」滚动摘要成一段简短记忆，存进 Store，
下次对话注入 system prompt。摘要用对话后端自己做（成本：每 N 轮一次短生成）。
"""

from __future__ import annotations

import logging

from app.llm.base import LLMBackend, Message
from app.store.base import Store

logger = logging.getLogger(__name__)


def _summary_prompt(character: str, max_chars: int) -> str:
    return (
        f"你在维护游戏角色「{character}」对一位玩家的长期记忆。请把【已有记忆】和【最近对话】"
        f"合并、更新成一段简短的长期记忆，便于下次对话时{character}还记得这位玩家。\n"
        f"要点：记住玩家是谁、聊过的重要的事、双方关系/称呼的进展、玩家的偏好；"
        f"用第三人称客观记述，不要编造，控制在 {max_chars} 字以内，只输出记忆本身。"
    )


class MemoryManager:
    def __init__(self, backend: LLMBackend, store: Store, *,
                 every: int = 6, max_chars: int = 600):
        self._backend = backend
        self._store = store
        self._every = every
        self._max_chars = max_chars

    def get(self, user_id: str, character: str) -> str:
        return self._store.get_memory(user_id, character)

    def observe(self, user_id: str, character: str, recent: list[Message]) -> None:
        """记一轮对话；到达节奏就滚动更新长期记忆。失败不影响主流程。"""
        n = self._store.bump_pair_turns(user_id, character)
        if self._every <= 0 or n % self._every != 0:
            return
        try:
            self._update(user_id, character, recent)
        except Exception as exc:  # noqa: BLE001 — 记忆更新失败不该打断对话
            logger.warning("长期记忆更新失败：%s", exc)

    def _update(self, user_id: str, character: str, recent: list[Message]) -> None:
        old = self._store.get_memory(user_id, character)
        convo = "\n".join(
            f"{'玩家' if t['role'] == 'user' else character}：{t['content']}" for t in recent
        )
        user = f"【已有记忆】\n{old or '（暂无）'}\n\n【最近对话】\n{convo}"
        summary = self._backend.generate(
            _summary_prompt(character, self._max_chars),
            [{"role": "user", "content": user}],
            max_tokens=max(64, self._max_chars // 2), temperature=0.3,
        ).strip()
        if not summary:
            return
        # 防记忆投毒：摘要里若含注入特征（玩家诱导写进长期记忆），不存
        from app.guard import rules
        if not rules.detect_injection(summary).allowed:
            logger.warning("长期记忆摘要含注入特征，丢弃不存（user=%s char=%s）", user_id, character)
            return
        self._store.set_memory(user_id, character, summary[: self._max_chars])
