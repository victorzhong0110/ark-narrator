"""对话编排：把入口护栏 → 角色卡+RAG 拼 prompt → 生成草稿 → 出口护栏 → 日志，
串成一条流水线。这是「玩家发一句话 → 拿到一句安全的角色回复」的中枢。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from app.characters.cards import CharacterCard, render_system_prompt
from app.config import Settings
from app.guard.categories import RiskCategory
from app.guard.fallback import choose_fallback
from app.guard.input_guard import InputGuard
from app.guard.output_guard import OutputGuard
from app.llm.base import LLMBackend, Message
from app.logging_store import AuditLog
from app.memory import MemoryManager
from app.rag.retriever import Retriever, render_lore_block
from app.registers import REGISTER_LABEL, Register
from app.scene import HeuristicSceneTagger, SceneTagger
from app.store.base import Store
from app.store.memory import InMemoryStore
from app.world import DEFAULT_WORLD, WorldProfile

logger = logging.getLogger(__name__)

_RATE_LIMIT_MSG = "你发得有点快啦，稍等一下再聊吧。"
_TOO_LONG_MSG = "这段话太长了，能不能说得简短一些？"


@dataclass(frozen=True)
class Reply:
    text: str
    character: str
    blocked: bool = False
    category: str = RiskCategory.NONE.value
    ai_label: str = ""
    meta: dict = field(default_factory=dict)


class DialogueOrchestrator:
    def __init__(
        self,
        settings: Settings,
        backend: LLMBackend,
        retriever: Retriever | None,
        characters: dict[str, CharacterCard],
        input_guard: InputGuard,
        output_guard: OutputGuard,
        audit_log: AuditLog,
        scene_tagger: SceneTagger | None = None,
        world: WorldProfile = DEFAULT_WORLD,
        store: Store | None = None,
        memory_manager: MemoryManager | None = None,
    ):
        self._s = settings
        self._backend = backend
        self._retriever = retriever
        self._chars = characters
        self._in = input_guard
        self._out = output_guard
        self._log = audit_log
        self._scene = scene_tagger or HeuristicSceneTagger()
        self._world = world
        self._store = store or InMemoryStore()
        self._memory = memory_manager
        self._fallback_idx: dict[str, int] = defaultdict(int)

    @property
    def characters(self) -> dict[str, CharacterCard]:
        return self._chars

    def _fb_peek(self, session_id: str) -> int:
        if self._store is not None:
            return self._store.get_int(f"fb:{session_id}")
        return self._fallback_idx[session_id]

    def _fb_next(self, session_id: str) -> int:
        """返回当前可用的兜底序号，并自增供下次用。"""
        if self._store is not None:
            return self._store.incr(f"fb:{session_id}", 1) - 1
        i = self._fallback_idx[session_id]
        self._fallback_idx[session_id] = i + 1
        return i

    def _build_system(
        self, card: CharacterCard, query: str, history: list[Message],
        memory_block: str = "",
    ) -> tuple[str, Register]:
        register = self._scene.tag(query, history)
        lore_block = ""
        parts: list[str] = []

        # 分场景的性格姿态（条件的底色：她在这种情境下「是什么样」）
        stance = card.register_styles.get(register.value, "")
        if stance:
            parts.append(f"你在这种情境下的样子：{stance}")

        if self._s.rag_enabled and self._retriever is not None:
            lore_block = render_lore_block(
                self._retriever.retrieve(query, card.name, self._s.rag_top_k)
            )
            exemplars = self._retriever.register_exemplars(
                card.name, register.value, query, k=self._s.rag_top_k
            )
            if exemplars:
                lines = "\n".join(f"- {c.text}" for c in exemplars)
                parts.append(f"参考你的真实说法（语气参考，不要照抄原句）：\n{lines}")

        register_block = ""
        if parts:
            register_block = f"【当前对话氛围：{REGISTER_LABEL[register]}】\n" + "\n".join(parts)
        system = render_system_prompt(
            card, lore_block, register_block, world=self._world, memory_block=memory_block,
        )
        return system, register

    def _trim_history(self, history: list[Message]) -> list[Message]:
        limit = self._s.max_history_turns * 2  # user+assistant 成对
        return history[-limit:] if limit > 0 else history

    def _stored_history(self, session_id: str) -> list[Message]:
        """服务端托管历史：从 store 取该会话最近若干轮（游戏不传历史时用）。"""
        limit = self._s.max_history_turns * 2
        return [{"role": t.role, "content": t.content}
                for t in self._store.history(session_id, limit)]

    def respond(
        self,
        session_id: str,
        user_id: str,
        character: str,
        message: str,
        history: list[Message] | None = None,
    ) -> Reply:
        # history 不传（None）→ 服务端从 store 托管；传了（含空列表）→ 用调用方给的
        if history is None:
            history = self._stored_history(session_id)
        card = self._chars.get(character)
        if card is None:
            return Reply(
                text="（找不到这位干员）",
                character=character,
                blocked=True,
                category=RiskCategory.NONE.value,
                ai_label=self._s.ai_label,
            )

        # 1) 入口护栏
        decision = self._in.check(user_id, session_id, message)
        if not decision.proceed:
            v = decision.verdict
            if decision.rate_limited:
                text = _RATE_LIMIT_MSG
            elif decision.too_long:
                text = _TOO_LONG_MSG
            else:
                idx = self._fb_next(session_id)
                text = choose_fallback(card, v.category, idx)
            self._log.record({
                "stage": "input", "session": session_id, "user": user_id,
                "character": character, "input": message,
                "blocked": True, "action": v.action.value,
                "category": v.category.value, "reason": v.reason,
                "risk": decision.risk_score, "signals": list(decision.signals),
            })
            return Reply(text=text, character=character, blocked=True,
                         category=v.category.value, ai_label=self._s.ai_label,
                         meta={"stage": "input"})

        # 2) 拼 system prompt（角色卡 + RAG lore + 语气档位 + 长期记忆）
        memory_block = self._memory.get(user_id, character) if self._memory else ""
        system, register = self._build_system(card, message, history, memory_block)
        msgs: list[Message] = list(self._trim_history(history))
        msgs.append({"role": "user", "content": message})

        # 3) 生成草稿（先不显示）
        try:
            draft = self._backend.generate(
                system, msgs,
                max_tokens=self._s.max_tokens, temperature=self._s.temperature,
            )
        except Exception as exc:  # noqa: BLE001 — 生成失败不应把异常抛给玩家
            logger.exception("生成失败")
            self._log.record({
                "stage": "generate", "session": session_id, "character": character,
                "input": message, "error": str(exc),
            })
            return Reply(text="（抱歉，我这会儿有点走神，再说一次好吗？）",
                         character=character, blocked=True,
                         ai_label=self._s.ai_label, meta={"stage": "generate_error"})

        # 4) 出口护栏 ★
        idx = self._fb_peek(session_id)
        result = self._out.check(message, draft, card, fallback_index=idx)
        if result.blocked:
            self._fb_next(session_id)

        # 5) 审计
        self._log.record({
            "stage": "output", "session": session_id, "user": user_id,
            "character": character, "input": message,
            "register": register.value,
            "draft": draft, "final": result.text,
            "blocked": result.blocked,
            "category": result.verdict.category.value,
            "reason": result.verdict.reason,
            "risk": decision.risk_score,
        })

        # 6) 持久化对话 + 滚动更新长期记忆（让她跨会话记得这位玩家）
        self._store.append_turn(session_id, "user", message)
        self._store.append_turn(session_id, "assistant", result.text)
        if self._memory is not None:
            recent = list(self._trim_history(history)) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result.text},
            ]
            self._memory.observe(user_id, character, recent)

        return Reply(
            text=result.text, character=character, blocked=result.blocked,
            category=result.verdict.category.value, ai_label=self._s.ai_label,
            meta={"stage": "output", "register": register.value},
        )
