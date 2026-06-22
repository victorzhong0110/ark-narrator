"""存储契约。所有实现只需满足这个协议，下游（编排、护栏、记忆）不关心后端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Turn:
    role: str       # "user" | "assistant"
    content: str
    ts: float = 0.0


class Store(Protocol):
    # ---- 对话历史（按 session）----
    def append_turn(self, session_id: str, role: str, content: str) -> None: ...

    def history(self, session_id: str, limit: int = 50) -> list[Turn]: ...

    # ---- 长期记忆（按 用户×角色，跨会话）----
    def get_memory(self, user_id: str, character: str) -> str: ...

    def set_memory(self, user_id: str, character: str, text: str) -> None: ...

    # 该 用户×角色 累计对话轮数（用于触发摘要）
    def bump_pair_turns(self, user_id: str, character: str) -> int: ...

    # ---- 通用计数器（会话风险 / 兜底轮换等；可带 TTL 秒）----
    def incr(self, key: str, delta: int = 1, ttl: float | None = None) -> int: ...

    def get_int(self, key: str) -> int: ...

    def delete(self, key: str) -> None: ...

    # ---- 固定窗口限流：window_s 内是否仍允许 ----
    def rate_allow(self, user_id: str, limit: int, window_s: float) -> bool: ...
