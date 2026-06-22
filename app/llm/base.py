"""LLM 后端抽象。"""

from __future__ import annotations

from typing import Iterator, Protocol, TypedDict


class Message(TypedDict):
    role: str       # "user" | "assistant"
    content: str


class LLMBackend(Protocol):
    """对话后端协议。实现方负责把 system + 多轮 messages 喂给模型。"""

    label: str

    def generate(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 320,
        temperature: float = 0.7,
    ) -> str:
        """同步生成完整回复（用于「先出草稿再审核」）。"""
        ...

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 320,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """逐 token 流式生成。"""
        ...
