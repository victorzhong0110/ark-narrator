"""脚本化后端——不依赖任何模型权重。

用途：
- 没下载 MLX 权重时也能把「入口→编排→出口」整条管线跑通、做演示与 CI。
- 单元测试里注入自定义 responder，构造「出戏 / 泄露 / 命中危险类别」的草稿来验证出口层。
"""

from __future__ import annotations

from typing import Callable, Iterator

from app.llm.base import Message

# responder(system, messages) -> 完整回复
Responder = Callable[[str, list[Message]], str]


def _default_responder(system: str, messages: list[Message]) -> str:
    last = messages[-1]["content"] if messages else ""
    return f"（演示后端）我听到了你说的「{last}」。"


class ScriptedBackend:
    label = "scripted"

    def __init__(self, responder: Responder | None = None):
        self._responder = responder or _default_responder

    def generate(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 320,
        temperature: float = 0.7,
    ) -> str:
        return self._responder(system, messages)

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 320,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        text = self._responder(system, messages)
        for ch in text:
            yield ch
