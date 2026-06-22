"""OpenAI 兼容的 API 后端（解锁中高部署档位：云/混合/旗舰）。

很多国内模型（MiniMax、DeepSeek、通义等）都提供 OpenAI 兼容端点，改 base_url + key + model
即可接入。client 可注入，便于测试不打真网络。
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

from app.llm.base import Message

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    idx = text.find("<think>")
    if idx != -1:
        text = text[:idx]
    return text.strip()


class APIBackend:
    def __init__(self, model: str, base_url: str, api_key: str, *,
                 disable_thinking: bool = False, client=None):
        self.label = f"api:{model}"
        self._model = model
        # 自托管 Qwen 节点：经 chat_template_kwargs 关思考（mlx_lm.server/vLLM 支持）
        self._extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
        )
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI  # 惰性导入
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _messages(self, system: str, messages: list[Message]) -> list[dict]:
        return [{"role": "system", "content": system}, *messages]

    def generate(self, system: str, messages: list[Message], *,
                 max_tokens: int = 320, temperature: float = 0.7) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=self._messages(system, messages),
            max_tokens=max_tokens, temperature=temperature, extra_body=self._extra_body,
        )
        return _strip_think(resp.choices[0].message.content or "")

    def stream(self, system: str, messages: list[Message], *,
               max_tokens: int = 320, temperature: float = 0.7) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model, messages=self._messages(system, messages),
            max_tokens=max_tokens, temperature=temperature, stream=True,
            extra_body=self._extra_body,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
