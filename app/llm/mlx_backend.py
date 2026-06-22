"""本地 MLX 后端（Apple Silicon）。

走 base 模型 + 角色卡 + RAG 路线（这是项目自测得出的结论：8B 上 harness > 微调）。
适配器是可选的——若配置了 ARK_ADAPTER_DIR 且存在则叠加，否则纯 base。
mlx_lm 仅在此处惰性导入，没装/没权重不影响其它模块与测试。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from app.llm.base import Message

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """去掉思考链。兼容三种情况：闭合块、仅残留 </think>、未闭合的尾部 <think>。"""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:                 # 闭合块残留 → 取最后一个之后
        text = text.rsplit("</think>", 1)[-1]
    idx = text.find("<think>")             # 未闭合（被 max_tokens 截断）→ 丢掉
    if idx != -1:
        text = text[:idx]
    return text.strip()


class MLXBackend:
    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-8B-4bit",
        adapter_dir: str | None = None,
    ):
        from mlx_lm import load  # 惰性导入

        self.model_path = model_path
        adapter = adapter_dir if (adapter_dir and Path(adapter_dir).exists()) else None
        if adapter_dir and adapter is None:
            logger.warning("适配器目录不存在，改用 base：%s", adapter_dir)

        logger.info("加载模型 %s%s", model_path, f" + 适配器 {adapter}" if adapter else "（base）")
        if adapter:
            self.model, self.tokenizer = load(model_path, adapter_path=adapter)
            self.label = f"{model_path} + LoRA"
        else:
            self.model, self.tokenizer = load(model_path)
            self.label = model_path

    def _prompt(self, system: str, messages: list[Message]) -> str:
        chat: list[dict] = [{"role": "system", "content": system}]
        chat.extend(messages)
        # 角色扮演不需要思考链：关掉它（更快、无 <think> 泄漏、护栏不会扫到推理文本）。
        try:
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

    @staticmethod
    def _sampler(temperature: float):
        try:
            from mlx_lm.sample_utils import make_sampler
            return make_sampler(temp=temperature)
        except Exception:  # noqa: BLE001
            return None

    def generate(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 320,
        temperature: float = 0.7,
    ) -> str:
        from mlx_lm import generate

        prompt = self._prompt(system, messages)
        sampler = self._sampler(temperature)
        if sampler is not None:
            out = generate(self.model, self.tokenizer, prompt=prompt,
                           max_tokens=max_tokens, verbose=False, sampler=sampler)
        else:
            out = generate(self.model, self.tokenizer, prompt=prompt,
                           max_tokens=max_tokens, verbose=False, temperature=temperature)
        return _strip_think(out)

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 320,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        from mlx_lm import stream_generate

        prompt = self._prompt(system, messages)
        sampler = self._sampler(temperature)
        kwargs: dict = {"max_tokens": max_tokens}
        if sampler is not None:
            kwargs["sampler"] = sampler
        else:
            kwargs["temperature"] = temperature
        for resp in stream_generate(self.model, self.tokenizer, prompt=prompt, **kwargs):
            yield resp.text if hasattr(resp, "text") else str(resp)
