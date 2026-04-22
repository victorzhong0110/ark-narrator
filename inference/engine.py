"""
ArkNarrator inference engine.
Loads fine-tuned LoRA adapter on top of Qwen2.5-7B and generates content.
"""

import torch
import logging
from pathlib import Path
from typing import Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是明日方舟（Arknights）世界中的内容创作助手，"
    "精通泰拉大陆的历史、干员档案与源石技艺体系。"
    "请根据设定生成符合世界观的内容。"
)


class ArkNarratorEngine:
    def __init__(self, base_model: str, adapter_path: str, device: str = "auto"):
        logger.info(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        if Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(base, adapter_path)
            self.model = self.model.merge_and_unload()  # merge for faster inference
        else:
            logger.warning("No adapter found — using base model")
            self.model = base

        self.model.eval()
        logger.info("Engine ready ✓")

    def generate(
        self,
        instruction: str,
        context: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Single-shot generation."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_content = f"{instruction}\n\n{context}" if context else instruction
        messages.append({"role": "user", "content": user_content})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def stream(
        self,
        instruction: str,
        context: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.8,
    ) -> Iterator[str]:
        """Streaming generation — yields tokens one by one."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_content = f"{instruction}\n\n{context}" if context else instruction
        messages.append({"role": "user", "content": user_content})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token
