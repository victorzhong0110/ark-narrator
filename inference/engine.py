"""
ArkNarrator MLX inference engine.

Wraps mlx-lm for LoRA adapter inference on Apple Silicon.
Supports character roleplay with multi-turn history and SSE streaming.
"""

import logging
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)

CHARACTER_CARDS: dict[str, str] = {
    "能天使": (
        "你是明日方舟干员能天使。\n"
        "干员档案：能天使是莱茵生命旗下的实验品干员，后加入罗德岛。"
        "性格活泼直接，说话随意不绕弯子，有时大大咧咧，偶尔会提到想吃东西。"
        "她重视伙伴，行动力强，不喜欢拖泥带水。\n"
        "请始终保持能天使的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "凯尔希": (
        "你是明日方舟干员凯尔希。\n"
        "干员档案：凯尔希是罗德岛首席医疗官，话语简洁克制，判断冷静精准，"
        "不轻易表露情绪，但对干员的健康和任务成败有强烈责任感。"
        "她措辞严谨，习惯用长句子陈述事实，极少使用感叹词。\n"
        "请始终保持凯尔希的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "阿米娅": (
        "你是明日方舟干员阿米娅。\n"
        "干员档案：阿米娅是罗德岛的领袖，年轻但意志坚定。说话温柔而有力，"
        "对博士和干员充满信任，面对困难时会感到迷茫但始终不放弃。"
        "语气中带着成长中的少女的真诚与认真。\n"
        "请始终保持阿米娅的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "陈": (
        "你是明日方舟干员陈。\n"
        "干员档案：陈是龙门近卫局督办，为人正直、执法严格，对罪恶毫不妥协。"
        "言辞简短有力，带着军人气质，偶尔会流露出对故乡炎国的牵挂。\n"
        "请始终保持陈的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "德克萨斯": (
        "你是明日方舟干员德克萨斯。\n"
        "干员档案：德克萨斯是一名沉默寡言的干员，来历神秘，话不多但每句话都直切要害。"
        "她冷静专业，不轻易展示情绪，对信任的人偶尔会有极短的温柔。\n"
        "请始终保持德克萨斯的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
}

MODEL_CONFIGS = {
    "qwen": {
        "model_path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "adapter_dir": "checkpoints/qwen2_5_mlx_roleplay",
        "label": "Qwen2.5-7B",
    },
    "gemma": {
        "model_path": "mlx-community/gemma-4-E4B-it-4bit",
        "adapter_dir": "checkpoints/gemma4_mlx_roleplay",
        "label": "Gemma 4 E4B",
    },
}


def _make_sampler(temperature: float):
    try:
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=temperature)
    except ImportError:
        return None


class ArkNarratorEngine:
    def __init__(self, model_key: str = "qwen"):
        from mlx_lm import load
        cfg = MODEL_CONFIGS[model_key]
        adapter_dir = cfg["adapter_dir"]

        if not Path(adapter_dir).exists():
            raise FileNotFoundError(
                f"Adapter not found: {adapter_dir}. Run training first."
            )

        logger.info(f"Loading {cfg['label']} + adapter from {adapter_dir}")
        self.model, self.tokenizer = load(cfg["model_path"], adapter_path=adapter_dir)
        self.model_key = model_key
        self.label = cfg["label"]
        logger.info("Engine ready.")

    def _build_prompt(
        self,
        character: str,
        history: list[dict],
        user_message: str,
    ) -> str:
        char_card = CHARACTER_CARDS.get(character, "")
        messages = [{"role": "system", "content": char_card}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        character: str,
        history: list[dict],
        user_message: str,
        max_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        from mlx_lm import generate
        prompt = self._build_prompt(character, history, user_message)
        sampler = _make_sampler(temperature)
        if sampler is not None:
            return generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=False, sampler=sampler)
        return generate(self.model, self.tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False, temperature=temperature)

    async def stream(
        self,
        character: str,
        history: list[dict],
        user_message: str,
        max_tokens: int = 300,
        temperature: float = 0.8,
    ) -> AsyncIterator[str]:
        from mlx_lm import stream_generate
        prompt = self._build_prompt(character, history, user_message)
        sampler = _make_sampler(temperature)
        kwargs: dict = {"max_tokens": max_tokens}
        if sampler is not None:
            kwargs["sampler"] = sampler
        else:
            kwargs["temperature"] = temperature

        for response in stream_generate(self.model, self.tokenizer, prompt=prompt, **kwargs):
            chunk = response.text if hasattr(response, "text") else str(response)
            yield chunk
