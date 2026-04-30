"""
build_memory.py — 用 base 模型生成预热输出，作为角色记忆库。

输出：eval/results/memory_bank.json
格式：{ "能天使": ["回复1", "回复2", ...], "凯尔希": [...], "阿米娅": [...] }

预热 prompt 与 eval prompt 刻意不同，避免循环：
  eval prompt 测日常/任务/感染者融入/挑战/情绪
  warmup prompt 测关系/过去经历/价值观/信念
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 角色卡（与 judge.py 保持一致）
# ---------------------------------------------------------------------------

CHARACTER_CARDS = {
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
}

# ---------------------------------------------------------------------------
# 预热 prompt（刻意区别于 eval 的 p1-p6）
# ---------------------------------------------------------------------------

WARMUP_PROMPTS = [
    ("能天使", "w1", "能天使，你在罗德岛有没有特别要好的朋友？"),
    ("能天使", "w2", "听说你以前在莱茵生命的日子不太好过，你愿意说说吗？"),
    ("凯尔希", "w3", "凯尔希，你认为感染者的治疗未来有没有可能实现根治？"),
    ("凯尔希", "w4", "凯尔希，资源紧缺的时候，你是怎么决定医疗优先级的？"),
    ("阿米娅", "w5", "阿米娅，有没有一次任务让你印象特别深刻？"),
    ("阿米娅", "w6", "阿米娅，你有没有想过有一天不用再打仗了，你会做什么？"),
]

import re
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def build_memory(model_path: str = "mlx-community/Qwen3-8B-4bit") -> dict:
    from mlx_lm import load, generate
    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.8)
        use_sampler = True
    except Exception:
        use_sampler = False

    logger.info(f"Loading base model: {model_path}")
    model, tokenizer = load(model_path)

    memory_bank: dict[str, list[str]] = {c: [] for c in CHARACTER_CARDS}

    for char, wid, user_text in WARMUP_PROMPTS:
        messages = [
            {"role": "system",  "content": CHARACTER_CARDS[char]},
            {"role": "user",    "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if use_sampler:
            output = generate(model, tokenizer, prompt=prompt,
                              max_tokens=300, verbose=False, sampler=sampler)
        else:
            output = generate(model, tokenizer, prompt=prompt,
                              max_tokens=300, verbose=False, temperature=0.8)

        output = strip_think(output)
        memory_bank[char].append(output)
        logger.info(f"  [{char}][{wid}] {output[:80]}…")

    return memory_bank


if __name__ == "__main__":
    out_path = Path("eval/results/memory_bank.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bank = build_memory()
    out_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Memory bank saved → {out_path}")
    for char, entries in bank.items():
        logger.info(f"  {char}: {len(entries)} entries")
