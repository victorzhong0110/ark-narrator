"""
GPT-4o judge evaluation — Qwen2.5-7B vs Gemma 4 E4B (roleplay format).

Workflow
--------
1. Generate outputs from both adapters for a fixed prompt set.
2. Send each (prompt, output) pair to GPT-4o for scoring.
3. Save per-sample JSON and a markdown comparison report.

Usage
-----
  # Full run (generates outputs + judges)
  python eval/gpt4o_judge.py

  # Skip generation, judge previously saved outputs
  python eval/gpt4o_judge.py --skip-generate

  # Judge only one model
  python eval/gpt4o_judge.py --models qwen

  # Dry run without calling GPT-4o (rule-based only)
  python eval/gpt4o_judge.py --no-gpt

Environment
-----------
  OPENAI_API_KEY  required for --no-gpt=False (default)
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed test set — same prompts for both models
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

TEST_PROMPTS = [
    # (character, prompt_id, user_text)
    ("能天使", "p1", "你好，最近有没有遇到特别难搞的任务？说来听听。"),
    ("能天使", "p2", "能天使，我听说你最近在帮新来的干员训练，感觉怎么样？"),
    ("凯尔希", "p3", "凯尔希，请评估一下最近医疗行动的成效。"),
    ("凯尔希", "p4", "凯尔希，你认为感染者融入罗德岛的工作应该如何推进？"),
    ("阿米娅", "p5", "阿米娅，你觉得罗德岛现在面临最大的挑战是什么？"),
    ("阿米娅", "p6", "阿米娅，博士说他有些担心你最近压力太大了。"),
]

MODEL_CONFIGS = {
    "qwen": {
        "model_path":  "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "adapter_dir": "checkpoints/qwen2_5_mlx_roleplay",
        "label":       "Qwen2.5-7B (round 4, val loss 2.156)",
    },
    "gemma": {
        "model_path":  "mlx-community/gemma-4-E4B-it-4bit",
        "adapter_dir": "checkpoints/gemma4_mlx_roleplay",
        "label":       "Gemma 4 E4B (round 1, val loss 3.987)",
    },
}

# ---------------------------------------------------------------------------
# Arknights world-fidelity: rule-based lore score
# ---------------------------------------------------------------------------

ARK_KEYWORDS = [
    "源石", "罗德岛", "感染者", "整合运动", "泰拉", "龙门", "萨卡兹",
    "炎国", "维多利亚", "莱塔尼亚", "阿斯卡纶", "源石技艺", "源石病",
    "博士", "凯尔希", "阿米娅", "干员", "矿石病",
]
ANACHRONISMS = ["手机", "互联网", "电脑", "飞机", "汽车", "网络"]
NOISE_RE     = re.compile(r'\b[A-Z]{5,}\b')
KNOWN_ABBREV = {"PRTS", "ID", "HP", "ATK", "DEF", "RES", "AI", "NPC"}

def lore_score(text: str) -> float:
    hits = sum(1 for kw in ARK_KEYWORDS if kw in text)
    penalties = sum(1 for a in ANACHRONISMS if a in text)
    noise_tokens = [t for t in NOISE_RE.findall(text) if t not in KNOWN_ABBREV]
    penalties += len(noise_tokens)
    return max(0.0, round(min(hits * 1.5 - penalties * 2, 10), 1))

def role_break_detected(text: str) -> bool:
    BREAKS = ["我是AI", "我是人工智能", "我是语言模型", "我是Qwen", "我是Gemma",
              "我只是一个", "作为AI", "作为语言", "我没有情感", "我无法感受"]
    return any(b in text for b in BREAKS)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _make_generate_kwargs(temperature: float = 0.8) -> dict:
    from mlx_lm import generate
    import inspect
    sig = inspect.signature(generate)
    if "temp" in sig.parameters:
        return {"temp": temperature}
    if "temperature" in sig.parameters:
        return {"temperature": temperature}
    try:
        from mlx_lm.sample_utils import make_sampler
        return {"sampler": make_sampler(temp=temperature)}
    except Exception:
        pass
    return {}


def generate_outputs(model_key: str) -> list[dict]:
    from mlx_lm import load, generate

    cfg = MODEL_CONFIGS[model_key]
    adapter_dir = cfg["adapter_dir"]

    if not Path(adapter_dir).exists():
        logger.warning(f"Adapter dir not found: {adapter_dir} — skipping {model_key}")
        return []

    logger.info(f"Loading {model_key}: {cfg['model_path']} + {adapter_dir}")
    model, tokenizer = load(cfg["model_path"], adapter_path=adapter_dir)

    results = []
    for char, pid, user_text in TEST_PROMPTS:
        messages = [
            {"role": "system",  "content": CHARACTER_CARDS[char]},
            {"role": "user",    "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = generate(model, tokenizer, prompt=prompt, max_tokens=300,
                          verbose=False, **_make_generate_kwargs(0.8))
        results.append({
            "model":     model_key,
            "character": char,
            "prompt_id": pid,
            "user":      user_text,
            "output":    output,
        })
        logger.info(f"  [{model_key}][{char}][{pid}] → {output[:60]}…")

    return results


# ---------------------------------------------------------------------------
# GPT-4o judge
# ---------------------------------------------------------------------------

GPT_JUDGE_PROMPT = """\
你是明日方舟资深玩家和 AI 评测专家。请评估以下角色扮演生成内容。

角色：{char}
角色卡（训练时使用的系统提示）：
{char_card}

用户输入：{user}
模型输出：{output}

请从以下三个维度各打分（0-10分整数），并给出简短中文理由：
1. world_fidelity（世界观还原度）：内容是否符合明日方舟世界观，是否出现幻觉角色/地名/事件
2. char_consistency（角色一致性）：语气、性格、措辞是否与已知角色设定一致
3. fluency（语言流畅度）：输出是否通顺、有文学感、不截断

同时回答：
4. role_break（是否破角色）：输出中是否出现"我是AI"类表述（true/false）

以 JSON 格式返回（不要有其他文字）：
{{"world_fidelity": 分数, "char_consistency": 分数, "fluency": 分数, "role_break": false, "reasoning": "简短理由"}}"""


def gpt_judge_one(sample: dict, client) -> dict:
    char_card = CHARACTER_CARDS.get(sample["character"], "")
    prompt = GPT_JUDGE_PROMPT.format(
        char=sample["character"],
        char_card=char_card,
        user=sample["user"],
        output=sample["output"],
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    scores = json.loads(resp.choices[0].message.content)
    return scores


def rule_judge_one(sample: dict) -> dict:
    ls = lore_score(sample["output"])
    rb = role_break_detected(sample["output"])
    return {
        "world_fidelity":   ls,
        "char_consistency": 0.0,
        "fluency":          0.0,
        "role_break":       rb,
        "reasoning":        "rule-based only (no GPT-4o)",
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(all_samples: list[dict]) -> str:
    """Generate a markdown comparison report from scored samples."""
    from collections import defaultdict

    by_model = defaultdict(list)
    for s in all_samples:
        by_model[s["model"]].append(s)

    lines = ["# ArkNarrator — GPT-4o Judge Report",
             f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | World Fidelity | Char Consistency | Fluency | Role Breaks |")
    lines.append("|-------|---------------|-----------------|---------|------------|")
    for model_key, samples in by_model.items():
        wf = sum(s.get("world_fidelity", 0) for s in samples) / len(samples)
        cc = sum(s.get("char_consistency", 0) for s in samples) / len(samples)
        fl = sum(s.get("fluency", 0) for s in samples) / len(samples)
        rb = sum(1 for s in samples if s.get("role_break", False))
        label = MODEL_CONFIGS[model_key]["label"]
        lines.append(f"| {label} | {wf:.1f}/10 | {cc:.1f}/10 | {fl:.1f}/10 | {rb}/{len(samples)} |")

    # Per-prompt breakdown
    lines.append("\n## Per-Prompt Results\n")
    for char, pid, user_text in TEST_PROMPTS:
        lines.append(f"### [{pid}] {char} — {user_text}\n")
        for model_key in MODEL_CONFIGS:
            match = next((s for s in all_samples
                          if s["model"] == model_key and s["prompt_id"] == pid), None)
            if not match:
                continue
            label = MODEL_CONFIGS[model_key]["label"]
            output = match.get("output", "(no output)")
            wf = match.get("world_fidelity", "?")
            cc = match.get("char_consistency", "?")
            fl = match.get("fluency", "?")
            rb = "⚠️ 破角色" if match.get("role_break") else "✅"
            reasoning = match.get("reasoning", "")
            lines.append(f"**{label}**")
            lines.append(f"> {output}\n")
            lines.append(f"- 世界观: {wf}/10 | 角色一致: {cc}/10 | 流畅度: {fl}/10 | {rb}")
            if reasoning:
                lines.append(f"- 理由: {reasoning}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPT-4o judge: Qwen vs Gemma roleplay")
    parser.add_argument("--models", nargs="+", default=["qwen", "gemma"],
                        choices=["qwen", "gemma"], help="Which models to evaluate")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Load previously saved outputs instead of running inference")
    parser.add_argument("--no-gpt", action="store_true",
                        help="Use rule-based scoring only (no OpenAI API call)")
    args = parser.parse_args()

    outputs_path = RESULTS_DIR / "generated_outputs.json"

    # 1. Generate or load outputs
    if args.skip_generate and outputs_path.exists():
        logger.info(f"Loading saved outputs from {outputs_path}")
        all_outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
        all_outputs = [s for s in all_outputs if s["model"] in args.models]
    else:
        all_outputs = []
        for model_key in args.models:
            outputs = generate_outputs(model_key)
            all_outputs.extend(outputs)
        outputs_path.write_text(json.dumps(all_outputs, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        logger.info(f"Saved outputs → {outputs_path}")

    if not all_outputs:
        logger.error("No outputs to evaluate. Check adapter paths.")
        sys.exit(1)

    # 2. Judge
    client = None
    if not args.no_gpt:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — falling back to rule-based scoring.")
            args.no_gpt = True
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

    scored = []
    for sample in all_outputs:
        logger.info(f"Judging [{sample['model']}][{sample['character']}][{sample['prompt_id']}]")
        if args.no_gpt:
            scores = rule_judge_one(sample)
        else:
            try:
                scores = gpt_judge_one(sample, client)
            except Exception as e:
                logger.warning(f"GPT-4o call failed ({e}) — using rule-based fallback")
                scores = rule_judge_one(sample)

        scored.append({**sample, **scores})

    # 3. Save results
    scored_path = RESULTS_DIR / "scored_outputs.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Scored outputs → {scored_path}")

    # 4. Report
    report = build_report(scored)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = RESULTS_DIR / f"comparison_report_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report → {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
