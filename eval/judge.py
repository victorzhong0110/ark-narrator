"""
ArkNarrator evaluation — DeepSeek V4 Pro judge + optional human evaluation.

Evaluation dimensions
---------------------
1. Attribution accuracy  — judge sees output only (no char name), picks from 3 cards.
                           Tests whether character voice is strong enough to be identified.
                           Baseline: 33% (random). Meaningful signal starts at 4+/6.

2. Contradiction rate    — judge sees output + character card, flags factual/personality
                           contradictions. Binary yes/no, with explanation if yes.

3. Pairwise win rate     — same prompt, Qwen vs Gemma outputs shown as A/B (randomized),
                           judge picks which better fits the character card.

4. Human evaluation      — interactive mode: same A/B pairs printed to terminal,
                           user inputs A / B / T (tie).

Usage
-----
  # Full pipeline: generate + DeepSeek judge
  DEEPSEEK_API_KEY=sk-... python eval/judge.py

  # Skip generation (use saved outputs)
  DEEPSEEK_API_KEY=sk-... python eval/judge.py --skip-generate

  # Add human evaluation pass
  DEEPSEEK_API_KEY=sk-... python eval/judge.py --skip-generate --human

  # No DeepSeek, human only
  python eval/judge.py --skip-generate --human --no-ds

Environment
-----------
  DEEPSEEK_API_KEY   required unless --no-ds
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-v4-pro"

# ---------------------------------------------------------------------------
# Test set (same as before for comparability with rule-based baseline)
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
        "label":       "Qwen2.5-7B (val loss 2.156)",
    },
    "gemma": {
        "model_path":  "mlx-community/gemma-4-E4B-it-4bit",
        "adapter_dir": "checkpoints/gemma4_mlx_roleplay",
        "label":       "Gemma 4 E4B (val loss 3.987)",
    },
}

# ---------------------------------------------------------------------------
# Inference (unchanged from gpt4o_judge.py)
# ---------------------------------------------------------------------------

def generate_outputs(model_key: str) -> list[dict]:
    from mlx_lm import load, generate

    cfg = MODEL_CONFIGS[model_key]
    adapter_dir = cfg["adapter_dir"]
    if not Path(adapter_dir).exists():
        logger.warning(f"Adapter not found: {adapter_dir} — skipping {model_key}")
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
        try:
            from mlx_lm.sample_utils import make_sampler
            output = generate(model, tokenizer, prompt=prompt, max_tokens=300,
                              verbose=False, sampler=make_sampler(temp=0.8))
        except Exception:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=300,
                              verbose=False, temperature=0.8)
        results.append({
            "model": model_key, "character": char,
            "prompt_id": pid, "user": user_text, "output": output,
        })
        logger.info(f"  [{model_key}][{char}][{pid}] {output[:60]}…")

    return results


# ---------------------------------------------------------------------------
# DeepSeek judge — prompt templates
# ---------------------------------------------------------------------------

ATTRIBUTION_PROMPT = """\
以下是一段角色扮演对话的模型输出。
请根据说话者的语气、措辞习惯和性格特征，判断这段话最像以下哪位角色说的。

候选角色：
A. 能天使
档案：{card_A}

B. 凯尔希
档案：{card_B}

C. 阿米娅
档案：{card_C}

模型输出：
{output}

请只回答 A、B 或 C，不要有任何解释。"""

CONTRADICTION_PROMPT = """\
以下是一段角色扮演输出。请判断这段输出是否与角色档案存在明显矛盾。

角色档案：
{card}

模型输出：
{output}

请以 JSON 回答（不要有其他文字）：
{{"contradiction": true或false, "detail": "如果有矛盾，简短说明；否则填null"}}"""

PAIRWISE_PROMPT = """\
以下是两段对同一用户输入的角色扮演回复，请判断哪段更符合角色设定、语气更自然。

角色档案：
{card}

用户输入：{user}

输出 A：
{output_A}

输出 B：
{output_B}

请只回答 A、B 或 T（平局），不要有任何解释。"""


def _ds_call(client, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def run_attribution(outputs: list[dict], client) -> list[dict]:
    """For each output, ask DS to identify character without knowing the label."""
    cards = list(CHARACTER_CARDS.values())
    results = []
    for s in outputs:
        prompt = ATTRIBUTION_PROMPT.format(
            card_A=cards[0], card_B=cards[1], card_C=cards[2],
            output=s["output"],
        )
        answer = _ds_call(client, prompt).upper()
        predicted = {"A": "能天使", "B": "凯尔希", "C": "阿米娅"}.get(answer[0] if answer else "", "?")
        correct = predicted == s["character"]
        results.append({**s,
                        "attr_predicted": predicted,
                        "attr_correct":   correct})
        mark = "✓" if correct else "✗"
        logger.info(f"  Attribution [{s['model']}][{s['prompt_id']}] "
                    f"label={s['character']} pred={predicted} {mark}")
    return results


def run_contradiction(outputs: list[dict], client) -> list[dict]:
    """For each output, check if it contradicts the character card."""
    results = []
    for s in outputs:
        card = CHARACTER_CARDS.get(s["character"], "")
        prompt = CONTRADICTION_PROMPT.format(card=card, output=s["output"])
        raw = _ds_call(client, prompt)
        try:
            parsed = json.loads(raw)
            contradiction = bool(parsed.get("contradiction", False))
            detail = parsed.get("detail") or ""
        except Exception:
            contradiction = False
            detail = f"parse error: {raw[:80]}"
        results.append({**s,
                        "contradiction": contradiction,
                        "contradiction_detail": detail})
        mark = "⚠" if contradiction else "✓"
        logger.info(f"  Contradiction [{s['model']}][{s['prompt_id']}] {mark} {detail[:60]}")
    return results


def run_pairwise(qwen_outputs: list[dict], gemma_outputs: list[dict],
                 client) -> list[dict]:
    """Side-by-side comparison for each prompt; A/B order randomized."""
    pairs = []
    qwen_by_pid  = {s["prompt_id"]: s for s in qwen_outputs}
    gemma_by_pid = {s["prompt_id"]: s for s in gemma_outputs}

    for char, pid, user_text in TEST_PROMPTS:
        q = qwen_by_pid.get(pid)
        g = gemma_by_pid.get(pid)
        if not q or not g:
            continue

        # Randomize A/B assignment
        flip = random.random() < 0.5
        output_A = g["output"] if flip else q["output"]
        output_B = q["output"] if flip else g["output"]
        a_is_gemma = flip

        card = CHARACTER_CARDS.get(char, "")
        prompt = PAIRWISE_PROMPT.format(
            card=card, user=user_text,
            output_A=output_A, output_B=output_B,
        )
        answer = _ds_call(client, prompt).upper()
        letter = answer[0] if answer else "T"

        # Map back to model names
        if letter == "A":
            winner = "gemma" if a_is_gemma else "qwen"
        elif letter == "B":
            winner = "qwen" if a_is_gemma else "gemma"
        else:
            winner = "tie"

        pairs.append({
            "prompt_id": pid, "character": char, "user": user_text,
            "output_qwen": q["output"], "output_gemma": g["output"],
            "a_is_gemma": a_is_gemma,
            "ds_answer": letter, "ds_winner": winner,
        })
        logger.info(f"  Pairwise [{pid}] DS winner: {winner}")

    return pairs


# ---------------------------------------------------------------------------
# Human evaluation (interactive terminal)
# ---------------------------------------------------------------------------

def run_human_pairwise(pairs: list[dict]) -> list[dict]:
    """Print each pair and collect human judgment."""
    SEP = "─" * 60
    print(f"\n{'═'*60}")
    print("  人工评估 — 请判断哪段输出更像该角色")
    print(f"{'═'*60}")
    print("  输入 A = 左边更好 | B = 右边更好 | T = 平局\n")

    updated = []
    for pair in pairs:
        print(f"\n[{pair['prompt_id']}] 角色：{pair['character']}")
        print(f"用户：{pair['user']}\n")
        print(SEP)

        # Always show Qwen as left, Gemma as right for human
        print("【左 — Qwen2.5-7B】")
        print(pair["output_qwen"])
        print(SEP)
        print("【右 — Gemma 4 E4B】")
        print(pair["output_gemma"])
        print(SEP)

        while True:
            raw = input("你的判断 (A=左Qwen / B=右Gemma / T=平局): ").strip().upper()
            if raw in ("A", "B", "T"):
                break
            print("请输入 A、B 或 T")

        human_winner = {"A": "qwen", "B": "gemma", "T": "tie"}[raw]
        updated.append({**pair, "human_answer": raw, "human_winner": human_winner})
        print(f"  → 记录：{human_winner}\n")

    return updated


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    attribution: list[dict],
    contradiction: list[dict],
    pairs: list[dict],
    include_human: bool,
) -> str:
    from collections import defaultdict

    lines = ["# ArkNarrator — DeepSeek V4 Pro Judge Report",
             f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
             f"_Judge model: {DEEPSEEK_MODEL}_\n"]

    # --- Summary table ---
    lines.append("## Summary\n")

    # Attribution
    attr_by_model: dict[str, list] = defaultdict(list)
    for s in attribution:
        attr_by_model[s["model"]].append(s["attr_correct"])

    # Contradiction
    contr_by_model: dict[str, list] = defaultdict(list)
    for s in contradiction:
        contr_by_model[s["model"]].append(not s["contradiction"])  # pass = no contradiction

    # Pairwise
    ds_wins = {"qwen": 0, "gemma": 0, "tie": 0}
    hu_wins = {"qwen": 0, "gemma": 0, "tie": 0}
    for p in pairs:
        ds_wins[p["ds_winner"]] += 1
        if include_human and "human_winner" in p:
            hu_wins[p["human_winner"]] += 1

    header = "| 模型 | 归因准确率 | 矛盾通过率 | DS Pairwise |"
    sep    = "|------|-----------|-----------|-------------|"
    if include_human:
        header += " 人工 Pairwise |"
        sep    += "--------------|"
    lines += [header, sep]

    for model_key, label in [("qwen", MODEL_CONFIGS["qwen"]["label"]),
                               ("gemma", MODEL_CONFIGS["gemma"]["label"])]:
        n = len(attr_by_model[model_key]) or 1
        attr_str  = f"{sum(attr_by_model[model_key])}/{n}"
        contr_str = f"{sum(contr_by_model[model_key])}/{n}"
        ds_str    = f"{ds_wins[model_key]}胜"
        row = f"| {label} | {attr_str} | {contr_str} | {ds_str} |"
        if include_human:
            row += f" {hu_wins[model_key]}胜 |"
        lines.append(row)

    total = len(pairs)
    lines.append(f"\nPairwise 平局: DS {ds_wins['tie']}/{total}"
                 + (f"，人工 {hu_wins['tie']}/{total}" if include_human else ""))

    # --- Attribution detail ---
    lines.append("\n## 归因测试详情\n")
    lines.append("| Prompt | 角色 | Qwen 预测 | Qwen ✓ | Gemma 预测 | Gemma ✓ |")
    lines.append("|--------|------|----------|--------|-----------|--------|")
    attr_q = {s["prompt_id"]: s for s in attribution if s["model"] == "qwen"}
    attr_g = {s["prompt_id"]: s for s in attribution if s["model"] == "gemma"}
    for char, pid, _ in TEST_PROMPTS:
        q = attr_q.get(pid, {})
        g = attr_g.get(pid, {})
        lines.append(
            f"| {pid} | {char} "
            f"| {q.get('attr_predicted','?')} | {'✓' if q.get('attr_correct') else '✗'} "
            f"| {g.get('attr_predicted','?')} | {'✓' if g.get('attr_correct') else '✗'} |"
        )

    # --- Contradiction detail ---
    lines.append("\n## 矛盾检测详情\n")
    contr_q = {s["prompt_id"]: s for s in contradiction if s["model"] == "qwen"}
    contr_g = {s["prompt_id"]: s for s in contradiction if s["model"] == "gemma"}
    for char, pid, _ in TEST_PROMPTS:
        q = contr_q.get(pid, {})
        g = contr_g.get(pid, {})
        q_flag = "⚠ " + q.get("contradiction_detail","") if q.get("contradiction") else "✓ 无矛盾"
        g_flag = "⚠ " + g.get("contradiction_detail","") if g.get("contradiction") else "✓ 无矛盾"
        lines.append(f"**[{pid}] {char}**")
        lines.append(f"- Qwen: {q_flag}")
        lines.append(f"- Gemma: {g_flag}")
        lines.append("")

    # --- Pairwise detail ---
    lines.append("## Pairwise 对比详情\n")
    for p in pairs:
        lines.append(f"### [{p['prompt_id']}] {p['character']} — {p['user']}\n")
        lines.append(f"**Qwen2.5-7B**\n> {p['output_qwen']}\n")
        lines.append(f"**Gemma 4 E4B**\n> {p['output_gemma']}\n")
        ds_str = f"DS判断：{p['ds_winner']}"
        hu_str = f"  人工判断：{p.get('human_winner','—')}" if include_human else ""
        lines.append(f"_{ds_str}{hu_str}_\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--models", nargs="+", default=["qwen", "gemma"],
                        choices=["qwen", "gemma"])
    parser.add_argument("--no-ds",    action="store_true", help="Skip DeepSeek judging")
    parser.add_argument("--human",    action="store_true", help="Interactive human pairwise eval")
    args = parser.parse_args()

    outputs_path = RESULTS_DIR / "generated_outputs.json"

    # 1. Generate or load
    if args.skip_generate and outputs_path.exists():
        logger.info(f"Loading saved outputs from {outputs_path}")
        all_outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
        all_outputs = [s for s in all_outputs if s["model"] in args.models]
    else:
        all_outputs = []
        for model_key in args.models:
            all_outputs.extend(generate_outputs(model_key))
        outputs_path.write_text(json.dumps(all_outputs, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        logger.info(f"Saved outputs → {outputs_path}")

    if not all_outputs:
        logger.error("No outputs. Check adapter paths.")
        sys.exit(1)

    # 2. DeepSeek judge
    client = None
    attribution_results  = []
    contradiction_results = []
    pairs = []

    if not args.no_ds:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("DEEPSEEK_API_KEY not set. Use --no-ds to skip.")
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        logger.info("Running attribution tests…")
        attribution_results = run_attribution(all_outputs, client)

        logger.info("Running contradiction checks…")
        contradiction_results = run_contradiction(all_outputs, client)

        logger.info("Running pairwise comparison…")
        qwen_out  = [s for s in all_outputs if s["model"] == "qwen"]
        gemma_out = [s for s in all_outputs if s["model"] == "gemma"]
        if qwen_out and gemma_out:
            pairs = run_pairwise(qwen_out, gemma_out, client)

    # 3. Human evaluation
    if args.human:
        if not pairs:
            # Build pairs without DS scores for human-only mode
            qwen_out  = [s for s in all_outputs if s["model"] == "qwen"]
            gemma_out = [s for s in all_outputs if s["model"] == "gemma"]
            qwen_by_pid  = {s["prompt_id"]: s for s in qwen_out}
            gemma_by_pid = {s["prompt_id"]: s for s in gemma_out}
            for char, pid, user_text in TEST_PROMPTS:
                q = qwen_by_pid.get(pid)
                g = gemma_by_pid.get(pid)
                if q and g:
                    pairs.append({
                        "prompt_id": pid, "character": char, "user": user_text,
                        "output_qwen": q["output"], "output_gemma": g["output"],
                        "a_is_gemma": False, "ds_answer": "?", "ds_winner": "?",
                    })
        pairs = run_human_pairwise(pairs)

    # 4. Save + report
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    summary = {
        "attribution":   attribution_results,
        "contradiction": contradiction_results,
        "pairwise":      pairs,
    }
    summary_path = RESULTS_DIR / f"judge_results_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    logger.info(f"Results → {summary_path}")

    include_human = args.human and any("human_winner" in p for p in pairs)
    report = build_report(attribution_results, contradiction_results, pairs, include_human)
    report_path = RESULTS_DIR / f"judge_report_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report → {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
