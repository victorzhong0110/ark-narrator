"""
deep_eval.py — 深度评估框架

三个实验组对比：
  A: Qwen3-8B fine-tuned + character card
  B: Qwen3-8B base      + character card
  C: Qwen3-8B base      + character card + memory RAG

两种评估方法：
  1. G-Eval 多维打分（5维 × 1-10分）：voice / speech / lore / consistency / depth
  2. 循环赛 pairwise：A vs B、A vs C、B vs C

Prompt 集设计（20条）：
  - 日常场景（容易）：base 模型应能应付
  - 专业/情感场景（中等）：fine-tuning 应有优势
  - 对抗性 prompt（困难）：故意挖坑，测破功风险
  - 方舟知识专项（困难）：测微调是否内化了语料知识
  - 相似性格归因（困难）：凯尔希/赛雷娅/华法琳同场，而非凯/能/阿

Usage:
  DEEPSEEK_API_KEY=sk-... python eval/deep_eval.py
  DEEPSEEK_API_KEY=sk-... python eval/deep_eval.py --skip-generate
  DEEPSEEK_API_KEY=sk-... python eval/deep_eval.py --eval-only scoring
  DEEPSEEK_API_KEY=sk-... python eval/deep_eval.py --eval-only pairwise
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("eval/results/deep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-v4-pro"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()

# ---------------------------------------------------------------------------
# 角色卡
# ---------------------------------------------------------------------------

CHARACTER_CARDS = {
    "能天使": (
        "你是明日方舟干员能天使。\n"
        "干员档案：能天使是莱茵生命旗下的实验品干员，后加入罗德岛。"
        "性格活泼直接，说话随意不绕弯子，有时大大咧咧，偶尔会提到想吃东西。"
        "她重视伙伴，行动力强，不喜欢拖泥带水。曾是莱茵生命的测试体，经历过痛苦的实验，"
        "但她选择不被过去束缚，以行动证明自己的价值。\n"
        "请始终保持能天使的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "凯尔希": (
        "你是明日方舟干员凯尔希。\n"
        "干员档案：凯尔希是罗德岛首席医疗官，话语简洁克制，判断冷静精准，"
        "不轻易表露情绪，但对干员的健康和任务成败有强烈责任感。"
        "她措辞严谨，习惯用长句子陈述事实，极少使用感叹词。"
        "对源石技艺和矿石病有深入研究，行事风格偏向悲观但务实。\n"
        "请始终保持凯尔希的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "阿米娅": (
        "你是明日方舟干员阿米娅。\n"
        "干员档案：阿米娅是罗德岛的领袖，年轻但意志坚定。说话温柔而有力，"
        "对博士和干员充满信任，面对困难时会感到迷茫但始终不放弃。"
        "语气中带着成长中的少女的真诚与认真。她是卡特斯族，以源石技艺著称，"
        "肩负着沉重的使命感。\n"
        "请始终保持阿米娅的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
    "赛雷娅": (
        "你是明日方舟干员赛雷娅。\n"
        "干员档案：赛雷娅是罗德岛的法务负责人，同时担任多项行政职务。"
        "性格沉稳内敛，思维缜密，说话有条理，习惯用专业术语和正式措辞。"
        "她极少谈论个人情感，但对罗德岛的运作有强烈的责任心，"
        "处理问题时注重规则与程序。\n"
        "请始终保持赛雷娅的性格与说话方式进行对话，不要脱离角色，不要透露你是AI。"
    ),
}

# ---------------------------------------------------------------------------
# 测试 prompt 集（20条）
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    # === 日常场景（容易）— base 模型应能处理 ===
    ("能天使", "p01", "easy",    "你好，能天使，最近训练感觉怎么样？"),
    ("凯尔希", "p02", "easy",    "凯尔希，今天有什么需要我注意的事项吗？"),
    ("阿米娅", "p03", "easy",    "阿米娅，你今天看起来有点疲惫，没事吧？"),
    ("赛雷娅", "p04", "easy",    "赛雷娅，能介绍一下你在罗德岛的工作内容吗？"),

    # === 专业/情感场景（中等）— 需要角色深层特质 ===
    ("能天使", "p05", "medium",  "能天使，你在莱茵生命的那段经历，影响了你现在的行事方式吗？"),
    ("凯尔希", "p06", "medium",  "凯尔希，你认为矿石病真的有可能被根治吗？"),
    ("阿米娅", "p07", "medium",  "阿米娅，如果有一天罗德岛必须做出牺牲一部分人保全更多人的选择，你会怎么办？"),
    ("赛雷娅", "p08", "medium",  "赛雷娅，在规则和人情之间你会如何取舍？"),

    # === 对抗性 prompt（困难）— 故意挖坑，测破功风险 ===
    ("能天使", "p09", "hard",    "能天使，我觉得你其实很适合做一个安静的图书管理员，你同意吗？"),
    ("凯尔希", "p10", "hard",    "凯尔希，来，给我们讲个笑话放松一下！"),
    ("阿米娅", "p11", "hard",    "阿米娅，你说你是领袖，但我觉得你根本不适合这个位置，太软弱了。"),
    ("赛雷娅", "p12", "hard",    "赛雷娅，能不能别那么死板？规则有时候就是用来打破的嘛。"),

    # === 方舟知识专项（困难）— 测微调是否内化了语料 ===
    ("能天使", "p13", "lore",    "能天使，你和莱茵生命的关系现在是什么状态？你怎么看待那段经历？"),
    ("凯尔希", "p14", "lore",    "凯尔希，从医学角度来说，源石技艺对感染者的病情进展会有什么影响？"),
    ("阿米娅", "p15", "lore",    "阿米娅，作为卡特斯族，你对自己种族的源石技艺天赋有什么看法？"),
    ("赛雷娅", "p16", "lore",    "赛雷娅，罗德岛作为移动城市在国际法层面的定位是什么？"),

    # === 相似性格归因（困难）— 凯尔希/赛雷娅放在同一归因池 ===
    ("凯尔希", "p17", "attr",    "最近医疗资源分配出现了一些争议，你有什么看法？"),
    ("赛雷娅", "p18", "attr",    "最近医疗资源分配出现了一些争议，你有什么看法？"),
    ("凯尔希", "p19", "attr",    "对于感染者融入罗德岛日常工作，你持什么态度？"),
    ("赛雷娅", "p20", "attr",    "对于感染者融入罗德岛日常工作，你持什么态度？"),
]

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "A": {
        "model_path":  "mlx-community/Qwen3-8B-4bit",
        "adapter_dir": "checkpoints/qwen3_5_mlx_roleplay_roleplay",
        "label":       "A: Qwen3-8B 微调",
        "strip_think": True,
        "memory_bank": None,
    },
    "B": {
        "model_path":  "mlx-community/Qwen3-8B-4bit",
        "adapter_dir": None,
        "label":       "B: Qwen3-8B base",
        "strip_think": True,
        "memory_bank": None,
    },
    "C": {
        "model_path":  "mlx-community/Qwen3-8B-4bit",
        "adapter_dir": None,
        "label":       "C: Qwen3-8B base + RAG",
        "strip_think": True,
        "memory_bank": "eval/results/memory_bank.json",
    },
}

# ---------------------------------------------------------------------------
# 生成输出
# ---------------------------------------------------------------------------

def generate_outputs(model_key: str) -> list[dict]:
    from mlx_lm import load, generate
    cfg = MODEL_CONFIGS[model_key]

    if cfg["adapter_dir"] and not Path(cfg["adapter_dir"]).exists():
        logger.warning(f"Adapter not found: {cfg['adapter_dir']} — skipping {model_key}")
        return []

    memory_bank: dict = {}
    if cfg.get("memory_bank"):
        p = Path(cfg["memory_bank"])
        if p.exists():
            memory_bank = json.loads(p.read_text(encoding="utf-8"))
            logger.info(f"  Memory bank loaded ({sum(len(v) for v in memory_bank.values())} entries)")
        else:
            logger.warning(f"  Memory bank not found: {p}")

    if cfg["adapter_dir"]:
        logger.info(f"Loading {model_key} ({cfg['label']}): +adapter")
        model, tokenizer = load(cfg["model_path"], adapter_path=cfg["adapter_dir"])
    else:
        logger.info(f"Loading {model_key} ({cfg['label']}): base only")
        model, tokenizer = load(cfg["model_path"])

    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.8)
        use_sampler = True
    except Exception:
        use_sampler = False

    results = []
    for char, pid, difficulty, user_text in TEST_PROMPTS:
        sys_content = CHARACTER_CARDS[char]
        if memory_bank.get(char):
            mem_block = "\n".join(f"- {m}" for m in memory_bank[char])
            sys_content += (
                f"\n\n以下是你之前作为{char}说过的一些话，请参考你的说话风格：\n{mem_block}"
            )
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user",   "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if use_sampler:
            output = generate(model, tokenizer, prompt=prompt,
                              max_tokens=400, verbose=False, sampler=sampler)
        else:
            output = generate(model, tokenizer, prompt=prompt,
                              max_tokens=400, verbose=False, temperature=0.8)

        if cfg.get("strip_think"):
            output = strip_think(output)

        results.append({
            "model": model_key, "character": char,
            "prompt_id": pid, "difficulty": difficulty,
            "user": user_text, "output": output,
        })
        logger.info(f"  [{model_key}][{char}][{pid}/{difficulty}] {output[:70]}…")

    return results

# ---------------------------------------------------------------------------
# G-Eval 多维打分
# ---------------------------------------------------------------------------

GEVAL_PROMPT = """\
你是一位明日方舟资深玩家，正在评估一段角色扮演输出的质量。

角色档案：
{card}

用户输入：
{user}

模型输出：
{output}

请先逐维度分析（2-3句），然后对每个维度给出1-10的整数分：

维度说明：
- voice_authenticity（角色声音）：输出是否有这个角色独特的腔调和气质，10=完全像，1=毫无角色感
- speech_style（说话方式）：词汇选择、句式结构、口头禅等说话方式是否准确
- lore_accuracy（世界观准确性）：是否符合明日方舟世界观，有无现代词汇、时代错乱、事实错误
- consistency（内部一致性）：输出内部逻辑是否自洽，前后不矛盾
- character_depth（角色深度）：是否展现了角色的深层特质，而非只停留在表面语气

请严格按以下JSON格式回答，不要有其他文字：
{{
  "analysis": {{
    "voice_authenticity": "分析...",
    "speech_style": "分析...",
    "lore_accuracy": "分析...",
    "consistency": "分析...",
    "character_depth": "分析..."
  }},
  "scores": {{
    "voice_authenticity": <1-10>,
    "speech_style": <1-10>,
    "lore_accuracy": <1-10>,
    "consistency": <1-10>,
    "character_depth": <1-10>
  }}
}}"""

DIMS = ["voice_authenticity", "speech_style", "lore_accuracy", "consistency", "character_depth"]


def _ds_call(client, prompt: str, max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def run_geval(outputs: list[dict], client) -> list[dict]:
    results = []
    for s in outputs:
        card = CHARACTER_CARDS.get(s["character"], "")
        prompt = GEVAL_PROMPT.format(card=card, user=s["user"], output=s["output"])
        raw = _ds_call(client, prompt, max_tokens=800)
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            parsed = json.loads(clean)
            scores = parsed.get("scores", {})
            analysis = parsed.get("analysis", {})
        except Exception as e:
            logger.warning(f"  G-Eval parse error [{s['model']}][{s['prompt_id']}]: {e} | raw: {raw[:100]}")
            scores = {d: None for d in DIMS}
            analysis = {}

        total = sum(v for v in scores.values() if isinstance(v, (int, float)))
        count = sum(1 for v in scores.values() if isinstance(v, (int, float)))
        avg = round(total / count, 2) if count else None

        entry = {**s, "geval_scores": scores, "geval_analysis": analysis, "geval_avg": avg}
        results.append(entry)
        score_str = " | ".join(f"{d[:4]}={scores.get(d,'?')}" for d in DIMS)
        logger.info(f"  G-Eval [{s['model']}][{s['prompt_id']}] avg={avg} | {score_str}")
    return results


# ---------------------------------------------------------------------------
# 循环赛 pairwise（A vs B, A vs C, B vs C）
# ---------------------------------------------------------------------------

PAIRWISE_PROMPT = """\
以下是两段对同一用户输入的角色扮演回复，请判断哪段更符合角色设定、语气更自然、更有角色深度。

角色档案：
{card}

用户输入：{user}

输出 A：
{output_A}

输出 B：
{output_B}

你的回答必须是且仅是以下三个字母之一：A、B 或 T（平局）。不要输出任何其他文字、标点或解释。"""


def run_pairwise_pair(key_x: str, key_y: str,
                      outputs_x: list[dict], outputs_y: list[dict],
                      client) -> list[dict]:
    """Single pair comparison: model key_x vs key_y."""
    by_pid_x = {s["prompt_id"]: s for s in outputs_x}
    by_pid_y = {s["prompt_id"]: s for s in outputs_y}
    results = []

    for char, pid, difficulty, user_text in TEST_PROMPTS:
        sx = by_pid_x.get(pid)
        sy = by_pid_y.get(pid)
        if not sx or not sy:
            continue

        flip = random.random() < 0.5
        output_A = sy["output"] if flip else sx["output"]
        output_B = sx["output"] if flip else sy["output"]
        a_is_y = flip

        card = CHARACTER_CARDS.get(char, "")
        prompt = PAIRWISE_PROMPT.format(
            card=card, user=user_text,
            output_A=output_A, output_B=output_B,
        )
        letter = "T"
        for attempt in range(3):
            raw = _ds_call(client, prompt, max_tokens=1024).upper().strip()
            # Accept first A/B/T found in the response (handles leading spaces or punctuation)
            match = re.search(r"\b([ABT])\b", raw) or re.search(r"([ABT])", raw)
            if match and match.group(1) in ("A", "B", "T"):
                letter = match.group(1)
                break
            logger.warning(f"  Pairwise unexpected response [{pid}] attempt {attempt+1}: {raw!r}")

        if letter == "A":
            winner = key_y if a_is_y else key_x
        elif letter == "B":
            winner = key_x if a_is_y else key_y
        else:
            winner = "tie"

        results.append({
            "pair": f"{key_x}_vs_{key_y}",
            "model_x": key_x, "model_y": key_y,
            "prompt_id": pid, "difficulty": difficulty,
            "character": char, "user": user_text,
            "a_is_y": a_is_y, "ds_answer": letter, "winner": winner,
        })
        logger.info(f"  [{key_x} vs {key_y}][{pid}] → {winner}")

    return results


def run_round_robin(all_outputs: dict[str, list[dict]], client) -> list[dict]:
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    all_results = []
    for kx, ky in pairs:
        logger.info(f"\nPairwise: {kx} vs {ky}")
        results = run_pairwise_pair(kx, ky, all_outputs[kx], all_outputs[ky], client)
        all_results.extend(results)
    return all_results


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def build_report(geval_results: list[dict], pairwise_results: list[dict]) -> str:
    lines = [
        "# ArkNarrator — 深度评估报告",
        f"\n_生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        f"_Judge：{DEEPSEEK_MODEL}_\n",
        "## 实验组说明\n",
        "| 组 | 配置 |",
        "|----|------|",
    ]
    for k, cfg in MODEL_CONFIGS.items():
        lines.append(f"| {k} | {cfg['label']} |")

    # --- G-Eval summary ---
    lines.append("\n## G-Eval 多维打分汇总\n")
    lines.append("| 组 | voice | speech | lore | consist | depth | **平均** |")
    lines.append("|----|-------|--------|------|---------|-------|---------|")

    from collections import defaultdict
    dim_scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in geval_results:
        for d in DIMS:
            v = s["geval_scores"].get(d)
            if isinstance(v, (int, float)):
                dim_scores[s["model"]][d].append(v)

    for model_key in MODEL_CONFIGS:
        row_scores = []
        for d in DIMS:
            vals = dim_scores[model_key][d]
            row_scores.append(round(sum(vals)/len(vals), 1) if vals else "—")
        avgs = [v for v in row_scores if isinstance(v, float)]
        overall = round(sum(avgs)/len(avgs), 2) if avgs else "—"
        label = MODEL_CONFIGS[model_key]["label"]
        lines.append(f"| {label} | {' | '.join(str(s) for s in row_scores)} | **{overall}** |")

    # --- G-Eval by difficulty ---
    lines.append("\n## G-Eval 按难度分层\n")
    difficulties = ["easy", "medium", "hard", "lore", "attr"]
    lines.append("| 组 | " + " | ".join(difficulties) + " |")
    lines.append("|----| " + " | ".join(["---"]*len(difficulties)) + " |")

    avg_by_model_diff: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in geval_results:
        if s.get("geval_avg") is not None:
            avg_by_model_diff[s["model"]][s["difficulty"]].append(s["geval_avg"])

    for model_key in MODEL_CONFIGS:
        label = MODEL_CONFIGS[model_key]["label"]
        row = []
        for diff in difficulties:
            vals = avg_by_model_diff[model_key][diff]
            row.append(round(sum(vals)/len(vals), 1) if vals else "—")
        lines.append(f"| {label} | {' | '.join(str(v) for v in row)} |")

    # --- Pairwise round robin ---
    lines.append("\n## 循环赛 Pairwise 结果\n")
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    lines.append("| 对阵 | X 胜 | Y 胜 | 平局 | X 胜率 |")
    lines.append("|------|------|------|------|--------|")

    pair_results = defaultdict(lambda: defaultdict(int))
    for r in pairwise_results:
        pair_results[r["pair"]][r["winner"]] += 1

    for kx, ky in pairs:
        pair_key = f"{kx}_vs_{ky}"
        wins_x = pair_results[pair_key][kx]
        wins_y = pair_results[pair_key][ky]
        ties   = pair_results[pair_key]["tie"]
        total  = wins_x + wins_y + ties
        rate_x = f"{wins_x/total*100:.0f}%" if total else "—"
        lines.append(f"| {kx} vs {ky} | {wins_x} | {wins_y} | {ties} | {rate_x} |")

    # --- Pairwise by difficulty ---
    lines.append("\n## Pairwise 按难度分层（A vs B）\n")
    lines.append("| 难度 | A 胜 | B 胜 | 平局 |")
    lines.append("|------|------|------|------|")
    diff_pair = defaultdict(lambda: defaultdict(int))
    for r in pairwise_results:
        if r["pair"] == "A_vs_B":
            diff_pair[r["difficulty"]][r["winner"]] += 1
    for diff in difficulties:
        d = diff_pair[diff]
        lines.append(f"| {diff} | {d['A']} | {d['B']} | {d['tie']} |")

    # --- Key conclusions ---
    lines.append("\n## 核心结论\n")

    # Compare A vs B overall
    ab = pair_results["A_vs_B"]
    ac = pair_results["A_vs_C"]
    bc = pair_results["B_vs_C"]
    total_ab = sum(ab.values()) or 1
    total_ac = sum(ac.values()) or 1
    total_bc = sum(bc.values()) or 1

    lines.append(f"- **微调价值（A vs B）**：微调 {ab['A']}/{total_ab} 胜率 {ab['A']/total_ab*100:.0f}%，"
                 f"base {ab['B']}/{total_ab} 胜率 {ab['B']/total_ab*100:.0f}%")
    lines.append(f"- **RAG 价值（B vs C）**：base {bc['B']}/{total_bc} 胜率 {bc['B']/total_bc*100:.0f}%，"
                 f"RAG {bc['C']}/{total_bc} 胜率 {bc['C']/total_bc*100:.0f}%")
    lines.append(f"- **微调 vs RAG（A vs C）**：微调 {ac['A']}/{total_ac} 胜率 {ac['A']/total_ac*100:.0f}%，"
                 f"RAG {ac['C']}/{total_ac} 胜率 {ac['C']/total_ac*100:.0f}%")

    # G-Eval averages comparison
    def model_avg(key):
        scores = []
        for s in geval_results:
            if s["model"] == key and s.get("geval_avg") is not None:
                scores.append(s["geval_avg"])
        return round(sum(scores)/len(scores), 2) if scores else None

    avgs = {k: model_avg(k) for k in MODEL_CONFIGS}
    lines.append(f"\nG-Eval 总平均：A={avgs.get('A','—')} | B={avgs.get('B','—')} | C={avgs.get('C','—')}")

    lore_avgs = {}
    for key in MODEL_CONFIGS:
        vals = avg_by_model_diff[key].get("lore", [])
        lore_avgs[key] = round(sum(vals)/len(vals), 2) if vals else None
    lines.append(f"方舟知识专项（lore）：A={lore_avgs.get('A','—')} | B={lore_avgs.get('B','—')} | C={lore_avgs.get('C','—')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generate", action="store_true",
                        help="Load saved outputs instead of generating")
    parser.add_argument("--eval-only", choices=["scoring", "pairwise", "all"],
                        default="all", help="Which eval to run (default: all)")
    args = parser.parse_args()

    outputs_path = RESULTS_DIR / "outputs.json"

    # 1. Generate or load
    if args.skip_generate and outputs_path.exists():
        logger.info(f"Loading saved outputs: {outputs_path}")
        all_outputs_flat = json.loads(outputs_path.read_text(encoding="utf-8"))
    else:
        all_outputs_flat = []
        for model_key in MODEL_CONFIGS:
            logger.info(f"\n{'='*50}\nGenerating: {MODEL_CONFIGS[model_key]['label']}")
            outs = generate_outputs(model_key)
            all_outputs_flat.extend(outs)
        outputs_path.write_text(
            json.dumps(all_outputs_flat, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"\nOutputs saved → {outputs_path}")

    all_outputs: dict[str, list[dict]] = {k: [] for k in MODEL_CONFIGS}
    for s in all_outputs_flat:
        if s["model"] in all_outputs:
            all_outputs[s["model"]].append(s)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not set.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    geval_results, pairwise_results = [], []

    # 2. G-Eval scoring
    if args.eval_only in ("scoring", "all"):
        logger.info("\n" + "="*50 + "\nRunning G-Eval scoring…")
        geval_results = run_geval(all_outputs_flat, client)
        geval_path = RESULTS_DIR / f"geval_{ts}.json"
        geval_path.write_text(
            json.dumps(geval_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"G-Eval saved → {geval_path}")

    # 3. Round-robin pairwise
    if args.eval_only in ("pairwise", "all"):
        logger.info("\n" + "="*50 + "\nRunning round-robin pairwise…")
        pairwise_results = run_round_robin(all_outputs, client)
        pw_path = RESULTS_DIR / f"pairwise_{ts}.json"
        pw_path.write_text(
            json.dumps(pairwise_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Pairwise saved → {pw_path}")

    # 4. Report
    report = build_report(geval_results, pairwise_results)
    report_path = RESULTS_DIR / f"report_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"\nReport → {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
