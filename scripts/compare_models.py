"""
Side-by-side comparison of Gemma 4 4B vs Qwen2.5-7B after fine-tuning.
Generates a markdown report with scores and example outputs.

Usage:
  python scripts/compare_models.py --samples data/sample/sample_data.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

from inference.engine import ArkNarratorEngine
from eval.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {
    "gemma4-ark": {
        "base_model": "google/gemma-4-27b-it",
        "adapter_path": "./checkpoints/gemma4_27b_ark/final",
        "label": "Gemma 4 27B-A4B + QLoRA",
        "params": "27B (MoE, ~4B active)",
        "vram": "~14 GB (inference)",
    },
    "qwen25-ark": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": "./checkpoints/qwen2_5_ark/final",
        "label": "Qwen2.5 7B + QLoRA",
        "params": "7B",
        "vram": "~12 GB",
    },
}


def run_comparison(samples: list[dict]) -> dict:
    results = {}
    evaluator = Evaluator(use_gpt_judge=True)

    for model_id, cfg in MODELS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {cfg['label']}")
        logger.info(f"{'='*50}")

        engine = ArkNarratorEngine(cfg["base_model"], cfg["adapter_path"])
        outputs = []
        for s in samples:
            output = engine.generate(s["instruction"], s.get("input", ""))
            outputs.append({**s, "output": output, "model": model_id})

        eval_results = evaluator.evaluate_outputs(outputs, model_id)
        results[model_id] = {
            "config": cfg,
            "eval_results": eval_results,
            "outputs": outputs,
        }

        del engine  # free VRAM between models

    return results


def generate_report(results: dict) -> str:
    lines = [
        "# ArkNarrator — Model Comparison Report",
        f"\n> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n---\n",
        "## 📊 Quantitative Results\n",
        "| Model | Params | VRAM | 世界观还原 | 角色一致性 | 语言流畅度 | 综合 |",
        "|-------|--------|------|-----------|-----------|-----------|------|",
    ]

    for model_id, data in results.items():
        cfg = data["config"]
        evals = data["eval_results"]
        avg = lambda key: round(sum(getattr(e, key) for e in evals) / len(evals), 2)
        lines.append(
            f"| {cfg['label']} | {cfg['params']} | {cfg['vram']} "
            f"| {avg('lore_score')} | {avg('consistency_score')} "
            f"| {avg('fluency_score')} | {avg('overall')} |"
        )

    lines += ["\n---\n", "## 💬 Sample Outputs\n"]

    # Show side-by-side examples
    sample_outputs = {mid: data["outputs"] for mid, data in results.items()}
    n_samples = min(3, len(list(sample_outputs.values())[0]))

    for i in range(n_samples):
        instruction = list(sample_outputs.values())[0][i]["instruction"]
        lines.append(f"### 样本 {i+1}\n")
        lines.append(f"**指令：** {instruction}\n")
        for model_id, outputs in sample_outputs.items():
            label = results[model_id]["config"]["label"]
            lines.append(f"**{label}：**")
            lines.append(f"> {outputs[i]['output'][:300]}...\n")

    lines += [
        "\n---\n",
        "## 🔍 结论\n",
        "| 维度 | 胜出模型 | 说明 |",
        "|------|---------|------|",
        "| 中文生成质量 | TBD | 填入评估后结论 |",
        "| 推理速度 | Gemma 4 4B | 参数量更小，延迟更低 |",
        "| 显存占用 | Gemma 4 4B | float16 LoRA，无需4bit量化 |",
        "| 世界观还原 | TBD | 填入评估后结论 |",
        "| 生产部署推荐 | TBD | 综合评估后填入 |",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/sample/sample_data.json")
    parser.add_argument("--output", default="eval/results/model_comparison.md")
    args = parser.parse_args()

    with open(args.samples, encoding="utf-8") as f:
        samples = json.load(f)

    results = run_comparison(samples)
    report = generate_report(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(f"\nReport saved → {out_path}")
    print(report)


if __name__ == "__main__":
    main()
