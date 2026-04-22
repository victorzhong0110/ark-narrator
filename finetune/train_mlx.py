"""
MLX LoRA fine-tuning — runs natively on Apple Silicon (no CUDA needed).
Requires: pip install mlx-lm

Usage:
  # Qwen2.5-7B（推荐，24GB 很舒适）
  python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml

  # Gemma 4 27B-A4B（本地快速验证，正式训练建议用 Kaggle A100）
  python finetune/train_mlx.py --config finetune/config/gemma4_mlx.yaml
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def convert_dataset_to_mlx_format(train_file: str, eval_file: str, system_prompt: str):
    """
    MLX-LM expects JSONL with {"text": "..."} format using the model's chat template.
    We convert from our {"instruction", "input", "output"} format.
    """
    import json

    def convert(in_path: str, out_path: str):
        records = []
        with open(in_path, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                user = f"{s['instruction']}\n\n{s['input']}" if s.get("input") else s["instruction"]
                # Use plain chat format compatible with both Qwen and Gemma
                text = (
                    f"<|system|>{system_prompt}<|end|>\n"
                    f"<|user|>{user}<|end|>\n"
                    f"<|assistant|>{s['output']}<|end|>"
                )
                records.append({"text": text})

        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Converted {len(records)} samples → {out_path}")

    mlx_dir = Path("./data/mlx_format")
    mlx_dir.mkdir(parents=True, exist_ok=True)
    convert(train_file, str(mlx_dir / "train.jsonl"))
    convert(eval_file, str(mlx_dir / "valid.jsonl"))
    return str(mlx_dir)


def run_mlx_training(cfg: dict):
    """Call mlx_lm.lora via subprocess — keeps dependency clean."""
    t = cfg["training"]
    lora = cfg["lora"]
    data_dir = convert_dataset_to_mlx_format(
        cfg["data"]["train_file"],
        cfg["data"]["eval_file"],
        cfg["data"]["system_prompt"],
    )

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model",          cfg["model"]["base_model"],
        "--train",
        "--data",           data_dir,
        "--iters",          str(t["iters"]),
        "--batch-size",     str(t["batch_size"]),
        "--learning-rate",  str(t["learning_rate"]),
        "--lora-layers",    "16",
        "--rank",           str(lora["rank"]),
        "--val-batches",    str(t["val_batches"]),
        "--save-every",     str(t["save_every"]),
        "--adapter-path",   t["output_dir"],
        "--max-seq-length", str(t["max_seq_length"]),
    ]

    if t.get("grad_checkpoint"):
        cmd.append("--grad-checkpoint")

    logger.info(f"Running MLX training: {cfg['model']['base_model']}")
    logger.info(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"✓ Adapter saved → {t['output_dir']}")


def run_mlx_inference(model: str, adapter_path: str, prompt: str):
    """Quick inference test after training."""
    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model",        model,
        "--adapter-path", adapter_path,
        "--prompt",       prompt,
        "--max-tokens",   "300",
        "--temp",         "0.8",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--test",
        default="以干员能天使的口吻，写一段与博士初次见面的自我介绍。",
        help="Quick inference test after training",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_mlx_training(cfg)

    # Quick test
    logger.info("\n=== Quick inference test ===")
    run_mlx_inference(cfg["model"]["base_model"], cfg["training"]["output_dir"], args.test)
