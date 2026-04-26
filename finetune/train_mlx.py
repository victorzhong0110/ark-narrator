"""
MLX LoRA fine-tuning for three dataset formats.

Formats supported:
  narrative       -> {"text": "..."} raw text, causal LM
  dialogue_window -> {"messages": [...]} chat format
  roleplay        -> {"messages": [...]} chat format (multi-turn)
  combined        -> all three merged into one training run
  all             -> train each format separately (three runs)

mlx-lm handles both {"text"} and {"messages"} natively — no manual
template application needed.

Usage:
  python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml --format combined
  python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml --format roleplay
  python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml --format all
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import yaml
import copy
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORMATS = ["narrative", "dialogue_window", "roleplay"]

# Regex to extract val loss lines from mlx-lm training output:
#   "Iter 200: Val loss 2.389, Val took 8.892s"
VAL_LOSS_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_combined_data_dir(processed_dir: str = "./data/processed") -> str:
    """
    Merge all three formats into a single train.jsonl + valid.jsonl.
    mlx-lm handles {"text": ...} and {"messages": [...]} in the same file.
    """
    import random, json

    mlx_dir = Path("./data/mlx_format_combined")
    mlx_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "eval"):
        all_lines = []
        for fmt in FORMATS:
            src = Path(processed_dir) / f"{fmt}_{split}.jsonl"
            if not src.exists():
                raise FileNotFoundError(
                    f"Missing {src} — run: python data_pipeline/dataset_builder.py"
                )
            with open(src, encoding="utf-8") as f:
                lines = f.readlines()
            all_lines.extend(lines)
            logger.info(f"  [{fmt}] {split}: {len(lines):,} samples")

        random.shuffle(all_lines)

        out_name = "train.jsonl" if split == "train" else "valid.jsonl"
        out_path = mlx_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(all_lines)
        logger.info(f"Combined {split}: {len(all_lines):,} samples -> {out_path}")

    return str(mlx_dir)


def prepare_mlx_data_dir(fmt: str, processed_dir: str = "./data/processed") -> str:
    """
    mlx-lm expects a directory with train.jsonl + valid.jsonl.
    Copy the format-specific files into a temp dir.
    """
    src_train = Path(processed_dir) / f"{fmt}_train.jsonl"
    src_eval  = Path(processed_dir) / f"{fmt}_eval.jsonl"

    if not src_train.exists():
        raise FileNotFoundError(
            f"Training data not found: {src_train}\n"
            f"Run: python data_pipeline/dataset_builder.py"
        )

    mlx_dir = Path(f"./data/mlx_format_{fmt}")
    mlx_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(src_train, mlx_dir / "train.jsonl")
    shutil.copy(src_eval,  mlx_dir / "valid.jsonl")

    train_count = sum(1 for _ in open(src_train, encoding="utf-8"))
    eval_count  = sum(1 for _ in open(src_eval,  encoding="utf-8"))
    logger.info(f"[{fmt}] {train_count} train / {eval_count} eval samples -> {mlx_dir}")

    return str(mlx_dir)


def select_best_checkpoint(output_dir: str, log_lines: list[str]) -> str | None:
    """
    Parse val loss lines from captured training output, find the iteration with
    the lowest val loss, and copy that checkpoint to best_adapters.safetensors.

    Returns the path to best_adapters.safetensors, or None if no checkpoints found.
    """
    best_iter, best_loss = None, float("inf")
    for line in log_lines:
        m = VAL_LOSS_RE.search(line)
        if m:
            it, loss = int(m.group(1)), float(m.group(2))
            if loss < best_loss:
                best_loss, best_iter = loss, it

    if best_iter is None:
        logger.warning("Could not parse any val loss from training output — skipping best checkpoint selection.")
        return None

    src = Path(output_dir) / f"{best_iter:07d}_adapters.safetensors"
    dst = Path(output_dir) / "best_adapters.safetensors"

    if not src.exists():
        logger.warning(f"Best checkpoint file not found: {src} — skipping copy.")
        return None

    shutil.copy(src, dst)
    # Also overwrite the primary adapters.safetensors so --adapter-path <dir>
    # always loads the best weights, not the (potentially NaN) final weights.
    primary = Path(output_dir) / "adapters.safetensors"
    shutil.copy(src, primary)
    logger.info(
        f"✓ Best checkpoint: iter {best_iter}, val loss {best_loss:.3f} "
        f"→ {dst} (also → {primary})"
    )
    return str(dst)


def run_mlx_training(cfg: dict, fmt: str):
    t    = cfg["training"]
    lora = cfg["lora"]

    # Each format gets its own checkpoint directory
    output_dir = t["output_dir"].rstrip("/") + f"_{fmt}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if fmt == "combined":
        logger.info("Merging all formats into combined dataset...")
        data_dir = prepare_combined_data_dir()
    else:
        data_dir = prepare_mlx_data_dir(fmt)

    # Use per-format iters if defined, else fall back to global iters
    iters = (
        t.get("iters_per_format", {}).get(fmt)
        or t.get("iters", 1000)
    )

    mlx_cfg = {
        "model":           cfg["model"]["base_model"],
        "train":           True,
        "data":            data_dir,
        "iters":           iters,
        "batch_size":      t["batch_size"],
        "learning_rate":   t["learning_rate"],
        "num_layers":      lora.get("num_layers", 8),
        "lora_parameters": {
            "rank":    lora["rank"],
            "scale":   lora.get("scale", lora["rank"] * 2.0),
            "dropout": lora.get("dropout", 0.05),
        },
        "val_batches":    t["val_batches"],
        "save_every":     t["save_every"],
        "adapter_path":   output_dir,
        "max_seq_length": t["max_seq_length"],
    }
    if t.get("grad_checkpoint"):
        mlx_cfg["grad_checkpoint"] = True
    if t.get("grad_clip") is not None:
        mlx_cfg["grad_clip"] = t["grad_clip"]
    if t.get("warmup"):
        mlx_cfg["warmup"] = t["warmup"]

    # lr_schedule: only pass if it is a dict.
    # Plain string (e.g. "cosine_decay") causes TypeError in build_schedule() —
    # this mlx-lm version requires {"name": ..., "warmup": ..., "arguments": [...]}.
    lr_sched = t.get("lr_schedule")
    if isinstance(lr_sched, dict):
        # Deep-copy so we can safely mutate arguments without touching the original cfg.
        sched = copy.deepcopy(lr_sched)
        # Override decay_steps (arguments[1]) to match this format's actual iters,
        # so cosine decay reaches zero exactly at the end of training.
        args = sched.get("arguments", [])
        if len(args) >= 2:
            args[1] = iters
            sched["arguments"] = args
        mlx_cfg["lr_schedule"] = sched
        logger.info(f"LR schedule: cosine_decay over {iters} iters (warmup={sched.get('warmup', 0)})")
    elif lr_sched is not None:
        logger.warning(
            f"lr_schedule in config is type {type(lr_sched).__name__!r} (expected dict) — "
            "skipping. Use dict format: {name: cosine_decay, warmup: 100, arguments: [5e-5, N, 0]}"
        )

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(mlx_cfg, tmp, allow_unicode=True)
    tmp.close()

    logger.info(f"\n{'='*50}")
    logger.info(f"Training format: [{fmt}]")
    logger.info(f"Model: {cfg['model']['base_model']}")
    logger.info(f"Output: {output_dir}  |  iters: {iters}  |  max_seq_length: {t['max_seq_length']}")
    logger.info(f"{'='*50}")

    # Stream output to stdout AND capture for best-checkpoint parsing.
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", tmp.name]
    captured_lines: list[str] = []

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_lines.append(line)
    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    logger.info(f"✓ [{fmt}] Adapter saved -> {output_dir}")

    # Auto-select best checkpoint from training log
    select_best_checkpoint(output_dir, captured_lines)

    return output_dir


def run_mlx_inference(model: str, adapter_path: str, prompt: str, fmt: str):
    """Quick inference test after training.

    --adapter-path must be a directory (mlx-lm looks for adapter_config.json
    there). select_best_checkpoint() has already overwritten adapters.safetensors
    with the best weights, so we always pass the directory.
    """
    logger.info(f"\n=== Quick inference test [{fmt}] (adapter dir: {adapter_path}) ===")
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model",        model,
        "--adapter-path", adapter_path,   # directory, not a file
        "--prompt",       prompt,
        "--max-tokens",   "300",
        "--temp",         "0.8",
    ]
    subprocess.run(cmd, check=True)


TEST_PROMPTS = {
    "narrative":       "【黎明前奏·城市之殇】\n阿米娅：博士，我们需要",
    "dialogue_window": "【场景：罗德岛作战室】\n凯尔希：本次任务目标已确认。\n阿米娅：博士，您怎么看？\n博士：我认为风险可控。\n凯尔希：那么，",
    "roleplay":        "你好，能天使。最近任务怎么样？",
    "combined":        "你好，能天使。最近任务怎么样？",
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--format", default="combined",
        choices=FORMATS + ["combined", "all"],
        help="Which dataset format to train on (default: combined)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    formats_to_run = FORMATS if args.format == "all" else [args.format]

    for fmt in formats_to_run:
        adapter_path = run_mlx_training(cfg, fmt)
        run_mlx_inference(
            cfg["model"]["base_model"],
            adapter_path,
            TEST_PROMPTS.get(fmt, "请介绍明日方舟的世界观。"),
            fmt,
        )

    logger.info("\n✓ All formats complete.")
