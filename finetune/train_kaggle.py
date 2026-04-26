"""
QLoRA fine-tuning for ArkNarrator — HuggingFace PEFT + TRL SFTTrainer.
Designed for Kaggle A100 (40 GB VRAM), but runs on any CUDA GPU.

Key differences from the MLX local script:
  - Full-precision base model (Qwen/Qwen2.5-7B-Instruct), 4-bit quantized at runtime
  - SFTTrainer handles chat-template application and loss masking
  - load_best_model_at_end=True replaces manual best-checkpoint logic
  - Early stopping with patience=3 prevents NaN / overfit runaway
  - ~100x faster than local MLX: 1500 steps ≈ 5-10 min on A100

Usage (Kaggle notebook cell):
  !python /kaggle/input/ark-narrator-code/finetune/train_kaggle.py \\
      --config /kaggle/input/ark-narrator-code/finetune/config/qwen2_5_kaggle.yaml \\
      --format roleplay \\
      --data-dir /kaggle/input/ark-narrator-data/processed

Usage (local, for debugging):
  python finetune/train_kaggle.py \\
      --config finetune/config/qwen2_5_kaggle.yaml \\
      --format roleplay
"""

import os
import sys
import json
import yaml
import random
import logging
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FORMATS = ["narrative", "dialogue_window", "roleplay"]

# Qwen2.5 linear layer names targeted by LoRA (all attention + MLP projections)
QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_chat_template(sample: dict, tokenizer) -> dict:
    """
    Convert a dataset sample to a single `text` string.

    narrative format:  {"text": "..."} → pass through
    chat formats:      {"messages": [...]} → apply Qwen chat template
                       We include ALL roles so the model trains on the
                       full conversation (system + user + assistant turns).
    """
    if "text" in sample:
        return {"text": sample["text"]}

    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_format_dataset(
    fmt: str,
    data_dir: str,
    tokenizer,
) -> tuple[Dataset, Dataset]:
    train_path = Path(data_dir) / f"{fmt}_train.jsonl"
    eval_path  = Path(data_dir) / f"{fmt}_eval.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run: python data_pipeline/dataset_builder.py"
        )

    raw_train = load_jsonl(str(train_path))
    raw_eval  = load_jsonl(str(eval_path))

    fn = lambda s: apply_chat_template(s, tokenizer)
    train_ds = Dataset.from_list(raw_train).map(fn, remove_columns=list(raw_train[0].keys()))
    eval_ds  = Dataset.from_list(raw_eval).map(fn, remove_columns=list(raw_eval[0].keys()))

    logger.info(f"[{fmt}] {len(train_ds):,} train / {len(eval_ds):,} eval samples")
    return train_ds, eval_ds


def load_combined_dataset(
    data_dir: str,
    tokenizer,
) -> tuple[Dataset, Dataset]:
    all_train, all_eval = [], []
    for fmt in FORMATS:
        tr, ev = load_format_dataset(fmt, data_dir, tokenizer)
        all_train.extend(tr)
        all_eval.extend(ev)
    random.shuffle(all_train)
    logger.info(f"[combined] {len(all_train):,} train / {len(all_eval):,} eval samples")
    return Dataset.from_list(all_train), Dataset.from_list(all_eval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument(
        "--format", default="roleplay",
        choices=FORMATS + ["combined", "all"],
    )
    parser.add_argument("--data-dir", default="./data/processed")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg  = cfg["lora"]
    t         = cfg["training"]

    formats_to_run = FORMATS if args.format == "all" else [args.format]

    for fmt in formats_to_run:
        output_dir = t["output_dir"].rstrip("/") + f"_{fmt}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── 1. QLoRA quantisation ────────────────────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # ── 2. Tokenizer & model ─────────────────────────────────────────────
        base_model = model_cfg["base_model"]
        logger.info(f"Loading tokenizer: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        logger.info(f"Loading model: {base_model} (4-bit QLoRA)")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        model.config.use_cache = False

        # ── 3. LoRA config ───────────────────────────────────────────────────
        rank  = lora_cfg["rank"]
        alpha = lora_cfg.get("alpha", rank * 2)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=QWEN_TARGET_MODULES,
            bias="none",
        )
        logger.info(f"LoRA: rank={rank}, alpha={alpha}, targets={QWEN_TARGET_MODULES}")

        # ── 4. Dataset ───────────────────────────────────────────────────────
        if fmt == "combined":
            train_ds, eval_ds = load_combined_dataset(args.data_dir, tokenizer)
        else:
            train_ds, eval_ds = load_format_dataset(fmt, args.data_dir, tokenizer)

        # ── 5. Training config ───────────────────────────────────────────────
        max_steps = (
            t.get("iters_per_format", {}).get(fmt)
            or t.get("iters", 1000)
        )
        batch_size   = t.get("batch_size", 2)
        grad_accum   = t.get("gradient_accumulation_steps", 4)
        save_every   = t.get("save_every", 100)
        max_seq      = t["max_seq_length"]
        # Set max length on tokenizer — compatible with both old and new TRL.
        # Older TRL (<=0.11) read this from SFTConfig; newer TRL reads it here.
        tokenizer.model_max_length = max_seq

        logger.info(
            f"\n{'='*55}\n"
            f"  Format   : {fmt}\n"
            f"  Steps    : {max_steps}  (eff. batch={batch_size * grad_accum})\n"
            f"  Seq len  : {max_seq}\n"
            f"  Output   : {output_dir}\n"
            f"{'='*55}"
        )

        sft_cfg = SFTConfig(
            output_dir=output_dir,

            # Steps
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,

            # LR with cosine decay (built-in scheduler)
            learning_rate=t["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_steps=t.get("warmup", 100),

            # Stability
            max_grad_norm=t.get("grad_clip", 0.5),
            bf16=True,
            gradient_checkpointing=True,

            # Eval & checkpointing
            eval_strategy="steps",
            eval_steps=save_every,
            save_strategy="steps",
            save_steps=save_every,
            save_total_limit=5,           # keep only 5 checkpoints on disk
            load_best_model_at_end=True,  # auto-selects best val-loss checkpoint
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Data (max_seq_length is set on tokenizer above for new TRL compat)
            dataset_text_field="text",
            packing=False,

            # Logging
            logging_steps=10,
            report_to="none",             # set "wandb" if you want W&B tracking
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
            callbacks=[
                # Stop early if eval_loss doesn't improve for 3 evaluations
                EarlyStoppingCallback(early_stopping_patience=3),
            ],
        )

        # ── 6. Train ─────────────────────────────────────────────────────────
        trainer.train()

        # ── 7. Save best adapter ─────────────────────────────────────────────
        # load_best_model_at_end=True already loaded the best checkpoint into
        # `model`. Save it now so inference can use adapter_path=output_dir.
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"✓ [{fmt}] Best adapter saved → {output_dir}")

    logger.info("\n✓ All formats complete.")


if __name__ == "__main__":
    main()
