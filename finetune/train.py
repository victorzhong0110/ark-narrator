"""
ArkNarrator fine-tuning script — supports Qwen2.5 and Gemma 4.
Usage:
  python finetune/train.py --config finetune/config/qwen2_5_lora.yaml
  python finetune/train.py --config finetune/config/gemma4_lora.yaml
"""

import os
import yaml
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Prompt builders per model family ──────────────────────────────────────────

def build_prompt_qwen(sample: dict, tokenizer, system_prompt: str) -> str:
    """Qwen2.5 chat template: <|im_start|> / <|im_end|>"""
    messages = [{"role": "system", "content": system_prompt}]
    user_content = f"{sample['instruction']}\n\n{sample['input']}" if sample.get("input") else sample["instruction"]
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": sample["output"]})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_prompt_gemma(sample: dict, tokenizer, system_prompt: str) -> str:
    """Gemma 4 chat template: <start_of_turn> / <end_of_turn>
    Gemma 4 embeds system prompt inside the first user turn."""
    user_content = f"{system_prompt}\n\n"
    user_content += f"{sample['instruction']}\n\n{sample['input']}" if sample.get("input") else sample["instruction"]
    messages = [
        {"role": "user", "content": user_content},
        {"role": "model", "content": sample["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


PROMPT_BUILDERS = {
    "qwen": build_prompt_qwen,
    "gemma": build_prompt_gemma,
}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_family = cfg["model"].get("model_family", "qwen")
    base_model = cfg["model"]["base_model"]
    use_4bit = cfg["model"].get("load_in_4bit", False)

    logger.info(f"Loading [{model_family}] {base_model} | 4-bit={use_4bit}")

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=cfg["model"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg["model"].get("bnb_4bit_use_double_quant", True),
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float16 if cfg["model"].get("torch_dtype") == "float16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype,
            device_map="auto", trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, model_family


def apply_lora(model, cfg: dict):
    lc = cfg["lora"]
    lora_config = LoraConfig(
        r=lc["r"], lora_alpha=lc["lora_alpha"],
        target_modules=lc["target_modules"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"], task_type=lc["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main(config_path: str):
    cfg = load_config(config_path)
    model_family = cfg["model"].get("model_family", "qwen")
    model, tokenizer, _ = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # Dataset
    dataset = load_dataset("json", data_files={
        "train": cfg["data"]["train_file"],
        "validation": cfg["data"]["eval_file"],
    })

    build_prompt = PROMPT_BUILDERS[model_family]
    system_prompt = cfg["data"]["system_prompt"]
    dataset = dataset.map(lambda s: {"text": build_prompt(s, tokenizer, system_prompt)})

    # Training
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        evaluation_strategy=t["evaluation_strategy"],
        save_total_limit=t["save_total_limit"],
        report_to=t["report_to"],
        run_name=f"ark-narrator-{model_family}",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=t["max_seq_length"],
        args=training_args,
    )

    logger.info(f"Training [{model_family}]...")
    trainer.train()

    out = Path(t["output_dir"]) / "final"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
