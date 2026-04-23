# ============================================================
# ArkNarrator — Kaggle Training Notebook
# GPU: T4 x2 or P100 (free) / A100 (preferred)
# 运行方式: 在 Kaggle Notebook 中逐 cell 执行
# ============================================================

# ── Cell 1: 安装依赖 ─────────────────────────────────────────
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])

pip("transformers>=4.40.0", "peft>=0.10.0", "trl>=0.8.0",
    "datasets>=2.18.0", "accelerate>=0.28.0",
    "bitsandbytes>=0.43.0", "wandb", "huggingface_hub")

print("✓ 依赖安装完成")

# ── Cell 2: 配置密钥（在 Kaggle Secrets 里设置，不要硬编码）─
from kaggle_secrets import UserSecretsClient
import os, wandb

secrets = UserSecretsClient()
os.environ["HF_TOKEN"]      = secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")

wandb.login()
print("✓ 密钥配置完成")

# ── Cell 3: 拉取项目代码 ─────────────────────────────────────
subprocess.run(["git", "clone",
    "https://github.com/victorzhong0110/ark-narrator.git",
    "/kaggle/working/ark-narrator"], check=True)

os.chdir("/kaggle/working/ark-narrator")
sys.path.insert(0, "/kaggle/working/ark-narrator")
print("✓ 代码拉取完成")

# ── Cell 4: 数据准备 ─────────────────────────────────────────
from data_pipeline.scraper import PRTSScraper
from data_pipeline.dataset_builder import DatasetBuilder
import json

scraper = PRTSScraper()
scraper.scrape_all()

with open("./data/raw/operator_profiles.json", encoding="utf-8") as f:
    profiles = json.load(f)

builder = DatasetBuilder()
builder.build_from_profiles(profiles)
builder.to_jsonl()
print(f"✓ 数据集就绪: train={sum(1 for _ in open('data/processed/train.jsonl'))} 条")

# ── Cell 5A: 训练 Qwen2.5-7B（QLoRA）────────────────────────
# 预计耗时: ~2-3h on T4, ~1h on A100
from finetune.train import main as train_main

train_main("./finetune/config/qwen2_5_lora.yaml")
print("✓ Qwen2.5-7B 训练完成")

# ── Cell 5B: 训练 Gemma 4 27B-A4B（QLoRA）──────────────────
# 预计耗时: ~3-4h on A100（T4 显存不够，跳过）
import torch
if torch.cuda.get_device_properties(0).total_memory > 30 * 1024**3:
    train_main("./finetune/config/gemma4_lora.yaml")
    print("✓ Gemma 4 27B-A4B 训练完成")
else:
    print("⚠ 显存不足 30GB，跳过 Gemma 27B（需要 A100）")

# ── Cell 6: 上传 adapter 到 HuggingFace Hub ─────────────────
from huggingface_hub import HfApi
api = HfApi()

for model_dir, repo_name in [
    ("./checkpoints/qwen2_5_ark/final", "ark-narrator-qwen2.5-7b-lora"),
    ("./checkpoints/gemma4_27b_ark/final", "ark-narrator-gemma4-27b-lora"),
]:
    from pathlib import Path
    if Path(model_dir).exists():
        repo_id = f"victorzhong0110/{repo_name}"
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=model_dir, repo_id=repo_id)
        print(f"✓ 上传完成: https://huggingface.co/{repo_id}")

# ── Cell 7: 快速推理测试 ─────────────────────────────────────
from inference.engine import ArkNarratorEngine

engine = ArkNarratorEngine(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    adapter_path="./checkpoints/qwen2_5_ark/final",
)

test_prompts = [
    "以干员能天使的口吻，写一段与博士初次见面的自我介绍。",
    "请介绍干员凯尔希的背景，她在罗德岛扮演什么角色？",
    "源石病在泰拉大陆是如何传播的？",
]

for prompt in test_prompts:
    print(f"\n指令: {prompt}")
    print(f"输出: {engine.generate(prompt)}")
    print("-" * 60)
