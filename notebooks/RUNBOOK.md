# ArkNarrator 训练运行手册

> 按顺序执行，每步把输出贴给 Claude，Claude 会生成训练日志。

---

## 阶段一：本地验证（Mac MLX，今天）

### Step 1 — 安装 MLX

```bash
cd ~/Desktop/ArkNarrator
pip install mlx-lm
```

期望输出：`Successfully installed mlx-lm-x.x.x`

---

### Step 2 — 检查数据集

```bash
python3 -c "
import json
train = sum(1 for _ in open('data/processed/train.jsonl'))
eval_ = sum(1 for _ in open('data/processed/eval.jsonl'))
print(f'train: {train} 条')
print(f'eval:  {eval_} 条')

# 抽查一条
with open('data/processed/train.jsonl') as f:
    d = json.loads(f.readline())
print(f'task_type: {d[\"task_type\"]}')
print(f'instruction: {d[\"instruction\"]}')
print(f'output前100字: {d[\"output\"][:100]}')
"
```

---

### Step 3 — 本地 MLX 微调（Qwen2.5-7B）

```bash
python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml
```

预计时间：1-3 小时（取决于 M 系列芯片型号）  
关键输出：每 100 步打印一次 train loss 和 val loss

---

### Step 4 — 本地推理测试

```bash
python3 -c "
import sys, os
sys.path.insert(0, '.')
import subprocess

prompts = [
    '以干员能天使的口吻，写一段与博士初次见面的自我介绍。',
    '请介绍干员凯尔希的背景，她在罗德岛扮演什么角色？',
    '源石病在泰拉大陆是如何传播的？',
]

for p in prompts:
    print(f'=== 指令 ===')
    print(p)
    print(f'=== 输出 ===')
    result = subprocess.run([
        sys.executable, '-m', 'mlx_lm.generate',
        '--model', 'mlx-community/Qwen2.5-7B-Instruct-4bit',
        '--adapter-path', './checkpoints/qwen2_5_mlx',
        '--prompt', p,
        '--max-tokens', '200',
        '--temp', '0.8',
    ], capture_output=True, text=True)
    print(result.stdout)
    print()
"
```

---

## 阶段二：Kaggle 正式训练（双模型对比）

### Step 5 — Kaggle 环境准备

1. 登录 [kaggle.com](https://kaggle.com) → Code → **New Notebook**
2. 右侧 Settings：
   - **Accelerator** → GPU T4 x2（免费）或 A100（如有配额）
   - **Internet** → 开启（需要下载模型）
3. 左侧 Add-ons → **Secrets** → 添加：
   - `HF_TOKEN` = 你的 HuggingFace token
   - `WANDB_API_KEY` = 你的 W&B key

---

### Step 6 — Kaggle Cell 1：环境安装

新建 Code Cell，粘贴运行：

```python
import subprocess, sys

pkgs = [
    "transformers>=4.40.0", "peft>=0.10.0", "trl>=0.8.0",
    "datasets>=2.18.0", "accelerate>=0.28.0",
    "bitsandbytes>=0.43.0", "wandb", "huggingface_hub",
    "beautifulsoup4", "sse-starlette", "fastapi",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "")
```

---

### Step 7 — Kaggle Cell 2：密钥 + 克隆代码

```python
from kaggle_secrets import UserSecretsClient
import os, wandb

secrets = UserSecretsClient()
os.environ["HF_TOKEN"]      = secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
wandb.login()

import subprocess, sys
subprocess.run(["git", "clone",
    "https://github.com/victorzhong0110/ark-narrator.git",
    "/kaggle/working/ark-narrator"], check=True)

os.chdir("/kaggle/working/ark-narrator")
sys.path.insert(0, "/kaggle/working/ark-narrator")
print("✓ Ready")
```

---

### Step 8 — Kaggle Cell 3：数据准备

```python
from data_pipeline.scraper import PRTSScraper
from data_pipeline.dataset_builder import DatasetBuilder
import json

PRTSScraper().scrape_all()

with open("./data/raw/operator_profiles.json", encoding="utf-8") as f:
    profiles = json.load(f)

builder = DatasetBuilder()
builder.build_from_profiles(profiles)
builder.to_jsonl()

train_n = sum(1 for _ in open("data/processed/train.jsonl"))
eval_n  = sum(1 for _ in open("data/processed/eval.jsonl"))
print(f"✓ train={train_n}, eval={eval_n}")
```

---

### Step 9 — Kaggle Cell 4A：训练 Qwen2.5-7B

```python
from finetune.train import main as train_main
train_main("./finetune/config/qwen2_5_lora.yaml")
print("✓ Qwen2.5-7B Done")
```

---

### Step 10 — Kaggle Cell 4B：训练 Gemma 4 27B-A4B

```python
import torch
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM: {vram_gb:.1f} GB")

if vram_gb >= 30:
    train_main("./finetune/config/gemma4_lora.yaml")
    print("✓ Gemma 4 27B Done")
else:
    print(f"⚠ VRAM {vram_gb:.1f}GB 不足，需要 A100 (40GB)")
```

---

### Step 11 — Kaggle Cell 5：上传到 HuggingFace

```python
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
models = {
    "./checkpoints/qwen2_5_ark/final": "ark-narrator-qwen2.5-7b-lora",
    "./checkpoints/gemma4_27b_ark/final": "ark-narrator-gemma4-27b-lora",
}

for local_path, repo_name in models.items():
    if Path(local_path).exists():
        repo_id = f"victorzhong0110/{repo_name}"
        api.create_repo(repo_id, exist_ok=True, private=True)
        api.upload_folder(folder_path=local_path, repo_id=repo_id)
        print(f"✓ https://huggingface.co/{repo_id}")
```

---

## 阶段三：评估（训练完成后）

### Step 12 — 生成对比报告

```bash
# 本地运行（需要 OpenAI API key 做 GPT-4o Judge）
export OPENAI_API_KEY=sk-xxx
python scripts/compare_models.py
cat eval/results/model_comparison.md
```

---

## 把以下内容贴给 Claude 生成训练日志

每个 Step 完成后，把以下信息发给 Claude：

```
Step X 完成
输出：[粘贴终端/Notebook 输出]
耗时：[大约几分钟]
```

Claude 会整理成结构化训练日志，最终写入 README。
