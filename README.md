<div align="center">

# 🎮 ArkNarrator

**Arknights × LLM — 游戏叙事内容生成系统**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> 基于明日方舟（Arknights）世界观语料，对 **Gemma 4 4B** 与 **Qwen2.5-7B** 分别进行 LoRA 微调，  
> 构建面向游戏策划的 NPC 对话生成、干员档案撰写与世界观一致性校验工具链。  
> 项目核心产出：**两模型的系统性对比实验报告**。

---

## 🧪 核心实验：Gemma 4 4B vs Qwen2.5-7B

| | Gemma 4 4B + LoRA | Qwen2.5-7B + QLoRA |
|---|---|---|
| 参数量 | 4B | 7B |
| 显存需求 | ~10 GB（float16） | ~12 GB（4-bit） |
| 微调方式 | LoRA（无需量化） | QLoRA（BitsAndBytes 4-bit） |
| 中文基础 | 多语言增强版 | 中文原生优化 |
| 推理速度 | ⚡ 更快 | 🐢 较慢 |
| 适用场景 | 轻量部署、快速迭代 | 高质量中文生成 |

> 评估结果（GPT-4o Judge，满分 10 分）训练完成后更新 ↓

| 模型 | 世界观还原 | 角色一致性 | 语言流畅度 | 综合 |
|------|-----------|-----------|-----------|------|
| Gemma 4 4B + LoRA | - | - | - | - |
| Qwen2.5-7B + QLoRA | - | - | - | - |
| GPT-4o（无微调基线） | - | - | - | - |

---

## 架构

```
原始数据（PRTS Wiki）
        │
        ▼
  data_pipeline/         ← 爬取 + 清洗 + 构建 instruction-tuning 数据集
        │
        ├─────────────────────────────┐
        ▼                            ▼
finetune/config/             finetune/config/
qwen2_5_lora.yaml            gemma4_lora.yaml
        │                            │
        ▼                            ▼
  Qwen2.5-7B + QLoRA      Gemma 4 4B + LoRA
        │                            │
        └──────────┬─────────────────┘
                   ▼
             eval/ + scripts/compare_models.py
             （GPT-4o Judge 双重评估 + 对比报告）
                   │
                   ▼
             inference/server.py
             （FastAPI + SSE，接入胜出模型）
```

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 模型 A | Gemma 4 4B-IT（Google DeepMind） |
| 模型 B | Qwen2.5-7B-Instruct（Alibaba） |
| 微调框架 | HuggingFace PEFT + TRL SFTTrainer |
| 量化 | BitsAndBytes QLoRA（仅 Qwen） |
| 实验追踪 | Weights & Biases |
| 推理服务 | FastAPI + SSE 流式输出 |
| 评估 | GPT-4o Judge + Rule-based LoreChecker |
| 数据处理 | BeautifulSoup4 + pandas |
| 训练环境 | Google Colab Pro / Kaggle（A100 40G） |

---

## 数据集

来源：PRTS Wiki（prts.wiki）干员档案、主线剧情、世界观文本

| 任务类型 | 样本数 | 说明 |
|---------|--------|------|
| profile_qa | ~1,500 | 干员背景问答 |
| dialogue | ~1,000 | 角色对话生成 |
| worldbuilding | ~500 | 世界观内容创作 |
| **合计** | **~3,000** | **train : eval = 9 : 1** |

---

## 快速开始

```bash
git clone https://github.com/victorzhong0110/ark-narrator.git
cd ark-narrator
pip install -r requirements.txt
cp .env.example .env

# 1. 收集数据
python scripts/run_pipeline.py --mode scrape

# 2. 构建数据集
python scripts/run_pipeline.py --mode build

# 3A. 训练 Gemma 4 4B（需要 ~10G VRAM）
python finetune/train.py --config finetune/config/gemma4_lora.yaml

# 3B. 训练 Qwen2.5-7B（需要 ~12G VRAM）
python finetune/train.py --config finetune/config/qwen2_5_lora.yaml

# 4. 对比评估（生成 eval/results/model_comparison.md）
python scripts/compare_models.py

# 5. 启动推理服务
python scripts/run_pipeline.py --mode serve
```

---

## 项目关联

本项目与以下作品共同构成游戏 AI 作品集：

- [Arknights-Gacha-Analysis-System](https://github.com/victorzhong0110/Arknights-Gacha-Analysis-System) — 明日方舟玩家流失预警（XGBoost + SHAP）
- [gacha-ltv-churn-predictor](https://github.com/victorzhong0110/gacha-ltv-churn-predictor) — 通用 Gacha LTV ML Pipeline
- [cc-code](https://github.com/victorzhong0110/cc-code) — 本地 AI Agent 编程助手（14 个 LLM Provider）

---

## License

MIT © 2026 · 数据来源 PRTS Wiki（CC BY-NC-SA）
