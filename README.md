<div align="center">

# ⚔️ ArkNarrator

**明日方舟 × LLM 干员对话生成系统**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX--LM-0.31.3-8B5CF6?logo=apple)](https://github.com/ml-explore/mlx-lm)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> 基于 PRTS Wiki 全量剧情语料，对 **Qwen2.5-7B** 与 **Gemma 4 E4B** 进行 LoRA 微调，  
> 构建面向游戏策划的干员角色扮演生成系统，并通过 GPT-4o Judge 进行系统性双模型对比评估。

---

## 实验结果摘要

| | Qwen2.5-7B（第四轮） | Gemma 4 E4B（第一轮） |
|---|---|---|
| 基础模型 | `mlx-community/Qwen2.5-7B-Instruct-4bit` | `mlx-community/gemma-4-E4B-it-4bit` |
| 参数量 | 7B | 4B（MoE效率优化架构） |
| 训练框架 | MLX-LM LoRA | MLX-LM LoRA + 架构 patch |
| max_seq_length | 4096 | 2048（24GB 内存约束） |
| 训练 iters | 1500 | 1500 |
| **最优 val loss** | **2.156**（iter 1200） | **3.987**（iter 1200） |
| 角色扮演质量 | ⚠️～✅✅ 多角色稳定 | ❌ 输出破碎（截断所致） |

> Gemma 的高 val loss 主要源于 `max_seq_length=2048` 对 roleplay 样本（最长 ~3800 tokens）的强截断，而非模型能力不足。在等内存条件下对比不公平，A100 环境下结论可能不同。

---

## 核心特性

- **三格式数据集构建**：Narrative 叙事续写 / Dialogue Window 对话窗口 / Roleplay 角色扮演
- **194 干员角色卡**：基于 handbook_info 档案自动生成系统提示，含档案资料与测试记录
- **mlx-lm Gemma4 架构修复**：一键 patch 脚本修复 0.31.3 中 126 个缺失参数问题
- **GPT-4o Judge 评估框架**：3 维度（世界观/角色一致/流畅度）+ role_break 检测，支持规则降级
- **FastAPI + SSE 推理服务**：多角色支持、多轮对话、流式 token 输出 + 内嵌 Demo UI

---

## 项目结构

```
ArkNarrator/
├── data_pipeline/
│   ├── scraper.py           # PRTS Wiki 爬虫（1904 故事节点）
│   └── dataset_builder.py   # 三格式数据集构建 + P9 噪声过滤
├── finetune/
│   ├── train_mlx.py         # MLX 训练入口（含 cosine LR + best checkpoint）
│   └── config/
│       ├── qwen2_5_mlx.yaml      # Qwen 第四轮配置
│       └── gemma4_mlx.yaml       # Gemma E4B 配置
├── inference/
│   ├── engine.py            # MLX 推理引擎（多角色 + 流式）
│   ├── server.py            # FastAPI 服务（/chat /stream /demo）
│   └── test_roleplay.py     # 命令行快速推理测试
├── eval/
│   ├── gpt4o_judge.py       # GPT-4o Judge 双模型评估流水线
│   └── results/             # 评估结果 JSON + Markdown 报告
├── patches/
│   └── apply_mlx_patch.py   # 修复 mlx-lm 0.31.3 Gemma4 架构缺陷
└── checkpoints/
    ├── qwen2_5_mlx_roleplay/    # Qwen LoRA 适配器
    └── gemma4_mlx_roleplay/     # Gemma LoRA 适配器
```

---

## 快速开始

### 环境准备

```bash
git clone https://github.com/victorzhong0110/ark-narrator.git
cd ark-narrator
pip install -r requirements.txt
```

### 使用已有适配器推理（跳过训练）

适配器文件较大未纳入仓库，需自行训练或联系作者获取。

```bash
# 命令行多角色推理测试
python inference/test_roleplay.py --model qwen

# 启动 Web Demo（默认 Qwen 适配器）
uvicorn inference.server:app --host 0.0.0.0 --port 8000
# 访问 http://localhost:8000 打开 Demo UI

# 使用 Gemma 适配器
MODEL_KEY=gemma uvicorn inference.server:app --port 8000
```

### 从头构建数据集

```bash
# 1. 爬取 PRTS Wiki 剧情脚本
python data_pipeline/scraper.py

# 2. 构建三格式数据集（需要 data/raw/character_table.json + handbook_info.json）
python data_pipeline/dataset_builder.py
# 输出：data/processed/roleplay_train.jsonl + roleplay_eval.jsonl 等
```

### 训练

```bash
# Qwen2.5-7B MLX LoRA（Apple Silicon 24GB）
python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml

# Gemma 4 E4B — 先打 patch
python patches/apply_mlx_patch.py
python finetune/train_mlx.py --config finetune/config/gemma4_mlx.yaml
```

---

## mlx-lm Gemma4 架构修复

mlx-lm 0.31.3 将 `num_kv_shared_layers=18` 误解为"最后 18 层复用前序层的 KV 向量"，导致加载 E4B checkpoint 时报错 126 个缺失参数。实际上所有 42 层均有独立权重。

```bash
# 一键修复（幂等，可重复运行）
python patches/apply_mlx_patch.py
```

修复内容：
1. `Attention.__init__`：`has_kv = True`（对所有层）
2. `Gemma4TextModel.__init__`：移除跨层 KV 复用逻辑

---

## GPT-4o 评估

```bash
# 生成输出 + 规则评分（无需 API Key）
python eval/gpt4o_judge.py --no-gpt

# 完整 GPT-4o 评分
export OPENAI_API_KEY=sk-...
python eval/gpt4o_judge.py

# 跳过生成，对已有输出重新评分
python eval/gpt4o_judge.py --skip-generate
```

评分维度：`world_fidelity`（世界观）/ `char_consistency`（角色一致）/ `fluency`（流畅度）/ `role_break`（破角色检测）

---

## API 接口

### `POST /chat`（非流式）

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"character":"凯尔希","message":"请评估本次行动的医疗损耗。","history":[]}'
```

### `POST /stream`（SSE 流式）

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"character":"能天使","message":"你好，最近有没有遇到难搞的任务？","history":[]}'
```

### `GET /characters`

```json
{"characters": ["能天使", "凯尔希", "阿米娅", "陈", "德克萨斯"]}
```

---

## 已知限制

| 问题 | 原因 | 状态 |
|------|------|------|
| Gemma 输出语义破碎 | `max_seq_length=2048` 截断 roleplay 样本（最长 3800t） | 硬件约束，待 A100 验证 |
| 能天使偶发幻觉地名 | P10：训练数据含非规范化地名标注 | 未修复，系统提示缓解中 |
| 角色知识截止 | 语料仅覆盖至爬取时的版本 | 定期重建数据集 |

---

## 相关项目

- [Arknights-Gacha-Analysis-System](https://github.com/victorzhong0110/Arknights-Gacha-Analysis-System) — 玩家流失预警（XGBoost + SHAP）
- [gacha-ltv-churn-predictor](https://github.com/victorzhong0110/gacha-ltv-churn-predictor) — 通用 Gacha LTV Pipeline

---

## License

MIT © 2026 · 数据来源 PRTS Wiki（CC BY-NC-SA）
