<div align="center">

# ArkNarrator

**明日方舟 × LLM 干员对话生成系统**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX--LM-0.31.3-8B5CF6?logo=apple)](https://github.com/ml-explore/mlx-lm)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> 基于 PRTS Wiki 全量剧情语料，对 **Qwen2.5-7B** 与 **Qwen3-8B** 进行 LoRA 微调，
> 构建面向游戏策划的干员角色扮演生成系统，并通过 **DeepSeek V4 Pro Judge** 进行系统性双模型对比评估。

---

## 实验结果摘要

### 训练结果

| | Qwen2.5-7B（第四轮） | Qwen3-8B（第一轮） |
|---|---|---|
| 基础模型 | `mlx-community/Qwen2.5-7B-Instruct-4bit` | `mlx-community/Qwen3-8B-4bit` |
| 参数量 | 7B | 8B |
| 训练框架 | MLX-LM 0.31.3 LoRA | MLX-LM 0.31.3 LoRA |
| max\_seq\_length | 4096 | 4096 |
| 训练 iters | 1500（roleplay format） | 1500（roleplay format） |
| **最优 val loss** | **2.156**（iter 1200） | **2.345**（iter 1400） |
| 峰值显存 | 8.3 GB / 24 GB | 8.3 GB / 24 GB |

### DeepSeek V4 Pro 评估结果（2026-04-27）

| 模型 | 归因准确率 | 矛盾通过率 | DS Pairwise |
|------|-----------|-----------|-------------|
| Qwen2.5-7B R4 | 5 / 6 | 6 / 6 | 2 胜 |
| **Qwen3-8B R1** | **5 / 6** | **6 / 6** | **3 胜 1 平** |

**核心发现**：Qwen3-8B 以第一轮训练（val loss 2.345）在 pairwise 对比中小胜 Qwen2.5-7B 第四轮（val loss 2.156），说明底座模型迭代带来的收益超过了额外训练轮次。两个模型均在 p5（阿米娅）出现误判，属于共同弱点，与阿米娅风格接近凯尔希的语料分布有关。

---

## 核心特性

- **三格式数据集构建**：Narrative 叙事续写 / Dialogue Window 对话窗口 / Roleplay 角色扮演
- **194 干员角色卡**：基于 handbook\_info 档案自动生成系统提示，含档案资料与测试记录
- **DeepSeek V4 Pro Judge 评估框架**：归因测试 / 矛盾检测 / Pairwise Win-rate，支持人工评估模式
- **FastAPI + SSE 推理服务**：多角色支持、多轮对话、流式 token 输出 + 内嵌 Demo UI
- **Qwen3 thinking 模式处理**：自动过滤 `<think>...</think>` 推理块，仅保留最终回复

---

## 项目结构

```
ArkNarrator/
├── data_pipeline/
│   ├── scraper.py               # PRTS Wiki 爬虫（1904 故事节点）
│   └── dataset_builder.py       # 三格式数据集构建 + P9 噪声过滤
├── finetune/
│   ├── train_mlx.py             # MLX 训练入口（cosine LR + best checkpoint 自动保存）
│   └── config/
│       ├── qwen2_5_mlx.yaml     # Qwen2.5-7B 第四轮配置
│       └── qwen3_5_mlx.yaml     # Qwen3-8B 第一轮配置
├── inference/
│   ├── engine.py                # MLX 推理引擎（多角色 + 流式）
│   ├── server.py                # FastAPI 服务（/chat /stream /）
│   └── test_roleplay.py         # 命令行快速推理测试
├── eval/
│   ├── judge.py                 # DeepSeek V4 Pro Judge 评估流水线
│   └── results/                 # 评估结果 JSON + Markdown 报告
└── checkpoints/
    ├── qwen2_5_mlx_roleplay/    # Qwen2.5 LoRA 适配器（最佳 val loss 2.156）
    └── qwen3_5_mlx_roleplay_roleplay/  # Qwen3 LoRA 适配器（最佳 val loss 2.345）
```

---

## 快速开始

### 环境准备

```bash
git clone https://github.com/victorzhong0110/ark-narrator.git
cd ark-narrator
pip install -r requirements.txt
```

### 推理

```bash
# 命令行测试（默认 Qwen2.5 适配器）
python inference/test_roleplay.py --model qwen

# 启动 Web Demo
uvicorn inference.server:app --host 0.0.0.0 --port 8000
# 访问 http://localhost:8000

# 切换到 Qwen3 适配器
MODEL_KEY=qwen3 uvicorn inference.server:app --port 8000
```

### 训练

```bash
# Qwen2.5-7B（Apple Silicon，24GB 统一内存）
python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml --format roleplay

# Qwen3-8B（同配置，峰值约 8.3GB）
python finetune/train_mlx.py --config finetune/config/qwen3_5_mlx.yaml --format roleplay
```

### 评估

```bash
# 完整流水线：生成 + DeepSeek V4 Pro 评估
DEEPSEEK_API_KEY=sk-... python eval/judge.py

# 跳过生成，对已有输出重新评估
DEEPSEEK_API_KEY=sk-... python eval/judge.py --skip-generate

# 加入人工 pairwise 评估
DEEPSEEK_API_KEY=sk-... python eval/judge.py --human
```

---

## 评估方法

DeepSeek V4 Pro 作为 Judge，三个客观指标：

| 指标 | 方法 | 说明 |
|------|------|------|
| **归因准确率** | 三选一闭卷测试 | Judge 只看输出，不看角色名，从三张角色卡中选最匹配的 |
| **矛盾通过率** | 二分类检测 | Judge 对比输出与角色卡，判断是否存在性格/世界观矛盾 |
| **Pairwise Win-rate** | 随机 A/B 比较 | 同 prompt 两模型输出，Judge 选更符合角色设定的一方 |

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
  -d '{"character":"能天使","message":"最近有没有遇到难搞的任务？","history":[]}'
```

---

## 已知问题

| 问题 | 根本原因 | 状态 |
|------|---------|------|
| 阿米娅归因误判（两模型共同） | 阿米娅语料中含大量非个人化叙事，与凯尔希风格边界模糊 | 待扩充 eval 样本验证 |
| Qwen3 偶发 thinking 模式 | Qwen3 chat template 在某些 prompt 下触发 `<think>` | 已在 judge.py 自动过滤；推理端需 `/no_think` system prompt |
| 能天使偶发幻觉地名 | 训练数据含非规范化地名标注（pattern bleed-through） | 未修复，系统提示缓解中 |

---

## 踩坑记录

### Qwen3.5 架构不兼容 mlx-lm 0.31.3

Qwen3.5（2026年4月发布）是 SSM+Transformer 混合架构（Mamba-style linear attention），mlx-lm 0.31.3 不支持其 SSM 层的 backward pass，训练必 OOM。Qwen3（2025年4月发布）是纯 Transformer，兼容正常。

```yaml
# Qwen3.5 config（不兼容）
model_type: qwen3_5
layer_types: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', ...]

# Qwen3 config（兼容）
model_type: qwen3
```

### Gemma 4 E4B 多模态架构问题

`mlx-community/gemma-4-E4B-it-4bit` 的架构为 `Gemma4ForConditionalGeneration`（多模态），在 mlx-lm 纯文本模式下前向传播异常：

- 基础模型输出乱码（英文也不行）
- 初始 val loss 18.8 > 随机基线 ln(256K) ≈ 12.4
- 详见 [`readmeplz/gemma4_mlx_compatibility_issue.md`](readmeplz/gemma4_mlx_compatibility_issue.md)

---

## 相关项目

- [Arknights-Gacha-Analysis-System](https://github.com/victorzhong0110/Arknights-Gacha-Analysis-System) — 玩家流失预警（XGBoost + SHAP）
- [gacha-ltv-churn-predictor](https://github.com/victorzhong0110/gacha-ltv-churn-predictor) — 通用 Gacha LTV Pipeline

---

## License

MIT © 2026 · 数据来源 PRTS Wiki（CC BY-NC-SA）
