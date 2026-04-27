# Gemma 4 E4B × mlx-lm 兼容性问题记录

> 日期：2026-04-27
> 结论：`mlx-community/gemma-4-E4B-it-4bit` 在 mlx-lm 0.31.3 纯文本训练/推理下行为异常，不可用于 ArkNarrator 对比实验。

---

## 症状

### 1. 基础模型推理输出乱码

即使用正确的 Gemma 4 chat template，`mlx_lm generate` 也输出多语言混杂乱码，英文也同样：

```
# 输入：你好，能天使！最近任务怎么样？
# 输出（with chat template）：
哦róantashing***x年以上 midline\_မဟုတ်lojah poiيد चाहेंれているوسي的状态...
```

### 2. 初始 Val loss 远超随机猜测

| 模型 | Iter 1 Val loss | 随机基线（ln 256K） | 分析 |
|------|----------------|-------------------|------|
| Qwen3-8B | 3.478 | ~12.4 | 正常，预训练模型有语言知识 |
| Gemma 4 E4B R2 | **18.801** | ~12.4 | 异常：**高于随机猜测**，说明前向传播本身有问题 |

### 3. 训练后输出仍破碎

1500 iters 后最佳 val loss 4.033，quick inference 输出：
```
在喷？ 芙兰德醒！ 开。 罗。 喧呢？ 临去！
```

---

## 根本原因

`mlx-community/gemma-4-E4B-it-4bit` 的架构为 **`Gemma4ForConditionalGeneration`**（多模态），包含视觉和音频编码器：

```json
{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "audio_config": { ... },
  "image_token_id": 258880,
  "audio_token_id": 258881
}
```

mlx-lm 0.31.3 以纯文本（causal LM）方式加载此多模态模型时，前向传播行为不确定，导致：
- loss 高于随机猜测（模型"反向预测"）
- 生成阶段输出乱码

对比：`gemma-4-26B-A4B-it-4bit`（MoE 文本模型）推理正常，但 14.5GB 推理内存 → 训练时 OOM。

---

## 第一轮训练数据的说明

之前记录的 Gemma 4 E4B 第一轮 val loss 3.987（iter 1200，max_seq_length=2048）：
- 该结果来源于更早的会话，具体训练过程未验证
- 与本轮 R2 训练结果（18.8 初始 loss）矛盾
- **不应将 3.987 作为可信参考数据**

---

## 解决方案：改用 Qwen2.5 vs Qwen3 对比（方案 A）

放弃 Gemma 4 作为对比模型，改为同数据集、不同代次的 Qwen 系列对比：

| 模型 | 轮次 | 最佳 Val loss | checkpoint |
|------|------|-------------|-----------|
| Qwen2.5-7B-Instruct-4bit | 第四轮（roleplay）| **2.156**（iter ~1200） | `checkpoints/qwen2_5_mlx_roleplay/` |
| Qwen3-8B-4bit | 第一轮（roleplay）| **2.345**（iter 1400） | `checkpoints/qwen3_5_mlx_roleplay_roleplay/` |

对比价值：同任务、同训练流程、不同代次模型，直接反映架构迭代带来的效果差异。

---

## 后续方向（如需 Gemma 对比）

- **云方案**：Kaggle A100（40GB），用 Gemma 4 26B A4B 跑 4096 seq_len 训练
- **前提**：需确认 mlx-lm 或 HuggingFace Transformers + PEFT 在 A100 上正确支持 Gemma 4 26B 的文本模式
- 优先级低，作为中期目标

---

*记录于 2026-04-27*
