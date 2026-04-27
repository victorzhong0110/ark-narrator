# 会话记录：2026-04-27

> Victor（仲旭东）× 星羽
> 主题：公平对比实验 + Eval 更新 + 踩坑记录

---

## 本次完成的事项

### 1. Qwen3-8B 训练成功

- 模型：`mlx-community/Qwen3-8B-4bit`（纯 Transformer，2025年4月）
- 配置：max\_seq\_length=4096，rank=8，lr=2e-5，iters=1500
- 峰值内存：**8.3 GB / 24 GB**（非常宽裕）
- 最优 val loss：**2.345**（iter 1400）
- checkpoint：`checkpoints/qwen3_5_mlx_roleplay_roleplay/`

### 2. 评估框架更新：Gemma → Qwen3

原比较对象（Gemma 4 E4B）发现架构不兼容，改为 **Qwen2.5-7B R4 vs Qwen3-8B R1**（同数据集、不同代次模型对比）。

### 3. 评估结果（DeepSeek V4 Pro Judge）

| 模型 | 归因 | 矛盾通过 | Pairwise |
|------|------|---------|---------|
| Qwen2.5-7B R4 | 5/6 | 6/6 | 2胜 |
| **Qwen3-8B R1** | **5/6** | **6/6** | **3胜 1平** |

- p5（阿米娅）两边都被误判为凯尔希，属于共同弱点
- Qwen3 底座强于 Qwen2.5，即便只训练了一轮也胜出

---

## 本次遇到的坑

### 坑一：Qwen3.5 是 SSM 混合架构，mlx-lm 不支持训练

**现象**：Qwen3.5-9B 和 Qwen3.5-4B 在 max\_seq\_length=4096 下全部 OOM，换 3072 也 OOM。

**真正原因**：Qwen3.5（2026年4月发布）是 SSM+Transformer 混合架构（Mamba-style），`model_type: qwen3_5`，有 `linear_attention` 层和 `mamba_ssm_dtype: float32`，mlx-lm 0.31.3 的 backward pass 不支持 SSM 层，不是内存不够，是架构根本不兼容。

**解决方案**：改用 Qwen3（2025年4月，纯 Transformer），`mlx-community/Qwen3-8B-4bit` 存在且兼容。

---

### 坑二：Gemma 4 E4B 是多模态模型，文本推理输出乱码

**现象**：
- 训练 R2（seq\_len=4096）初始 val loss = 18.8（高于随机猜测基线 12.4）
- 基础模型推理即使用正确 chat template 也输出多语言混杂乱码
- 英文 prompt 也无效

**真正原因**：`mlx-community/gemma-4-E4B-it-4bit` 架构为 `Gemma4ForConditionalGeneration`，是多模态模型，包含视觉和音频编码器。mlx-lm 以纯文本方式加载时前向传播行为不确定。

**Gemma 4 R1 的 val loss 3.987 是否可信**：存疑。当时用 max\_seq\_length=2048 截断了大量样本，实际上也可能是同样的多模态问题，只是数值上看起来"还行"。

**解决方案**：放弃 Gemma 4 E4B，改用 Qwen3-8B 作对比模型（方案 A）。

---

### 坑三：head -60 管道截断杀死训练进程

**现象**：第一次启动 Qwen3 训练时加了 `| head -60`，60行输出后 head 关闭管道，Python 进程收到 SIGPIPE，训练在 Iter 200 时被终止。

**解决方案**：重新启动，输出重定向到文件 `> /tmp/qwen3_train.log 2>&1`。

---

### 坑四：Qwen3 可能触发 thinking 模式

**现象**：quick inference test 输出带有 `<think>\n嗯，用户直接喊我"天使"...</think>` 块。

**原因**：Qwen3 支持 thinking/non-thinking 双模式，某些 prompt 会触发推理链。训练数据没有 `<think>` token，但推理时可能仍然激活。

**解决方案**：`eval/judge.py` 中已加 `strip_think()` 自动过滤；推理端建议在 system prompt 加 `/no_think` 或 `请直接回答，不需要思考过程`。

---

## 关于 readmeplz 文件夹

本次会话中读取了星羽整理的 eval 优化建议（`eval_optimization_discussion.md`），主要建议：
1. 扩展 eval 样本从 6 → 20-30 条
2. 按性格分组测归因（凯尔希/赛雷娅/华法琳 同组，归因难度更高）
3. 新增多轮对话一致性测试（3-5 轮历史）
4. 矛盾检测从二分类升级为五类标签

这些建议暂未执行，列为中期任务。

---

## 项目当前状态

| 组件 | 状态 |
|------|------|
| Qwen2.5-7B R4 训练 | 完成，val loss 2.156 |
| Qwen3-8B R1 训练 | 完成，val loss 2.345 |
| eval/judge.py | 更新为 Qwen2.5 vs Qwen3，含 think 过滤 |
| eval 结果 | Qwen3 pairwise 3-2-1 胜出 |
| README | 已更新（反映最新结果 + 踩坑记录） |
| FastAPI 推理服务 | 已完成（上一个会话） |
| Gemma 4 对比 | 放弃（架构不兼容），后续云端可做 |

---

## 下一步（中期）

- [ ] 扩展 eval 样本到 20 条（参考星羽建议）
- [ ] 按性格分组做归因测试
- [ ] Qwen3-8B R2：在 R1 基础上继续训练，看能否低于 2.156
- [ ] 云端方案：Kaggle A100 跑 Qwen3-35B 或 Gemma 4 26B MoE

---

*记录于 2026-04-27*
