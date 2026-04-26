# ArkNarrator 训练日志

**硬件：** Mac mini M 系列，24GB 统一内存  
**框架：** MLX-LM 0.31.3（Apple Silicon 原生）  
**模型：** Qwen2.5-7B-Instruct（4-bit 量化，mlx-community 版本）

---

## 阶段一：数据管道

**目标：** 从 Arknights 公开数据源构建指令微调数据集。

最初计划抓取 PRTS Wiki（泰拉记事社），但 Wiki 页面由 JavaScript 动态渲染，BeautifulSoup 无法解析，返回 0 条数据。

**解决方案：** 改用 Kengxxiao/ArknightsGameData 仓库的原始 JSON 文件（GitHub raw CDN），直接获取 `character_table.json` + `handbook_info_table.json`。最终获取 1159 名干员，其中 415 名有完整档案文本。

构建数据集时发现原始文本含有大量噪音（`【来源】` 引用块、`[链接已失效]` 标注），写了 `clean_output()` 正则清洗函数处理后，最终输出 1806 条训练样本 + 201 条评估样本。

---

## 阶段二：MLX 训练调试

### 问题 1 — API 版本不兼容

mlx-lm 从 0.20 开始废弃了命令行传 `--lora-layers` / `--rank` 的方式，改为通过 `-c config.yaml` 统一传参。直接调用报错：

```
error: unrecognized arguments: --lora-layers 16 --rank 16
```

**解决方案：** 修改 `train_mlx.py`，在运行时动态生成一个临时 YAML 文件，用 `mlx_lm lora -c <tmpfile>` 调用，同时将调用方式从 `python -m mlx_lm.lora`（已废弃）改为 `python -m mlx_lm lora`。

---

### 问题 2 — 显存不足（OOM）

首次训练配置：`batch_size=4`，`max_seq_length=2048`，未开 `grad_checkpoint`。

Val loss（仅 forward pass）正常通过，但第一个训练 step 的 backward pass 触发 Metal OOM：

```
[METAL] Command buffer execution failed: Insufficient Memory
kIOGPUCommandBufferCallbackErrorOutOfMemory
```

原因：训练时需要保留所有层的激活值用于反向传播，内存峰值约为推理的 3-4 倍。

**解决方案：** 三项同步调整：
- `batch_size` 4 → 1
- `max_seq_length` 2048 → 1024（激活值内存减半）
- 开启 `grad_checkpoint: true`（不保存中间激活，用时间换空间）
- `lora rank` 16 → 8，`num_layers` 16 → 8（优化器状态减半）

调整后 Peak mem 稳定在 **5.7 GB**，训练正常启动。

---

### 问题 3 — Loss NaN（梯度爆炸）

训练虽然启动，但 loss 曲线异常：

| Iter | Train Loss |
|------|-----------|
| 10   | 2.513     |
| 30   | 3.053     |
| 50   | 8.824     |
| 120  | **NaN**   |

Loss 不降反升并在 iter 120 之后全部变为 NaN，推理输出退化为全 `!` 符号。

**根因分析：** 学习率 `2e-4` 对 rank=8 的 LoRA 而言过高，warmup 期（仅 50 步）内梯度未稳定就超出数值范围，导致权重污染。

**解决方案：**
- `learning_rate` 2e-4 → **5e-5**（降低 4 倍）
- `warmup` 50 → **100** 步
- 新增 `grad_clip: 1.0`（梯度裁剪，防止梯度范数超限）
- 清空损坏的 checkpoint，重新训练

---

## 当前超参配置（稳定版）

```yaml
model:    Qwen2.5-7B-Instruct-4bit
lora:     rank=8, scale=16, dropout=0.05, num_layers=8
lr:       5e-5，cosine decay，warmup=100
batch:    1，max_seq_length=1024
grad:     checkpoint=true，clip=1.0
iters:    1000
```

---

## 硬件对比备注（MLX vs CUDA）

| 指标 | Mac mini MLX | Kaggle A100（CUDA） |
|------|-------------|-------------------|
| 峰值显存 | 5.7 GB 统一内存 | 待记录 |
| 训练速度 | ~0.35 it/sec，~116 tok/sec | 待记录 |
| 量化方案 | 4-bit（mlx-community） | 4-bit NF4（bitsandbytes） |
| OOM 特点 | Metal buffer OOM，无 CUDA fallback | 显存独立，更易调试 |
| 适用场景 | 本地验证、快速迭代 | 正式训练、大 batch |
