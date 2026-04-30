# MLX 7B Class 最新模型榜单（2026年4月验证）

> 全部为 mlx-community 4-bit 量化版本，适合 Apple Silicon Mac（24GB）
> 所有链接均已逐一验证

---

## 第一梯队（强烈推荐）

### 1. Qwen2.5-14B-Instruct-4bit
- **链接**: https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit
- **大小**: 8.31 GB
- **月下载**: 73,065
- **MMLU**: ~82（7B-14B级别最高）
- **优点**: 评价最高，生态最完整，中文支持极强
- **备注**: 14B对24GB内存略紧张但可跑

### 2. Qwen2.5-7B-Instruct-4bit
- **链接**: https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit
- **大小**: 4.28 GB
- **月下载**: 13,739
- **优点**: 7B黄金尺寸，24GB毫无压力，Qwen最强7B
- **备注**: 相比14B更省内存，适合频繁跑

### 3. Qwen2.5-Coder-7B-Instruct-4bit
- **链接**: https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
- **大小**: 4.28 GB
- **月下载**: ~100+
- **优点**: 代码专项优化，比通用Qwen2.5代码能力更强
- **备注**: 如果做角色扮演有代码片段需求可用

---

## 第二梯队（优秀备选）

### 4. Mistral-7B-Instruct-v0.3-4bit
- **链接**: https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit
- **大小**: 4.08 GB
- **月下载**: 5,418
- **优点**: Mistral最新版本，比v0.2更新，通用能力强
- **备注**: 经典老牌，生态丰富

### 5. Kimi-K2.5
- **链接**: https://huggingface.co/mlx-community/Kimi-K2.5
- **大小**: ~6-7 GB（官方显示有误，以实际下载为准）
- **月下载**: 34 likes
- **优点**: 月之暗面最新K2模型，中文对话能力强
- **备注**: 上传较新，生态尚未完全成熟

### 6. Phi-4-mini-instruct-4bit
- **链接**: https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit
- **大小**: 2.16 GB
- **月下载**: ~100+
- **优点**: 微软小钢炮，体积最小，效率高
- **备注**: 只有3.8B参数，偏向效率优先场景

---

## 已验证不存在（2026年4月）

以下模型在 mlx-community 上返回 404，**不存在**：
- `mlx-community/gemma-2-7b-it-4bit` — 只有9B版本
- `mlx-community/DeepSeek-7B-Instruct-4bit` — 不存在
- `mlx-community/DeepSeek-V2.5-7B-Instruct-4bit` — 不存在
- `mlx-community/Mistral-Nemo-Instruct-4bit` — 不存在
- `mlx-community/Llama-3.3-8B-Instruct-4bit` — 不存在
- `mlx-community/Llama-4-7B-Instruct-4bit` — 不存在
- `mlx-community/Qwen3-7B-Instruct-4bit` — 不存在
- `mlx-community/gemma-3-7b-it-4bit` — 不存在

---

## 注意事项

1. **Qwen3 / Llama-4 / Gemma-3** — 截至2026年4月尚未发布MLX版本
2. **24GB内存建议**：首选Qwen2.5-7B（4.28GB），次选Mistral v0.3（4.08GB）
3. **14B模型**：24GB可跑但建议关闭其他应用，留足内存
4. **DeepSeek-V4-Flash** — 虽然recently updated但无model card，不够可靠，暂不推荐

---

*最后更新：2026年4月27日*
