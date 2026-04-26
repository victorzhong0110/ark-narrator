# ArkNarrator 数据管线 v2 — 设计文档

> 更新日期：2026-04-24

---

## 一、为什么要重做数据管线

第一轮训练（1000 iters，Qwen2.5-7B）暴露了三个根本问题：

| 问题 | 表现 | 根因 |
|------|------|------|
| 输出是第三人称 | 模型描述干员，不扮演干员 | 训练数据全是档案描述（profile_qa） |
| 对话空洞不像任何人 | 台词无角色辨识度 | `dialogue` 类型 output 字段为空，未填充 |
| 出现乱码 | `iaiscommands`、`minefield` 等 | 爬虫噪声混入训练集 |

**结论：旧数据集（1806条 profile_qa）全部废弃，改用完整剧情脚本。**

---

## 二、新数据来源

**来源：** `Kengxxiao/ArknightsGameData`（官方游戏数据，GitHub 公开仓库）

**文件：** `zh_CN/gamedata/story/` 下的 `.txt` 剧情脚本

**路径映射：** `storyInfo "info/X/Y"` → `story/X/Y.txt`

**规模：** 456 章节，1909 个故事节点

**脚本格式：**
```
[name="阿米娅"]  怎么回事......？！为什么，为什么整合运动会......
[name="杜宾"]    不，他们的攻势相当猛烈。这绝对是有预谋的行动。
[Background(...)]   ← 场景标记，解析时保留章节/场景信息
```

---

## 三、三种训练格式

### Format A — 叙事续写（Narrative Continuation）

**目标：** 让模型学会泰拉大陆的叙事风格、世界观语言习惯。

**数据结构：**
```json
{"text": "【黎明前奏·城市之殇】\n阿米娅：怎么回事......？！\n杜宾：他们的攻势相当猛烈。\n..."}
```

**训练方式：** 因果语言建模（causal LM），直接续写。

**适合：** 世界观内容生成、剧情创作、世界观一致性校验。

---

### Format B — 对话窗口（Dialogue Window）

**目标：** 让模型学会在给定上下文下，预测下一句台词（含说话者）。

**数据结构：**
```json
{
  "messages": [
    {"role": "system",    "content": "你是明日方舟叙事生成助手..."},
    {"role": "user",      "content": "【场景：切尔诺伯格】\n前5行对话..."},
    {"role": "assistant", "content": "阿米娅：怎么回事......？！"}
  ]
}
```

**窗口大小：** 6 轮（5 行上文 + 1 行输出）

**适合：** 通用剧情续写、下一句预测。

---

### Format C — 角色扮演（Roleplay）

**目标：** 让模型**成为**某个干员，从该干员视角参与多轮对话。

**核心设计：**
- 以干员 X 为轴心重组对话
- X 的台词 → `assistant` 消息
- 其他所有人的台词（可能多个角色连续说话）→ `user` 消息合并
- `system` 注入角色卡（人格 + 说话风格）

**数据结构：**
```json
{
  "messages": [
    {"role": "system",    "content": "你是明日方舟干员可萝尔。干员简介：温柔坚韧的骑士..."},
    {"role": "user",      "content": "【骑兵与猎人·日正当中】\n赏金猎人：这女人，还不肯说吗？\n？？？：好渴......"},
    {"role": "assistant", "content": "......我不会.....不会告诉你的......"},
    {"role": "user",      "content": "赏金猎人：别给她水喝，直到她说了为止！"},
    {"role": "assistant", "content": "我会死......吗？"},
    ...
  ]
}
```

**为什么这个格式解决了上下文问题：**
- 说话对象、人物关系、情景全部保留在 user 消息里
- 同一干员对不同人、不同情景的反应差异自然体现
- 多轮对话让模型学会保持角色一致性

**适合：** NPC 对话系统、角色声音学习、roleplay 场景。

---

## 四、预期数据量（全量 1909 节点）

| 格式 | 预估样本数 | 覆盖范围 |
|------|-----------|---------|
| A (narrative) | ~2,000 文本块 | 全部剧情 |
| B (dialogue_window) | ~60,000–80,000 条 | 全部剧情 |
| C (roleplay) | ~2,000–4,000 条 | 100+ 干员 |

---

## 五、运行方式

```bash
# 1. 构建全部三种格式的数据集（约 2-3 分钟）
python data_pipeline/dataset_builder.py

# 2. 训练——三种格式全跑
python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml --format all

# 或单独跑某种格式
python finetune/train_mlx.py --config finetune/config/qwen2_5_mlx.yaml --format roleplay
```

每种格式输出到独立 checkpoint 目录：
```
checkpoints/qwen2_5_mlx_narrative/
checkpoints/qwen2_5_mlx_dialogue_window/
checkpoints/qwen2_5_mlx_roleplay/
```

---

## 六、第一轮 vs 第二轮对比

| 维度 | 第一轮 | 第二轮 |
|------|--------|--------|
| 数据来源 | 干员档案（handbook_info） | 完整剧情脚本（1909节点） |
| 数据类型 | 第三人称描述 | 第一人称对话 |
| 样本数 | 1,806 条（单一格式） | ~80,000+ 条（三种格式） |
| 角色上下文 | 无 | 完整保留（说话对象、场景） |
| 角色扮演 | 无 | Format C 专项训练 |
| val loss 最优 | 2.395（iter 400，非最终） | 待测 |
| 推理输出 | 第三人称、语气不对 | 待测 |

---

## 七、待改进项

- [ ] 角色卡目前使用 `character_table` 的简短 description，后续可加入 handbook 档案信息以丰富人格描述
- [ ] `？？？` 等未知说话者占比高，可考虑过滤或单独处理
- [ ] Format C 中连续 assistant 消息较长时可考虑分割
- [ ] 增加 LR cosine decay + warmup（已在 config 中配置，需验证）
- [ ] iter 100 快速推理验证机制（避免重蹈第一轮覆辙）
