# LLM 量化评估指标全景图

> 整理：星羽
> 目标：覆盖市面上主流的 LLM 评估指标，按场景分类，适合 Victor 的 ArkNarrator 项目参考

---

## 一、传统文本生成指标（经典 NLP）

### 1.1 基于 N-gram 重叠的指标

| 指标 | 全称 | 适用场景 | 局限性 |
|------|------|---------|--------|
| **BLEU** | Bilingual Evaluation Understudy | 机器翻译、摘要 | 对同义词/改写不敏感，无法捕捉语义 |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | 摘要、文本生成 | 偏向召回，忽略流畅度和语义 |
| **METEOR** | Metric for Evaluation of Translation with Explicit ORdering | 机器翻译 | 与人类判断相关性有限 |
| **chrF** | Character n-gram F-score | 翻译，尤其多语言 | 无法评估语义质量 |

> ⚠️ **重要提醒**：BLEU/ROUGE 与人类判断相关性普遍偏低（尤其 open-ended 生成任务），学术界已逐渐放弃作为主要指标，但工程验证中仍有参考价值。

### 1.2 基于 Embedding 的指标

| 指标 | 核心思想 | 优点 | 适用场景 |
|------|---------|------|---------|
| **BERTScore** | 用 BERT 语义向量计算 token 级别相似度 | 捕捉语义，不受同义词限制 | 文本生成质量评估 |
| **BARTScore** | 用 BART 计算 Source→Hyp 的生成质量 | 端到端评估，语义+流畅度 | 摘要、翻译 |
| **UniEval** | 基于 T5 的统一评估框架 | 多维度可扩展 | 对话、摘要、事实一致性 |
| **GPTEval** | 用 GPT-3/4 打分 | 接近人类判断 | 开放式生成评估 |

---

## 二、学术标准 Benchmark（封闭问答）

### 2.1 综合理解与推理

| Benchmark | 全称 | 规模 | 核心能力 | 备注 |
|-----------|------|------|---------|------|
| **MMLU** | Massive Multitask Language Understanding | 57 个学科，15,908 题 | 知识 + 推理 | 业界标准基线，5-shot 测试 |
| **BIG-Bench** | Beyond the Imitation Game Benchmark | 200+ 任务 | 多维度综合 | 包含编程/道德/推理等 |
| **HellaSwag** | Hard Language Swag | 70k 题 | 常识推理 | 人类通过率 ~95%，SOTA ~89% |
| **ARC** | AI2 Reasoning Challenge | 7,787 题 | 复杂推理 | 科学问题，难度较高 |
| **WinoGrande** | Winograd Schema Grande | 44k 题 | 常识推理，歧义消解 | 灵感来自 Winograd Schema |
| **Drop** | Discrete Reasoning Over Paragraphs | 96k 问答对 | 阅读理解 + 离散推理 | 需要多步算术/计数 |

### 2.2 数学与逻辑推理

| Benchmark | 规模 | 特点 |
|-----------|------|------|
| **GSM8K** | 8,500 题（小学数学） | 需要多步推理，GPT-4 达 92% |
| **MATH** | 12,500 题（竞赛数学） | 高中/大学难度，LLM 仍有挑战 |
| **MMLU-Math** | MMLU 数学子集 | 纯数学推理 |
| **TheoremQA** | 800 题 | 数学+定理应用 |
| **GAOKAO-Bench** | 中国高考题 | 中文基准，适合国产模型 |

### 2.3 编程与代码生成

| Benchmark | 全称 | 核心能力 |
|-----------|------|---------|
| **HumanEval** | HumanEval | Python 代码补全，164 题 |
| **MBPP** | Mostly Basic Python Problems | 974 题，基础编程 |
| **BigCodeBench** | BigCode Benchmark | 1,140 题，真实代码场景 |
| **MultiPL-E** | Multi-Programming Language Eval | 跨语言代码生成 |
| **APPS** | Automated Programming Progress Standard | 5,000 题，竞赛难度 |
| **DS-1000** | Data Science 1000 | Pandas/Matplotlib 等数据分析 |

---

## 三、对话与指令遵循评估

### 3.1 对话质量 Benchmark

| Benchmark | 评估方式 | 特点 |
|-----------|---------|------|
| **MT-Bench** | Multi-turn Questions，100 题 | 双轮对话，8 大类目，LLM-as-Judge |
| **Chatbot Arena** (LMSYS) | 人类 ELO 投票 | 真实人类偏好，Top 模型排行 |
| **AlpacaEval** | LLM-as-Judge，单轮指令 | 14 指标，快速自动化 |
| **FLASK** | Fine-grained LLM Assessment | 12 维度细粒度评分 |
| **LF-COMPASS** | 中文对话评估 | 中文场景，针对性更强 |

### 3.2 指令遵循

| Benchmark | 全称 | 说明 |
|-----------|------|------|
| **IFEval** | Instruction Following Eval | 25 条可验证指令规则，精确度评估 |
| **PromptBench** | Prompt-level Adversarial | 对抗性 prompt 鲁棒性 |
| **Big-R** | Reasoning benchmark | 推理类指令遵循 |

### 3.3 对齐与偏好

| Benchmark | 说明 |
|-----------|------|
| **HH-RLHF** | Anthropic 的_helpful vs _harmless 偏好数据集 |
| **SHP** | Stanford Human Preferences，68k 偏好对 |
| **PKU-Alignment/PKU** | 北京大学对齐数据 |

---

## 四、LLM-as-Judge 评估框架

这是目前最主流的**开放式生成任务**评估方法，尤其适合角色扮演类任务。

### 4.1 核心方法

**A. Pairwise Win-Rate（成对比较）**
```
给模型相同输入 X，收集两个模型的输出 A 和 B
让 Judge（LLM 或人类）判断哪个更好
统计胜率：WinRate(A) = wins(A) / (wins(A) + wins(B) + ties)
```

**B. Rating Scale（打分制）**
```
定义 1-5 或 1-10 量表
每个维度单独打分：流畅度、角色一致性、世界观一致性等
最终汇报平均分或各维度分数
```

**C. Multi-Dimensional G-Eval**
```python
# G-Eval 思路（来自微软）
1. 定义评估维度（角色一致性、流畅度等）
2. 用 CoT 让 LLM 生成评分步骤
3. 计算最终得分的期望
```

### 4.2 主流 Judge 模型

| 模型 | 特点 | 适合场景 |
|------|------|---------|
| **GPT-4o** | 最常用，效果稳定 | 通用判断，有 API |
| **Claude-3.5** | 推理能力强，判断细致 | 复杂推理判断 |
| **DeepSeek V4 Pro** | 成本低，效果接近 GPT-4 | 成本敏感场景（Victor 正在用） |
| **Qwen2.5-72B** | 开源可本地部署 | 无 API 成本顾虑 |
| **人类 Judge** |Ground Truth | 小规模精细评估 |

### 4.3 Pairwise 评估注意事项

1. **A/B 顺序随机化** — 防止位置偏好
2. **Judge 模型温度 ≤ 0.2** — 保证判断一致性
3. **加入 Tie 选项** — 避免强制二选一
4. **多 Judge 交叉验证** — 不同模型轮流做 Judge

---

## 五、角色扮演 / 人格一致性评估

这是最直接跟 ArkNarrator 相关的方向，目前业界专项 benchmark 相对较少，但有不少值得参考的思路。

### 5.1 现有相关 Benchmark

| 名称 | 说明 |
|------|------|
| **SARA** | Self-Aware Roleplay Assessment，测角色一致性 |
| **CharacterBench** | 角色扮演一致性基准（新兴） |
| **Persona Consistency Eval** | 多轮对话中人格一致性评估 |
| **ToMi** | Theory of Mind，测角色心理建模能力 |

### 5.2 可借鉴的评估维度（适合 ArkNarrator）

| 维度 | 测试方式 | 说明 |
|------|---------|------|
| **角色归因** | 三选一/五选一，看输出能否被正确识别 | 类似 Victor 现有的 attribution test |
| **性格一致性** | 对抗性 prompt（如让沉稳角色说笑话）| 检测破功风险 |
| **说话风格** | 词汇/句式分析 | 专业术语、口头禅、语体 |
| **世界观一致性** | 是否出现常识错误/时代错乱 | 明日方舟世界观知识 |
| **角色关系一致性** | 对不同对象的态度是否符合关系 | 如凯尔希对阿米娅的态度 |
| **多轮记忆一致性** | 5 轮对话后的事实一致性 | 是否"忘记"之前的承诺 |
| **情感弧线一致性** | 随对话推进情绪变化是否合理 | 悲伤→被安慰→释然 |

### 5.3 角色扮演专项测试设计建议

```
测试分组：
├── 角色归因组（闭卷）：只给角色卡，不给对话历史
├── 角色归因组（开卷）：给角色卡 + 3 轮对话历史
├── 对抗性 prompt 组：故意挖坑（角色知识盲区、性格冲突场景）
├── 多轮记忆组：5 轮历史后问新问题
└── 跨角色区分组：两个相似角色（赛雷娅 vs 凯尔希）同一场景
```

---

## 六、特定领域 Benchmark

### 6.1 中文 / 多语言

| Benchmark | 说明 |
|-----------|------|
| **CMMLU** | 中文多任务语言理解，115 大学科 |
| **C-Eval** | 中文基础模型评估，1.3 万题 |
| **SuperCLUE** | 中文通用能力基准 |
| **MOSS** | 中文对话、工具使用、插件调用 |
| **AlignBench** | 中文对齐评估，8 大维度 |

### 6.2 医疗 / 法律 / 金融

| 领域 | Benchmark |
|------|-----------|
| 医疗 | **MedQA**（USMLE）、**PubMedQA**、**MedicalBench** |
| 法律 | **LegalBench**、**CaseHOLD** |
| 金融 | **FinanceBench**、**BBQ-Finance** |
| 科学 | **ScienceQA**、**SciEval** |

---

## 七、模型对比平台（可直接使用）

| 平台 | 说明 |
|------|------|
| **Chatbot Arena** (chat.lmsys.org) | 人类 ELO 排名，最权威 |
| **OpenCompass** (opencompass.org.cn) | 书生·万卷，国产全面评估 |
| **Hugging Face Leaderboard** | 各模型开源 Benchmark 汇总 |
| **Imzoo** | 多模型对比平台 |
| **lmarena.ai** | 在线 Arena 对比 |

---

## 八、ArkNarrator 可直接采用的指标建议

基于 Victor 目前的项目特点（角色扮演 + Qwen vs Gemma 对比），建议分层建立：

### 第一层：快速自动化指标（每次训练必跑）

| 指标 | 方法 | 工具 |
|------|------|------|
| Val Loss | 训练时自动记录 | MLX-LM 内置 |
| 困惑度 (PPL) | 验证集困惑度 | MLX-LM |
| 角色归因准确率 | 三选一闭卷测试 | 自建 Judge |
| 矛盾检测率 | 二分类（是否破角色） | 自建 Judge |

### 第二层：深度质量指标（每周跑一次）

| 指标 | 方法 | 工具 |
|------|------|------|
| Pairwise Win-Rate | 同 prompt 两模型对比 | DeepSeek V4 Pro |
| 多轮对话一致性 | 3-5 轮历史后提问 | 自建 Judge |
| 说话风格分析 | 词汇统计 + Judge | DeepSeek V4 Pro |
| 角色关系测试 | 对抗性 prompt | 自建 Judge |

### 第三层：人类评估（阶段性手动跑）

| 指标 | 说明 |
|------|------|
| 人工 Pairwise | 直接对比两模型输出 |
| 主观质量打分 | 1-5 分，流畅度/角色一致性/趣味性 |
| A/B Test 主观偏好 | 盲测，哪个更像"真正的干员" |

---

## 九、避坑提示

1. **BLEU/ROUGE 不能衡量角色扮演质量** — 它们只看 n-gram 重叠，角色语气是否正确完全不相关
2. **单轮测试不足以证明角色一致性** — 必须有多轮测试
3. **Val Loss 和实际角色质量相关性有限** — Val Loss 低不等于角色扮演好
4. **Judge 模型也会偏心** — 某些模型可能偏好更长的输出，需要控制长度
5. **样本量小时不要下强结论** — 6 条 prompt 的结果方差太大

---

## 十、参考文献（可进一步阅读）

- **MMLU**: "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2021)
- **BIG-Bench**: "Beyond the Imitation Game Benchmark" (Srivastava et al., 2022)
- **MT-Bench**: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023)
- **AlpacaEval**: "AlpacaEval: An Automatic Evaluator of Instruction-Following Models" (Dubois et al., 2024)
- **G-Eval**: "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (Liu et al., 2024)
- **lm-evaluation-harness**: EleutherAI 开源评估框架，支持 60+ Benchmark
- **HH-RLHF**: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"

---

*整理完毕，如有遗漏欢迎补充～*
