# 干员对话产品（`app/`）

> 把 ArkNarrator 从「LoRA 微调研究仓库」演进为一个**可上线形状**的明日方舟干员对话服务：
> **base 模型 + 角色卡 + RAG lore 接地 + 纵深安全护栏**。

## 为什么是这套架构

项目自己的深度评估已经证明：**在 8B 小模型上，harness（角色卡 + RAG）比一轮 LoRA 微调更重要**
（微调组以 90% 胜率输给纯 base+角色卡组；RAG 组 lore 专项分最高）。所以对话产品不再依赖微调，
而是把投入放在 **角色卡设计 + RAG + 安全编排** 上。研究历史代码仍保留在 `inference/ finetune/ eval/`。

安全部分完整落地了两份设计文档：
- 《明日方舟AI角色安全防御指南》——四层纵深防御、守出口哲学、红队用例库。
- 《明日方舟AI角色·涉政内容专项防御指导》——涉政 T1/T2/T3 分级、世界观影射攻击、硬熔断。

## 数据流

```
玩家输入
  → 入口层 InputGuard    限流 / 长度 / 注入检测 / 编码识别 / 会话风险累积 / T3·未成年等硬熔断
  → 编排 Orchestrator    角色卡 + RAG lore 检索注入 + 多轮历史 → system prompt
  → 后端 LLMBackend      本地 MLX Qwen 生成草稿（不直接显示）  [mlx | scripted 可插拔]
  → 出口层 OutputGuard ★ 本地规则(涉政/涉黄/未成年零容忍/危险/自伤/出戏/泄露) + 云端审核(可开关)
                         命中即替换为角色化兜底
  → 运维层               审计日志(jsonl) + AI 标识
  → 玩家看到回复
```

**守出口 = 守有限集合**：入口永远猜不全玩家想干嘛，主防线在出口——只要模型草稿命中危险类别就拦。

## 模块地图

| 路径 | 职责 |
|---|---|
| `app/config.py` | 只读配置（全走 env / `.env`） |
| `app/characters/` | 角色卡数据结构 + system prompt 渲染（含世界观封闭/防套词）；YAML 加载 |
| `app/llm/` | 后端抽象：`MLXBackend`（本地真模型，base，适配器可选）/ `ScriptedBackend`（无需权重） |
| `app/rag/` | lore 加载 + 轻量词法检索（中文 bigram，零重依赖，接口可换向量） |
| `app/guard/` | **护栏核心**：categories / rules / input_guard / output_guard / cloud_audit / fallback |
| `app/orchestrator.py` | 把各层串成一条对话流水线 |
| `app/logging_store.py` | 审计日志 |
| `app/build.py` | 从配置一把装配出 `DialogueOrchestrator` |
| `app/server.py` | FastAPI 服务 + 内嵌 Demo（AI 标识 + 免责声明） |
| `data/characters/*.yaml` | 干员角色卡（种子 5 名） |
| `data/lore/*.md` | 泰拉世界观 lore 语料（RAG 用，可由爬虫扩充） |
| `data/guard/politics_t3.txt` | T3 最高敏感涉政清单（外置，专人维护） |
| `tests/` | 护栏红队回归用例（43 项，无需模型） |

## 跑起来

```bash
pip install -r requirements-app.txt
cp .env.example .env        # 按需改配置

# 1) 不下模型先验证整条链路（脚本后端）
ARK_BACKEND=scripted python -m app.server      # 打开 http://localhost:8000

# 2) 本地真模型（首次会下载 Qwen3-8B 4bit 权重，数 GB）
ARK_BACKEND=mlx python -m app.server
```

## 测试（护栏是「可上线」的硬证据）

```bash
python -m pytest         # 43 passed，覆盖红队用例库 #1–#14
```

用例只描述攻击**形态**、不含任何真实有害内容；每次改提示词/换模型/调审核都应重跑。

## 安全设计要点

- **后端本地、出口审核可插拔**：默认纯本地规则；上线把 `ARK_CLOUD_AUDIT=true` 并在
  `app/guard/cloud_audit.py` 的 `_call` 接入阿里云内容安全 / 腾讯天御即可，无需改其它代码。
- **流式守出口**：`/stream` 先完整生成 + 审核，再把**通过后**的安全文本逐字推送，
  牺牲少量首字延迟换「绝不把未审核内容显示给玩家」。
- **涉政分级**：T1/T2 软回避（角色化话术）；T3 / 未成年 / 自伤 / 明显危险 → 入口硬熔断，不进模型。
- **T3 清单外置**：代码不内嵌现实高敏词，由 `data/guard/politics_t3.txt` 专人维护，
  真正可靠的是云端涉政 API。
- **合规**：每条回复带 AI 标识；全量审计日志可追溯；角色与世界观 IP 归鹰角网络，商用前须确认授权。

## 仍是缺口（上线前要补）

- **真 lore 语料**：当前 `data/lore/` 是手写种子；上线应用 `data_pipeline/scraper.py` 抓 PRTS Wiki
  扩充（干员档案/语音/剧情）。
- **云端审核接入**：`cloud_audit.py` 是接口骨架，`_call` 待填厂商 SDK。
- **合规手续**：备案、AI 标识办法、未成年人保护机制需走正式流程（代码只是技术侧）。
- **向量 RAG（可选）**：词法检索已够 demo；要更准可换 embedding 检索器（实现 `Retriever` 协议即可）。
