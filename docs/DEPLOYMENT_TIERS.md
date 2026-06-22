# 部署方案分档（按公司资金 / 硬件选）

> 一个公司评估「能不能用」时，第一个问题是「我这点预算/这些卡跑得起吗」。所以本产品不只
> 一种形态，而是给**三档可选方案**——每档是一组协调好的配置（模型/后端/审核/记忆/服务），
> `ARK_PROFILE` 一键切换。各档因为底层全可插拔（后端、审核、场景判定、存储）才得以成立。

> 成本为**粗估**、随价格与用量浮动，仅供量级判断；并发为单节点经验值。

## 总览

| | **轻量自托管 budget** | **标准混合 standard** | **云旗舰 flagship** |
|---|---|---|---|
| 公司画像 | 小团队/试点/预算紧 | 正经上线、扛真实流量 | 旗舰体验、不差钱、要最像 |
| 硬件 | 1×消费级 GPU(4090/A10) 或 **1 台 Mac mini** | 2-4×中端 GPU 或 **Mac mini 集群** | 少量自托管 + 前沿 API（或 H 卡机队 / Mac Studio 集群）|
| 生成模型/后端 | 本地 Qwen 7-8B 4-bit（MLX/llama.cpp/vLLM-small） | 自托管 Qwen 14-32B(vLLM 批处理) 或 API | 前沿大模型 API 或 72B+ 自托管 |
| 场景判定 | 关键词（零额外调用） | LLM 语义 | LLM 语义 |
| 内容审核 | 本地规则 + 最低档云审 | 商用云审 API + 缓存 | 商用云审 + 人工复核回路 |
| 记忆/存储 | SQLite / 进程内 | Redis(会话/限流) + Postgres(历史/长期记忆) | 全量长期记忆 + 向量记忆，跨会话 |
| 服务 | 单机，并发 ~几十 | 多 worker、可扩，并发 ~百-千 | 云原生、高并发 |
| 质量 | 及格~良好 | 好（竞品基线） | 最佳 |
| 成本量级[粗估] | 一台机器折旧 + 少量云审；~千元/月级 | GPU 机队 + 云审 + 存储；~万-十万/月级 | API token + 云审 + 运维；随 DAU 线性，~十万+/月级 |

## 一键切换

```bash
ARK_PROFILE=budget   python -m app.server     # 轻量自托管
ARK_PROFILE=standard python -m app.server     # 标准混合（需配 ARK_API_BASE_URL/KEY、审核网关）
ARK_PROFILE=flagship python -m app.server     # 云旗舰（需配前沿 API、厂商云审凭证）
```

优先级：**显式环境变量 > `.env` > 档位预设 > 代码硬默认**——档位只填你没单独设的项，可逐项微调。
预设内容见 `app/profiles.py`。

## 每档要补的部署项

| 档 | 你需要提供 |
|---|---|
| budget | 一台 GPU 机；（合规上线时）一档云审网关地址/凭证 |
| standard | 模型服务（自托管 vLLM 或 `ARK_API_BASE_URL/KEY`）+ 审核网关 `ARK_CLOUD_AUDIT_ENDPOINT` + Redis/Postgres（记忆层，工作流 B）|
| flagship | 前沿模型 API 凭证 + 厂商云审凭证（`ARK_CLOUD_AUDIT_PROVIDER=aliyun` 等）+ 完整记忆/监控栈 |

## 硬件路线：Mac mini 横向扩展（Apple Silicon 自托管）

本产品底层是 MLX（Apple Silicon）栈，所以 **Mac mini 集群是一条对口且差异化的自托管路线**：
perf/瓦、perf/元都强，体积小、能耗低（满载约数十瓦）、安静，**数据全留本地**——对要隐私/合规
的游戏公司很合适。

### 单机容量（4-bit，粗估，随机型/量化浮动）

| 机型 | 统一内存 | 能跑 | 单流速度[粗估] | 定位 |
|---|---|---|---|---|
| Mac mini M4 | 16-32GB | 7-8B | ~20-40 tok/s | budget 节点 |
| Mac mini M4 Pro | 48-64GB | 至 ~32B | ~10-25 tok/s | standard 节点 |
| Mac Studio M4 Max/Ultra | 128-512GB | 70B+ | 视模型 | flagship 自托管节点 |

### 拓扑：副本横向扩展（推荐，面向「多用户并发」）

对话是「多用户、各自一条请求」的负载——**不需要把一个大模型拆到多机**，而是**多副本**：
每台 mini 跑一个完整模型实例，前面一个负载均衡器分发请求，**状态外置共享**。

```
                 ┌── Mac mini #1
玩家 → LB/路由 ──┼── Mac mini #2        每台 = 一个完整模型实例（无状态 worker）
                 └── Mac mini #N
                          │ 共享状态
            Redis(会话/限流) + Postgres(对话历史/长期记忆)   ← 工作流 B
```

N 台 ≈ N× 吞吐。**前提是 worker 无状态**——会话历史/长期记忆/限流/会话风险都已外置到可插拔
存储（`app/store/`：memory / sqlite / **redis**），用 `ARK_STORE=redis` 多机共享即可，加一台 mini
就是加一份吞吐。（Postgres 等更重的持久层可作为 redis 之外的选项后续接入。）

### 两种接法（都已被现有代码支持）

1. **每台 mini 跑完整 app**（`ARK_BACKEND=mlx`）：LB 在 app 实例间做 L7 均衡，状态走 Redis。
2. **推理与 app 分离**（更利于各自扩缩）：每台 mini 跑 `python -m mlx_lm.server`（**OpenAI 兼容，已验证可用**），
   中心 app 层用 `ARK_BACKEND=api`、`ARK_API_BASE_URL=http://<mini集群-LB>/v1` 调用整个 mini 池。
   ——`APIBackend`（工作流分档时已加）直接就能指向 mini 机队，无需改代码。

### 取舍：Mac mini 集群 vs NVIDIA GPU 服务器

| | Mac mini 集群 | GPU 服务器(A10/L20 + vLLM) |
|---|---|---|
| 强项 | 低 capex/能耗/体积、数据本地、perf/元优（中等并发） | 单机批处理吞吐高、vLLM 连续批处理成熟（高并发） |
| 弱项 | 单机批处理吞吐弱、无 CUDA 生态 | capex/能耗高 |
| 适合 | 中等规模、隐私敏感、预算受限的自托管 | 高并发规模化 |

> 另有「分布式 MLX / exo 把一个超大模型拆到多台 mini」的玩法——用于**单机装不下的大模型**
> 在本地跑（如 70B+ 拆 2-3 台）；延迟更高、更脆，属旗舰自托管的小众选项。**面向多用户服务，
> 优先副本横向扩展，而非拆单模型。**

### 加一台 Mac：具体步骤

前提（一次性）：①一个共享 Redis（`ARK_STORE=redis`）②一个 LB（`deploy/nginx.example.conf`）。

**形态 A（推荐，Mac 当模型节点）**
```bash
# 新 Mac 上：
pip install mlx-lm
python -m mlx_lm.server --model mlx-community/Qwen3-8B-4bit --host 0.0.0.0 --port 8080
# 然后在 LB 的 mlx_model_pool 加一行 server <新Mac-IP>:8080; → nginx -s reload
# app 层不动、不重启。app 层用 ARK_BACKEND=api、ARK_API_BASE_URL=http://<LB>/v1
```

**形态 B（每台 Mac 跑完整 app）**
```bash
# 新 Mac 上：
git clone <repo> && cd ark-narrator && pip install -r requirements-app.txt
export ARK_BACKEND=mlx ARK_STORE=redis ARK_REDIS_URL=redis://<redis主机>:6379/0
uvicorn app.server:app --host 0.0.0.0 --port 8000
# LB 的 ark_app_pool 加一行 server <新Mac-IP>:8000; → reload
# 状态在共享 Redis，新机即刻共享会话/长期记忆；LB 用 /readyz 探活
```

> 诚实说明：这是架构**设计支持**的路径（Redis 存储、API 后端、mlx_lm.server 兼容、/readyz
> 探针均已验证），但跨两台物理 Mac 的端到端集群我尚未实测；LB 与 Redis 需自行架设；
> 每个节点各需下载一份模型权重。

## 升档不改代码

三档共用同一套引擎与安全护栏。从 budget 升 standard，只是把后端从本地 8B 换成更大模型、
审核从 mock 换成真网关、存储从 SQLite 换成 Redis——**全是配置切换**，因为这些都做成了可插拔接口
（`app/llm/`、`app/guard/cloud_audit.py`、存储层见工作流 B）。这意味着公司可以**低成本起步、随业务增长平滑升档**。

## 还需公司侧承担（非代码）

备案 / AI 标识合规、GPU 机队采购与运维、全角色内容团队、SLA 与 oncall——这些属公司侧投入，
不在引擎范围内，但分档时已按「自托管 vs 云」把硬件与人力的量级标清。
