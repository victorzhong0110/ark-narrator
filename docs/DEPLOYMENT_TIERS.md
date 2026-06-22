# 部署方案分档（按公司资金 / 硬件选）

> 一个公司评估「能不能用」时，第一个问题是「我这点预算/这些卡跑得起吗」。所以本产品不只
> 一种形态，而是给**三档可选方案**——每档是一组协调好的配置（模型/后端/审核/记忆/服务），
> `ARK_PROFILE` 一键切换。各档因为底层全可插拔（后端、审核、场景判定、存储）才得以成立。

> 成本为**粗估**、随价格与用量浮动，仅供量级判断；并发为单节点经验值。

## 总览

| | **轻量自托管 budget** | **标准混合 standard** | **云旗舰 flagship** |
|---|---|---|---|
| 公司画像 | 小团队/试点/预算紧 | 正经上线、扛真实流量 | 旗舰体验、不差钱、要最像 |
| 硬件 | 1×消费级 GPU(4090/A10) 或 Apple Silicon | 2-4×中端 GPU(A10/L20/H20) 自托管 | 少量自托管 + 前沿 API（或 H 卡机队） |
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

## 升档不改代码

三档共用同一套引擎与安全护栏。从 budget 升 standard，只是把后端从本地 8B 换成更大模型、
审核从 mock 换成真网关、存储从 SQLite 换成 Redis——**全是配置切换**，因为这些都做成了可插拔接口
（`app/llm/`、`app/guard/cloud_audit.py`、存储层见工作流 B）。这意味着公司可以**低成本起步、随业务增长平滑升档**。

## 还需公司侧承担（非代码）

备案 / AI 标识合规、GPU 机队采购与运维、全角色内容团队、SLA 与 oncall——这些属公司侧投入，
不在引擎范围内，但分档时已按「自托管 vs 云」把硬件与人力的量级标清。
