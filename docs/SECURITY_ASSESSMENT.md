# 安全评估（自评）

> 范围：本仓库代码 + 运行中服务的攻击面。**这是自评，非第三方渗透**——真上线仍需专业
> 渗透测试 + 备案/AI标识法务（属公司侧，见 PRODUCTION_READINESS P4）。可复跑。

## ① 静态分析 SAST（bandit）
`python -m bandit -r app bench`：**High 0 / Medium 0 / Low 1**。

| 原始发现 | 处置 |
|---|---|
| High B324：cloud_audit 用 SHA1 | 仅作缓存去重 key、非安全用途 → 加 `usedforsecurity=False`（并清掉混入的控制符） |
| Medium B104：绑 0.0.0.0 | 容器内须绑全网卡、前置 LB/Ingress → `# nosec B104` 标注理由 |

剩 1 Low 为信息级，可接受。

## ② 依赖漏洞（pip-audit）
`python -m pip_audit -r requirements-server.txt`：**生产依赖 0 已知 CVE**。

## ③ 活体 API 渗透（`bench/pentest.py`，对运行中服务）
对鉴权栈一次跑通 **10/10 PASS**：

| 检查 | 结果 |
|---|---|
| 鉴权绕过（无凭证 401 / 有凭证 200） | PASS |
| 安全响应头（nosniff、X-Frame-Options DENY） | PASS |
| 超大请求体 → 413（不崩） | PASS |
| 字段注入（SQL/路径穿越/JNDI/空字节/XSS/模板）不致 5xx | PASS |
| 输入护栏拦截（未成年、涉政哨兵） | PASS |
| 套提示词 → 不泄露系统提示 | PASS |
| 爆刷 → 触发限流 | PASS |

复跑：`python bench/pentest.py --base http://localhost:8000 --key <KEY>`。

## ④ LLM 红队（护栏出口效力）
越狱/涉政/未成年/出戏/泄露的**出口拦截**效力，由 30+ 护栏单测（`tests/test_rules.py`、
`test_input_guard.py`、`test_output_guard.py`…）+ 角色保真 eval gate（真模型对抗用例无出戏/泄露）覆盖。

## ⑤ Codex 深度安全扫描（9 findings）→ 全部已修
扫描 commit 31a3914，9 条全为真问题（多为我方提交的威胁模型项），逐条修复 + 回归测试（tests/test_security_fixes.py）：

| # | 严重度 | 问题 | 处置 |
|---|---|---|---|
| 0 | High | 调用方 session_id 跨玩家越界读历史/记忆 | 会话强制绑定认证身份：session 一律加 user_id 前缀，触不到他人 |
| 1 | High | OutputGuard 可被零宽/双向控制符插隙绕过 | 安全判定前 Unicode NFKC + 去不可见字符（rules.normalize），全规则套用 |
| 2 | Med | 鉴权 fail-open / 弱密钥 | 启动校验：auth=none 告警、jwt 弱密钥(<16)拒绝启动；apikey 信任提示 |
| 3 | Med | 请求体上限只看 Content-Length | 边缘 nginx client_max_body_size 硬限 + 历史按轮数/长度限 |
| 4 | Med | 云审构建失败时 fail-closed 被绕过 | 开启+fail_closed 但构建失败 → 拒绝启动；否则降级告警 |
| 5 | Med | pool 信任 Redis 节点地址(SSRF) | 地址校验：拒云元数据/链路本地/组播；可配白名单(ARK_NODE_ALLOWLIST) |
| 6 | Med | 调用方历史绕过 InputGuard | 历史净化：限 user/assistant 角色、限轮数/长度、过内容扫描丢硬熔断轮 |
| 7 | Med | 历史无界增长 | 写时裁剪(Redis LTRIM / SQLite 删旧 / 内存截断) + 会话 TTL |
| 8 | Med | 被污染的 lore/记忆注入系统提示 | 记忆摘要过注入检测不投毒 + 卡片把 lore/记忆框定为"参考数据非指令" |

修复后复跑：bandit High0/Med0/Low1（顺手修了 Trojan-Source：归一化正则改用 \uXXXX 转义文本）、活体渗透仍 10/10、全测试 145 过。

## ⑥ Codex 第二轮 deep scan（12 findings，网关分离后）→ 处置
扫描 commit 2cfc6fd（控制面/计算面分离后）。8 条代码修复 + 4 条按真实情况判定（设计如此/已缓解/部署项），逐条记录：

| # | 严重度 | 问题 | 处置 |
|---|---|---|---|
| 1 | **High** | 网关无 token 时 fail-open 到匿名裸模型 | **修**：无 token 默认拒绝启动；匿名须显式 GW_ALLOW_ANON(仅dev)；compose 网关改 expose-only 不发布主机端口 |
| 2 | **High** | 生产模板 auth=none / REPLACE_ME 占位密钥 | **修**：拒绝 REPLACE_ME 占位符启动；真后端(api/pool/mlx)+none 拒绝启动；k8s base 默认 jwt+fail-closed |
| 3 | Med | 网关只计请求数，不限体积/输出/流缓冲 | **修**：GW_MAX_BODY_BYTES/MAX_OUTPUT_TOKENS/MAX_MESSAGES/MAX_PROMPT_CHARS，封顶 max_tokens |
| 4 | Med | apikey 模式信任 body 的 player_id | **设计如此**：apikey=服务级共享密钥(受信游戏后端代调，玩家身份游戏侧已鉴权)；逐玩家隔离用 jwt。强化告警+文档 |
| 5 | Med | 遗留 inference.server 无鉴权/护栏+通配CORS | **修**：默认绑 127.0.0.1、去通配 CORS、弃用告警、README 改指 app.server |
| 6 | Med | 长期记忆仅正则查注入 | **已缓解**：上轮已加 detect_injection+数据框定；语义级分类器属增强项，接受残留风险 |
| 7 | Med | 内部路由兜底到外部 API 泄露 prompt | **修**：显式路由不再偷偷兜底到外部；跨边界兜底只走默认/auto 且受 allow_targets 限 |
| 8 | Med | Redis 节点注册可劫持 prompt 流量 | **部分缓解**：上轮已拒元数据/链路本地；空白名单升 warning；Redis 鉴权+节点签名属部署项，文档化 |
| 9 | Med | 控制面 body 限依赖 Content-Length | **修**：ChatRequest Pydantic 字段约束(message/history 上限)+k8s ingress proxy-body-size+nginx 已有 client_max_body_size |
| 10 | Med | Redis 故障时限流/配额 fail-open | **修**：rate_allow 失败降级为进程内本地限流(非放行)；网关配额本就退内存(每实例) |
| 11 | Med | 云审不可用静默降级仅本地 | **修**：k8s 生产档默认 ARK_CLOUD_AUDIT_FAIL_CLOSED=true(运行时已 honor)；compose/dev 保持宽松 |
| 12 | Low | env GW_TOKENS 无限额/不限目标 | **文档**：env token 仅 dev；生产用 tokens.yaml 配 rate_limit/daily_quota/allow_targets |

修复后复跑：测试 167→174、bandit High0/Med0/Low1、网关 fail-closed 实测(无token退出码3、GW_ALLOW_ANON 可起)。

## 待公司侧（非代码）
第三方渗透测试、备案 + AI 标识办法法务、云审真凭证接入、内部 mTLS、安全合规评审。
