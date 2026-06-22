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

## 待公司侧（非代码）
第三方渗透测试、备案 + AI 标识办法法务、云审真凭证接入、内部 mTLS、安全合规评审。
