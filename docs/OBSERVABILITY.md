# 可观测与运维（P5）

线上看得见、坏了能报警、有人能处理。

## 一键起监控栈
```bash
docker compose -f deploy/compose-prod.yml --profile monitoring up -d --scale app=3
# Grafana http://localhost:3000（匿名可看，预置「ArkNarrator 概览」仪表盘）
# Prometheus http://localhost:9090（已抓所有 app 副本 + 告警规则）
# Jaeger http://localhost:16686（设 ARK_OTEL_ENDPOINT=http://jaeger:4318/v1/traces 后有链路）
```

## 指标（/metrics）
| 指标 | 含义 |
|---|---|
| `ark_requests_total{path,status}` | 请求量（含按 Prometheus `instance` 看各副本 → LB 分散） |
| `ark_request_seconds` | 延迟直方图（p50/p95/p99 用 histogram_quantile） |
| `ark_blocked_total{category}` | 护栏拦截率（涉政/涉黄/出戏/泄露…） |
| `ark_register_total{register}` | 语气档位分布（是否卡单一模式） |
| `ark_replies_total{character}` | 各角色回复量 |
| `ark_overloaded_total`/`ark_timeout_total`/`ark_auth_fail_total`/`ark_store_errors_total` | 429/504/401/存储降级 |

## 仪表盘
`deploy/monitoring/grafana/dashboards/ark.json`：请求速率/延迟分位/拦截率/异常/各实例分散/语气档位
六面板，Grafana 自动预置。

## 告警（`deploy/monitoring/alerts.yml`）
5xx 错误率>5% · p95>1s(SLO) · 存储降级 · 拦截率>20%(疑似攻击) · 持续 429 过载。
接 Alertmanager 即可发钉钉/PagerDuty。

## 链路追踪
`app/tracing.py`（OTel，env 开关）：装 otel 包 + 设 `ARK_OTEL_ENDPOINT` → FastAPI 自动打点经 OTLP
上报 Jaeger；未配置则 no-op。

## SLO（建议起点）
| 指标 | 目标 |
|---|---|
| 可用性 | 99.9% |
| p95 延迟 | < 1s（不含模型；含模型按机队容量定，见 CAPACITY.md） |
| 护栏漏报（涉政/未成年） | 0 容忍（出事即 P0） |

## On-call Runbook（速查）
- **5xx 飙升**：看 Grafana「各实例分散」定位是否单副本坏 → `/readyz` 不过的副本会被 LB 剔除；
  必要时 `docker compose restart`/`kubectl rollout restart`。
- **持续 429**：扩 app 副本（`--scale`/HPA）或加模型节点（`join_cluster.sh`）。
- **存储降级告警**：查 Redis 健康；ResilientStore 已保请求不 500，但记忆/历史会暂失 → 修复 Redis。
- **拦截率突增**：查 `ark_blocked_total{category}`——涉政/未成年突增可能被攻击，启动红队 + 收紧审核。
- **延迟 p95 破 SLO**：查模型机队容量（CAPACITY.md 的 N=ceil(Q·R/T)）→ 加节点。
