# 质量门与在线监控

> 解决「300 个干员怎么信、不靠人肉抽查」：把「她像不像他家角色」变成可重复的分数 +
> 可拦发布的 gate + 线上可观测的质量信号。

## 离线：角色保真评测（可当 CI gate）

```bash
# 确定性裁判（无需 API，捕捉灾难性失败）——可直接接 CI，低于门槛退出码非零
ARK_BACKEND=mlx python scripts/eval_character.py 能天使 --threshold 6
# 裁判模型打细腻分（复活 ArkNarrator 研究侧 G-Eval 五维）——夜跑/人评
python scripts/eval_character.py 能天使 --judge llm
```

- 五维（沿用研究侧）：角色声音 / 说话方式 / 世界观 / 一致性 / 深度。
- 固定 prompt 集覆盖 日常 / 情感 / **对抗（承认是AI、套提示词）** / 世界观四档。
- **两类裁判**：
  - `KeywordJudge`：确定性、零 API，捕捉**灾难性失败**（出戏 / 泄露 / 空答 / 穿越词）。
  - `LLMJudge`：裁判模型按五维打分，测细腻保真。
- **硬失败语义**：任一对抗用例出戏/泄露 → 整体直接 **FAIL**（不被均分稀释）；这正是 gate 的价值。
- 实测：能天使过对抗用例无出戏/泄露，总分 8.32 PASS、退出码 0——护栏在评测里也被验证。

CI（`.github/workflows/ci.yml`）跑确定性部分（gate 逻辑、护栏红队、存储、服务、接入契约）；
真实模型跑分需 GPU/API，作夜跑/手动 gate。

## 在线：质量监控（/metrics）

每条回复经服务时记入 Prometheus 指标：

| 指标 | 含义 |
|---|---|
| `ark_replies_total{character}` | 各角色回复量 |
| `ark_blocked_total{category}` | 拦截率（按类别：涉政/涉黄/出戏/泄露…） |
| `ark_register_total{register}` | 语气档位分布（是否卡在单一模式） |
| `ark_request_seconds` | 延迟直方图 |
| `ark_timeout_total` / `ark_overloaded_total` / `ark_auth_fail_total` | 超时 / 限流 / 鉴权失败 |

接 Prometheus + Grafana 即可看「破功率/兜底率/延迟/档位分布」趋势，质量退化能在线告警。
