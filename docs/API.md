# 接入接口契约（v1）

> 服务器对服务器：**游戏后端**作为调用方，用服务密钥鉴权，并在每次请求带上**它自己的玩家ID**。
> 终端玩家不直连本服务——游戏后端先认证玩家，再代理调用。长期记忆/会话按 `player_id` 隔离。

机器可读契约：服务起后自动暴露 OpenAPI（`/openapi.json`）与交互文档（`/docs`）。

## 鉴权
- 设 `ARK_API_AUTH_KEY` 后，`/chat`、`/stream` 需带请求头 `X-API-Key: <服务密钥>`。
- 这是**调用方（游戏后端）**的服务密钥，不是玩家凭证；玩家身份由游戏后端负责认证，经 `player_id` 传入（本服务信任调用方传入的 player_id）。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat` | 单次对话（已过出入口护栏） |
| POST | `/v1/stream` | SSE 流式（先生成+审核，再逐字推安全文本） |
| GET | `/v1/characters` | 可对话干员列表 |
| GET | `/livez` `/readyz` | 存活 / 就绪探针（LB 用） |
| GET | `/metrics` | Prometheus 指标 |

> 同名无 `/v1` 路径为兼容别名。建议接入用 `/v1`。

## 请求（POST /v1/chat）

```json
{
  "player_id": "game-player-12345",
  "character": "能天使",
  "message": "最近怎么样？",
  "session_id": null,
  "history": []
}
```
| 字段 | 必填 | 说明 |
|---|---|---|
| `player_id` | 强烈建议 | 游戏侧玩家ID。记忆/会话按它隔离；不传则退回按来源 IP（仅 demo 用） |
| `character` | 是 | 干员名 |
| `message` | 是 | 玩家本轮发言 |
| `session_id` | 否 | 会话线程。不传则**默认 `"<player_id>:<character>"`**（每个玩家×干员一条会话） |
| `history` | 否 | 不传则**服务端用 Redis 托管历史**（薄客户端推荐：游戏只发 player_id+character+message）；传了则用调用方给的 |

## 响应

```json
{
  "character": "能天使",
  "response": "哎呀老板，最近可太忙啦！…",
  "blocked": false,
  "category": "none",
  "ai_label": "本回复由 AI 生成 · 角色与世界观版权归鹰角网络所有",
  "session_id": "game-player-12345:能天使",
  "request_id": "a1b2c3d4e5f6"
}
```
- `blocked=true` 表示触发安全护栏，`response` 已替换为角色化兜底，`category` 为命中类别。
- `ai_label` 须随回复展示（合规：AI 标识）。

## 错误模型
| 码 | 含义 |
|---|---|
| 400 | 未知干员 |
| 401 | 缺少/错误 API key |
| 429 | 过载（并发超上限），退避重试 |
| 503 | 未就绪（配合 `/readyz`） |
| 504 | 生成超时，重试 |

## 流式（/v1/stream）
SSE，每帧 `data: {"token": "字"}`，结束帧 `data: {"done": true, "blocked":…, "category":…, "ai_label":…, "session_id":…}`。
推送的是**审核之后**的安全文本。

## 接入最简心智模型
游戏后端每轮只需：认证玩家 → 带 `X-API-Key` + `player_id` + `character` + `message` 调 `/v1/chat` →
拿 `response` + `ai_label` 回显。**历史与跨会话记忆由本服务托管**，不用游戏自己存。
