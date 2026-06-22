# 接入你的 IP（Integration）

> 这是一个**角色对话引擎**，不是一个明日方舟专用 demo。
> 把它指向**你公司自己的角色数据**，下游（角色卡、RAG、分场景语气、纵深安全护栏、
> 服务）全部复用。明日方舟只是**参考实现**（用公开数据跑通）。

## 两个解耦的接缝

引擎只依赖两个抽象，换 IP 就是替换这两样，**不改引擎代码**：

| 接缝 | 抽象 | 你提供什么 |
|---|---|---|
| **数据** | `OperatorSource → OperatorIR`（`app/ingest/`） | 你角色库的一个适配器，或把数据导出成 OperatorIR JSON |
| **世界观/IP** | `WorldProfile`（`data/world.yaml`） | 作品名 / 世界名 / 角色统称 / 版权方 |

下游全部源无关：建知识库、角色卡脚手架、语气档位、安全护栏（世界观封闭按 WorldProfile 模板化）、FastAPI 服务。

## 3 步接入

```bash
# 1) 配置你的 IP（换掉明日方舟）
cat > data/world.yaml <<'YAML'
work: 星海纪元
universe: 银河
role_term: 探员
ip_owner: 你的公司
YAML

# 2) 接入你的角色数据（二选一）
#    a. 零代码：把每个角色导出成 OperatorIR JSON（schema 见下），放 data/companies/<ip>/<name>.json
#    b. 写适配器：实现 app/ingest/source.py 的 OperatorSource 协议，直连你的数据库/API
python scripts/build_operator_kb.py 星澪 --source generic \
    --data data/companies/example/lyra.json --slug lyra
#    → data/lore/operators/lyra.jsonl（打 character/type/register 标签的检索池）
#    → 终端打印各档位候选金句，供下一步人工筛

# 3) 人工筛出精准人设（唯一需要判断的一步）
#    据候选金句写 data/characters/lyra.yaml（profile / example_lines / register_styles /
#    forbidden / fallback_lines），然后：
ARK_BACKEND=mlx python -m app.server     # 你的角色就在引擎里跑起来了
```

## OperatorIR JSON schema（零代码接入只需导成这个）

```json
{
  "name": "星澪",
  "codename": "Lyra",
  "faction": "星海纪元 · 深空探勘局",
  "profile_facts": ["第三人称档案事实，一条一句"],
  "voice_lines": [{"title": "条目名", "text": "第一人称语音台词"}],
  "story_lines": [
    {"text": "剧情里她说的一句话",
     "interlocutor": "这句说给谁",
     "prev": "她在回应的上一句（语境）",
     "scene": "场景/章节名"}
  ]
}
```

字段含义与价值：
- `profile_facts`（档案）— 校正硬事实、防幻觉/防假身世。
- `voice_lines`（语音）— **角色声音的金标**，最适合做 few-shot。
- `story_lines`（剧情）— **情境中的声音 + 关系 + 情绪**，最能「像她」；`interlocutor`/`scene`
  让引擎能按*当前场景情绪*调用她在相似情境下的真实说法（分场景语气档位）。

> 三类源缺一不可：只用档案不够（角色会「懂设定但不像本人」）。你内部的角色 bible、
> 配音台本、剧情脚本，分别对应这三类。

## 已验证：非明日方舟数据零改代码跑通

`data/companies/example/lyra.json` 是一个**完全虚构、非明日方舟**的科幻角色「星澪」。
`scripts/build_operator_kb.py 星澪 --source generic --data ... --slug lyra` 直接产出她的
打标签知识库与各档位候选金句——**没有改任何引擎代码**。见 `tests/test_ingest.py`。

## 你提供 vs 引擎复用

| 你（接入方）提供 | 引擎复用（不动） |
|---|---|
| 角色数据（适配器或 JSON 导出） | 知识库构建 + 共享检索 |
| WorldProfile（你的 IP 词） | 纵深安全护栏（入口/出口/涉政分级/未成年零容忍/出戏/泄露） |
| 人工筛人设（角色卡 + register_styles） | 分场景语气档位 + LLM 场景判定 |
| 你的合规口径（接你的内容安全 API） | 云端审核接口（`app/guard/cloud_audit.py` 留好，填 `_call`） |
| 你的模型/算力 | 后端抽象（本地 MLX / 可换 API / vLLM） |

> 安全护栏里的**法规红线（涉政/未成年/涉黄等）是按司法辖区**的，与 IP 无关，开箱即用；
> 只有「世界观封闭」那部分随 WorldProfile 变。
