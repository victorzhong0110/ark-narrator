# 干员复刻流程（Operator Pipeline）

> 把「做一个像样的干员」固化成可复用的 5 步流程。能天使是第一个跑通的样板，
> 换任何干员只需重跑这套流程 + 一步人工筛选。

## 核心理念：人设 ≠ 知识，别按干员分库

一个干员要两样东西，**必须分开**，否则又乱又难维护：

| | 是什么 | 放哪 | 作用 |
|---|---|---|---|
| **人设包 Persona**（每人一份，小而精） | 她是谁、怎么说话 | **常驻 system prompt**：`data/characters/<slug>.yaml`（角色卡 + ~10 条招牌台词 few-shot） | 让她**听起来像她** |
| **知识库 Knowledge**（全干员**共用一个**，打标签） | 可调用的事实：世界观、背景、事件、关系、台词 | **RAG 检索**：`data/lore/**`，每个 chunk 带 `character` / `type` / `tags` | 让她**答得准、不幻觉** |

**不要给每个干员建独立 KB。** 那会切断共享世界观与跨干员关系（能天使聊德克萨斯、叙拉古、源石时要用到公共 lore）。正确做法是**一个共享语料库 + 标签**：按干员分*文件*（`data/lore/operators/<slug>.jsonl`）只是便于维护，加载后是**一个统一索引**，检索时按 `character` 过滤/加权，同时永远允许通用世界 lore。

`type` 标签：`world`（通用世界观）/ `archive`（档案，第三人称事实）/ `voice`（语音台词，自成一句的风格锚）/ `story`（剧情台词，情境中的真实声音）。

## 关键：剧情要按「语气档位」用，不能拍平

同一个干员**对不同对象、在不同情绪/场景下，说话风格完全不同**——对队友调侃、对敌人
挑衅嘴硬、对脆弱难过的人温柔安抚、谈信念生死时庄重起誓、谈任务报酬时干脆利落。
把她所有台词拍平成一袋按话题检索，会丢掉这层差异，让她变成一个「平均音」、甚至串味。

所以剧情数据要**保留并利用语气档位（register）**：

1. **打标**（`build_operator_kb.py`）：每条剧情台词结合「她回应的那句 + 她的话」判定档位
   （`app/registers.py` 的 5 档：轻松调侃 / 挑衅嘴硬 / 温柔安抚 / 庄重认真 / 干活报酬），
   连同对话对象（interlocutor）一起写进 chunk。
2. **运行时判定**（`app/scene.py` `SceneTagger`）：每轮先判定**玩家这句话**的情绪档位。
3. **调用匹配示范**（`orchestrator` + `retriever.register_exemplars`）：取该干员**该档位下**
   最相关的真实台词，注入 system prompt——「当前氛围偏【X】，你在这种情境下会这样说：…」。

效果：同一个能天使，你示弱她会温柔安抚、你挑衅她会嘴硬反击、你谈钱她甩出「你开路我殿后
奖金对半分」、问她生死信念她会「以这把守护铳起誓」。

> 运行时场景判定默认走 **LLM 语义判定**（`ARK_SCENE_TAGGER=llm`，`app/scene.py` `LLMSceneTagger`）：
> 能判出无关键词的情绪（自我怀疑、间接挑衅、欲言又止的伤感），解析失败/出错自动回退关键词版
> `HeuristicSceneTagger`。代价是每轮多一次短分类调用。`heuristic` 可切回零成本纯关键词。
> （剧情台词的*打标*仍用关键词分类器，离线批处理，可同样换 LLM。）

### 性格底色是「分场景的」，不是一个概率

各档位台词的**占比**（如能天使 84% 是轻松调侃）**不是她的性格底色**——那只是她**碰巧参与的场景构成**（喜剧向活动多，自然日常台词多），说的是*那些剧情*而非*这个人*。把跨场景占比平均成「85% 爱玩」，正好犯了「把分场景的性格拍平成一个全局数」的错。

正确的底色是**条件的**：*在某种场景下，她的特征姿态是什么*。所以：

- `register_styles`（角色卡字段，键=语气档位）记录她**在每种情境下「是什么样」**：
  日常吊儿郎当 / 被挑衅嘴硬 / 有人脆弱时用打趣安抚不煽情 / 触及信仰才短暂庄重又跳回 / 谈钱干脆讲义气。
  运行时按当前档位，把对应那条姿态 + 该档位真实台词一起注入提示词。
- `data/lore/operators/<slug>.jsonl` 是**检索池**：按*当前场景*档位捞对应真台词。捞庄重台词时池里有多少日常毫无影响，决定「此刻像不像」的是*该档位有没有足够真料*。
  故入库做**均衡封顶**（`_PER_REGISTER`）：稀有档位（庄重/安抚）全留、过剩日常封到够检索——避免膨胀/噪声，对行为无损。
- 占比仅作**覆盖度**参考（每个档位料够不够），**不是性格信号**——别拿它写人设。

写 `register_styles` 的方法：读 `build_operator_kb.py` 打印的**各档位真实台词**，归纳她在该情境下的姿态。

## 三类源数据，缺一不可

| 源 | 是什么 | 价值 | 抓取 |
|---|---|---|---|
| 档案 handbook | 基础信息 + 性格描述 | 校正硬事实（种族/出身/阵营），防假身世 | `fetch_operator.py` |
| 语音 charword | 她的语音台词（第一人称短句） | **声音金标**，最适合做 few-shot | `fetch_operator.py` |
| 剧情 story | 她在剧情里的成段对话 | **情境中的声音 + 关系**，最能「像她」 | `mine_stories.py` |

> 教训：只用档案是不够的。能天使一开始「不像」，根因之一就是只有档案、没有剧情；
> 而且初版卡片把她写成「莱茵生命实验品」（错），剧情/语音数据直接证伪了它。

## 5 步流程

以能天使（slug=`exusiai`）为例：

```bash
# 1) 抓档案 + 语音（官方游戏数据）
python scripts/fetch_operator.py 能天使
#    → data/raw/operator_能天使.json（含 char_id=char_103_angel、9 段档案、38 条语音）

# 2) 挖剧情台词（按说话人 [name="能天使"] 提取）
python scripts/mine_stories.py 能天使 --acts act5d0      # 指定活动，快
#    （或不带 --acts 全扫；首次会下载剧情脚本并缓存到 data/raw/stories/）
#    → data/raw/stories_能天使.json（《喧闹法则》182 条台词）

# 3) 汇成共享知识库的打标签 chunk + 打印候选金句
python scripts/build_operator_kb.py 能天使 --slug exusiai
#    → data/lore/operators/exusiai.jsonl（archive+voice+story，全部 character=能天使）
#    → 终端打印「候选金句」供下一步人工筛

# 4) ★人工筛选★：把 data/characters/exusiai.yaml 调成精准人设
#    - 用档案校正 profile（种族/出身/阵营/称呼）
#    - 从候选金句里挑 ~10 条最有辨识度的做 example_lines（voice + story 混搭）
#    - 读各档位真实台词，为 register_styles 写「她在每种场景下是什么样」（分场景姿态，非占比）
#    - 写 forbidden / fallback_lines / politics_deflections（都用她的口吻）
#    这一步是流程里唯一需要人判断的环节——「她的本质」无法全自动。

# 5) 验证
python -m pytest                                          # 护栏回归
ARK_BACKEND=mlx python scripts/try_dialogue.py 能天使       # 真模型对话观感
```

## 产出物归属

| 文件 | 入库？ | 说明 |
|---|---|---|
| `data/raw/**`（官方表、剧情脚本、提取结果） | ❌ gitignore | 大、可重抓 |
| `data/lore/operators/<slug>.jsonl`（标签 chunk） | ✅ 版本控制 | 知识库的可复现产物 |
| `data/characters/<slug>.yaml`（人设包） | ✅ 版本控制 | 人工筛选的结晶 |

## 扩展到更多干员 / 更多剧情

- 换干员：把上面命令里的「能天使 / exusiai」替换即可；第 4 步照样人工筛。
- 更全的剧情：`mine_stories.py` 去掉 `--acts` 即全扫所有剧情节点（缓存后重跑免下载）；
  也可多给几个 `--acts` 合并她出场的多个活动/主线。
- 检索升级：`LexicalRetriever`（字符 bigram）已够；要更准可实现向量版 `Retriever`，
  按 `type` 给不同权重（如答事实偏 archive、要口吻偏 voice/story），编排层不用改。
