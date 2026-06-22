"""把某干员的三类源数据，汇成共享知识库里的「打标签 chunk」。

输入（由前两步产出）：
  data/raw/operator_<name>.json   档案 + 语音（fetch_operator.py）
  data/raw/stories_<name>.json    剧情台词（mine_stories.py）
输出：
  data/lore/operators/<slug>.jsonl   每行一个 chunk：{text, character, type, tags, source}
    type ∈ {archive, voice, story}；character 统一为干员名
  并打印「候选金句」（语音+剧情各若干），供人工筛入角色卡 example_lines。

用法：python scripts/build_operator_kb.py 能天使 --slug exusiai
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.registers import REGISTER_LABEL, Register, classify_register

RAW = Path("data/raw")
OUT_DIR = Path("data/lore/operators")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_MIN_STORY_LEN = 8       # 太短的剧情台词（语境依赖强）不进库
_PER_REGISTER = 40       # 每个语气档位最多保留多少条（均衡，防日常档挤掉稀薄档）


def _load(name: str, kind: str) -> dict:
    p = RAW / f"{kind}_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--slug", required=True, help="输出文件名（英文），如 exusiai")
    args = ap.parse_args()

    op = _load(args.name, "operator")
    st = _load(args.name, "stories")

    chunks: list[dict] = []

    # 档案 → type=archive
    for s in op.get("stories", []):
        text = s.get("text", "").strip()
        if len(text) >= _MIN_STORY_LEN:
            chunks.append({"text": text, "character": args.name, "type": "archive",
                           "tags": [args.name, s.get("title", "")], "source": "handbook"})

    # 语音 → type=voice（自成一句的风格锚）
    for v in op.get("voices", []):
        text = v.get("text", "").strip()
        if len(text) >= 4:
            chunks.append({"text": text, "character": args.name, "type": "voice",
                           "tags": [args.name, v.get("title", "")], "source": "charword"})

    # 剧情 → type=story：先按语气档位分桶，每档位均衡保留 _PER_REGISTER 条
    story_lines = [ln for ln in st.get("lines", []) if len(ln.get("text", "")) >= _MIN_STORY_LEN]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ln in story_lines:
        reg = classify_register(f"{ln.get('prev', '')} {ln['text']}")  # 回应句+她的话，更准
        buckets[reg.value].append(ln)
    for reg_val, items in buckets.items():
        # 偏好有信息量的中等长度句子（12–80 字），同长度下取更长的
        items.sort(key=lambda x: (0 if 12 <= len(x["text"]) <= 80 else 1, -len(x["text"])))
        for ln in items[:_PER_REGISTER]:
            chunks.append({"text": ln["text"], "character": args.name, "type": "story",
                           "tags": [args.name, ln.get("act", "")], "source": "story",
                           "register": reg_val, "interlocutor": ln.get("interlocutor", "")})

    out = OUT_DIR / f"{args.slug}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_type = Counter(c["type"] for c in chunks)
    by_reg = Counter(c["register"] for c in chunks if c["type"] == "story")
    print(f"写入 {len(chunks)} 个 chunk → {out}")
    print(f"  type 分布：{dict(by_type)}")
    # 仅作覆盖度参考——占比 ≠ 性格（多少是「她参与的场景构成」的产物，不是她的底色）
    print(f"  各档位覆盖（入库后）：{ {REGISTER_LABEL[Register(k)]: v for k, v in by_reg.items()} }")
    print("  ↓ 据每个档位下的真实台词，为 register_styles 写「她在这种场景下是什么样」")

    # 候选金句·语音
    print("\n=== 候选金句·语音（self-contained，最适合做 few-shot）===")
    for v in op.get("voices", [])[:10]:
        print(f"  ({v.get('title','')}) {v.get('text','')}")

    # 候选金句·剧情，按档位分组（便于人工为每个档位挑示范）
    print("\n=== 候选金句·剧情（按语气档位分组）===")
    story_chunks = [c for c in chunks if c["type"] == "story"]
    for reg in Register:
        sample = [c["text"] for c in story_chunks
                  if c["register"] == reg.value and 10 <= len(c["text"]) <= 70][:4]
        if sample:
            print(f"\n【{REGISTER_LABEL[reg]}】")
            for t in sample:
                print(f"  - {t}")


if __name__ == "__main__":
    main()
