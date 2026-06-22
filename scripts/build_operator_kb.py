"""把一个角色的 OperatorIR，汇成共享知识库里的「打标签 chunk」+ 角色卡脚手架。

源无关：数据从 PRTS（明日方舟公开数据）还是公司自家 JSON 进来，这一步完全一样。

用法：
  # 明日方舟参考数据（需先 fetch_operator.py / mine_stories.py）
  python scripts/build_operator_kb.py 能天使 --slug exusiai
  # 公司自家数据（导出成 OperatorIR schema 的 JSON）
  python scripts/build_operator_kb.py 星澪 --source generic --data data/companies/example/lyra.json --slug lyra
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from app.ingest import OperatorIR, get_source
from app.registers import REGISTER_LABEL, Register, classify_register

OUT_DIR = Path("data/lore/operators")

_MIN_STORY_LEN = 8       # 太短的剧情台词（语境依赖强）不进库
_PER_REGISTER = 40       # 每个语气档位最多保留多少条（均衡，防日常档挤掉稀薄档）


def build_chunks(ir: OperatorIR) -> list[dict]:
    """OperatorIR → 打标签的知识库 chunk 列表。"""
    chunks: list[dict] = []

    # 档案 → type=archive
    for fact in ir.profile_facts:
        if len(fact) >= _MIN_STORY_LEN:
            chunks.append({"text": fact, "character": ir.name, "type": "archive",
                           "tags": [ir.name], "source": "archive"})

    # 语音 → type=voice（自成一句的风格锚）
    for v in ir.voice_lines:
        if len(v.text) >= 4:
            chunks.append({"text": v.text, "character": ir.name, "type": "voice",
                           "tags": [ir.name, v.title], "source": "voice"})

    # 剧情 → type=story：按语气档位分桶，每档位均衡保留
    story = [s for s in ir.story_lines if len(s.text) >= _MIN_STORY_LEN]
    buckets: dict[str, list] = defaultdict(list)
    for s in story:
        reg = classify_register(f"{s.prev} {s.text}")   # 回应句+她的话，判定更准
        buckets[reg.value].append(s)
    for reg_val, items in buckets.items():
        items.sort(key=lambda x: (0 if 12 <= len(x.text) <= 80 else 1, -len(x.text)))
        for s in items[:_PER_REGISTER]:
            chunks.append({"text": s.text, "character": ir.name, "type": "story",
                           "tags": [ir.name, s.scene], "source": "story",
                           "register": reg_val, "interlocutor": s.interlocutor})
    return chunks


def write_jsonl(chunks: list[dict], slug: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{slug}.jsonl"
    import json
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out


def _print_candidates(ir: OperatorIR, chunks: list[dict]) -> None:
    by_type = Counter(c["type"] for c in chunks)
    by_reg = Counter(c["register"] for c in chunks if c["type"] == "story")
    print(f"  type 分布：{dict(by_type)}")
    print(f"  各档位覆盖（入库后）：{ {REGISTER_LABEL[Register(k)]: v for k, v in by_reg.items()} }")
    print("  ↓ 据每个档位下的真实台词，为 register_styles 写「她在这种场景下是什么样」")

    print("\n=== 候选金句·语音（self-contained，最适合 few-shot）===")
    for v in ir.voice_lines[:10]:
        print(f"  ({v.title}) {v.text}")
    print("\n=== 候选金句·剧情（按语气档位分组）===")
    story_chunks = [c for c in chunks if c["type"] == "story"]
    for reg in Register:
        sample = [c["text"] for c in story_chunks
                  if c["register"] == reg.value and 10 <= len(c["text"]) <= 70][:4]
        if sample:
            print(f"\n【{REGISTER_LABEL[reg]}】")
            for t in sample:
                print(f"  - {t}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--slug", required=True, help="输出文件名（英文），如 exusiai")
    ap.add_argument("--source", default="prts", choices=["prts", "generic"])
    ap.add_argument("--data", default="", help="generic 源的 JSON 文件/目录")
    args = ap.parse_args()

    source = get_source(args.source, data_path=Path(args.data) if args.data else None)
    ir = source.fetch(args.name)
    print(f"接入：{ir.summary()}（源={args.source}）")

    chunks = build_chunks(ir)
    out = write_jsonl(chunks, args.slug)
    print(f"写入 {len(chunks)} 个 chunk → {out}")
    _print_candidates(ir, chunks)


if __name__ == "__main__":
    main()
