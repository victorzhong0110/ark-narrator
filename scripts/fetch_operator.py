"""抓某干员在官方游戏数据里的真实档案 + 语音台词，用于重建角色卡 / RAG。

来源：Kengxxiao/ArknightsGameData（官方数据公开镜像）
  - character_table.json   基础信息（按名字定位 char_id）
  - handbook_info_table.json  干员档案（种族/出身/性格文本）
  - charword_table.json    语音记录（第一人称台词——角色声音的金标）

用法：python scripts/fetch_operator.py 能天使
输出：data/raw/operator_<name>.json（含 profile 段 + voice 台词列表）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

BASE = ("https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData"
        "/master/zh_CN/gamedata/excel")
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

_TAG_RE = re.compile(r"<[^>]+>")


def _get(name: str) -> dict:
    cache = RAW / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    print(f"下载 {name} …")
    resp = requests.get(f"{BASE}/{name}.json", timeout=60)
    resp.raise_for_status()
    cache.write_text(resp.text, encoding="utf-8")
    return resp.json()


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text or "")
    return text.replace("\\n", " ").strip()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "能天使"

    char_table = _get("character_table")
    char_id = next((cid for cid, v in char_table.items() if v.get("name") == name), None)
    if not char_id:
        print(f"未找到干员：{name}")
        return
    base = char_table[char_id]
    print(f"{name} → {char_id} | 职业={base.get('profession')} | 阵营={base.get('nationId')} {base.get('teamId')}")

    # 档案
    handbook = _get("handbook_info_table")
    stories = []
    dict_ = handbook.get("handbookDict", handbook)
    entry = dict_.get(char_id, {})
    for st in entry.get("storyTextAudio", []):
        for s in st.get("stories", []):
            txt = _clean(s.get("storyText", ""))
            if txt:
                stories.append({"title": st.get("storyTitle", ""), "text": txt})

    # 语音
    charword = _get("charword_table")
    words = charword.get("charWords", charword)
    voices = []
    for wid, w in words.items():
        if w.get("charId") == char_id:
            txt = _clean(w.get("voiceText", ""))
            if txt:
                voices.append({"title": w.get("voiceTitle", ""), "text": txt})

    out = {"name": name, "char_id": char_id,
           "profession": base.get("profession"),
           "itemUsage": _clean(base.get("itemUsage", "")),
           "itemDesc": _clean(base.get("itemDesc", "")),
           "stories": stories, "voices": voices}
    path = RAW / f"operator_{name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n档案 {len(stories)} 段 | 语音 {len(voices)} 条 → {path}")

    print("\n=== 档案（前2段）===")
    for s in stories[:2]:
        print(f"[{s['title']}] {s['text'][:200]}")
    print("\n=== 语音台词（抽样20条）===")
    for v in voices[:20]:
        print(f"[{v['title']}] {v['text']}")


if __name__ == "__main__":
    main()
