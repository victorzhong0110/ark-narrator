"""从剧情脚本里按说话人挖某干员的台词（带语境）。

剧情是「她在真实情境里说话」——比档案/语音更能体现角色声音。脚本格式：
  [name="能天使"]   台词文本

来源：Kengxxiao/ArknightsGameData。文件按 flat 名缓存到 data/raw/stories/，重跑免下载。

用法：
  python scripts/mine_stories.py 能天使                 # 扫全部剧情节点
  python scripts/mine_stories.py 能天使 --acts act5d0   # 只扫指定活动（快）
  python scripts/mine_stories.py 能天使 --limit 200      # 限制节点数
输出：data/raw/stories_<name>.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

BASE = ("https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData"
        "/master/zh_CN/gamedata")
RAW = Path("data/raw")
STORY_CACHE = RAW / "stories"
STORY_CACHE.mkdir(parents=True, exist_ok=True)

_LINE_RE = re.compile(r'^\[name="(?P<who>[^"]+)"\]\s*(?P<text>.+?)\s*$')
_TAG_RE = re.compile(r"<[^>]+>|\\n")


def _story_review() -> dict:
    cache = RAW / "story_review_table.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    resp = requests.get(f"{BASE}/excel/story_review_table.json", timeout=60)
    resp.raise_for_status()
    cache.write_text(resp.text, encoding="utf-8")
    return resp.json()


def _fetch_story(story_info: str) -> str | None:
    safe = story_info.replace("/", "_").replace("info_", "") + ".txt"
    cache = STORY_CACHE / safe
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    rel = "story/" + (story_info[5:] if story_info.startswith("info/") else story_info) + ".txt"
    try:
        resp = requests.get(f"{BASE}/{rel}", timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        cache.write_text(resp.text, encoding="utf-8")
        time.sleep(0.03)
        return resp.text
    except Exception:  # noqa: BLE001
        return None


def _collect_nodes(sr: dict, acts: list[str] | None) -> list[dict]:
    nodes = []
    for cid, ch in sr.items():
        if acts and cid not in acts:
            continue
        for n in ch.get("infoUnlockDatas", []):
            si = n.get("storyInfo", "")
            if si:
                nodes.append({"act": cid, "act_name": ch.get("name", cid),
                              "story_name": n.get("storyName", ""), "story_info": si})
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--acts", default="", help="逗号分隔的 act id；留空=全扫")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    aliases = {args.name}
    sr = _story_review()
    acts = [a.strip() for a in args.acts.split(",") if a.strip()] or None
    nodes = _collect_nodes(sr, acts)
    if args.limit:
        nodes = nodes[: args.limit]
    print(f"扫描 {len(nodes)} 个剧情节点，找「{args.name}」的台词…")

    lines: list[dict] = []
    seen: set[str] = set()
    appeared_in: set[str] = set()
    for i, node in enumerate(nodes):
        txt = _fetch_story(node["story_info"])
        if not txt:
            continue
        if args.name not in txt:
            continue
        appeared_in.add(node["act_name"])
        prev_text = ""           # 上一句的内容
        last_other = ""          # 最近一个「非她」的说话人 = 她说话的对象
        for raw in txt.splitlines():
            m = _LINE_RE.match(raw.strip())
            if not m:
                continue
            who = m.group("who")
            text = _TAG_RE.sub(" ", m.group("text")).strip()
            if who in aliases and len(text) >= 4 and text not in seen:
                seen.add(text)
                lines.append({
                    "text": text,
                    "prev": prev_text,             # 她在回应的那句话（语境）
                    "interlocutor": last_other,     # 她在对谁说（关系/对象）
                    "act": node["act_name"],
                    "story": node["story_name"],
                })
            if who not in aliases:
                last_other = who
            prev_text = f"{who}：{text}"
        if (i + 1) % 50 == 0:
            print(f"  …{i + 1}/{len(nodes)}，已收 {len(lines)} 条")

    out = {"name": args.name, "line_count": len(lines),
           "appeared_in": sorted(appeared_in), "lines": lines}
    path = RAW / f"stories_{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共 {len(lines)} 条台词，出现于 {len(appeared_in)} 个剧情 → {path}")
    print("\n=== 台词样本 ===")
    for ln in lines[:15]:
        print(f"[{ln['act']}] {ln['text'][:90]}")


if __name__ == "__main__":
    main()
