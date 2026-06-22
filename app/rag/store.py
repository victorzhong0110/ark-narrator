"""lore 语料存储与加载。

支持两种文件：
- *.jsonl：每行一个 chunk，字段 {text, character?, tags?, source?}
- *.md：可选 YAML frontmatter（character / tags），正文按空行分段成多个 chunk

正式上线可用 data_pipeline/scraper.py 抓 PRTS Wiki 干员档案/语音/剧情扩充本目录。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoreChunk:
    text: str
    character: str = ""               # 关联干员（空=通用世界观）
    tags: tuple[str, ...] = field(default_factory=tuple)
    source: str = ""
    type: str = "world"               # world | archive | voice | story
    register: str = ""                # 语气档位（仅 story：banter/taunt/comfort/solemn/business）
    interlocutor: str = ""            # 这句话说给谁（剧情里的对象）


_FRONTMATTER_RE = "---"


def _parse_md(path: Path) -> list[LoreChunk]:
    raw = path.read_text(encoding="utf-8")
    character, tags = "", ()
    body = raw

    # 解析可选 frontmatter
    if raw.startswith(_FRONTMATTER_RE):
        parts = raw.split(_FRONTMATTER_RE, 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                character = str(meta.get("character", "")).strip()
                t = meta.get("tags")
                if isinstance(t, (list, tuple)):
                    tags = tuple(str(x) for x in t)
                elif isinstance(t, str):
                    tags = (t,)
                body = parts[2]
            except Exception as exc:  # noqa: BLE001
                logger.warning("frontmatter 解析失败 %s：%s", path.name, exc)

    chunks: list[LoreChunk] = []
    for para in body.split("\n\n"):
        text = para.strip()
        # 去掉 markdown 标题井号
        text = text.lstrip("#").strip()
        if len(text) >= 8:
            chunks.append(
                LoreChunk(text=text, character=character, tags=tags, source=path.name)
            )
    return chunks


def _parse_jsonl(path: Path) -> list[LoreChunk]:
    chunks: list[LoreChunk] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            text = str(d.get("text", "")).strip()
            if not text:
                continue
            tags = d.get("tags") or []
            chunks.append(
                LoreChunk(
                    text=text,
                    character=str(d.get("character", "")).strip(),
                    tags=tuple(str(x) for x in tags),
                    source=str(d.get("source", path.name)),
                    type=str(d.get("type", "world")).strip() or "world",
                    register=str(d.get("register", "")).strip(),
                    interlocutor=str(d.get("interlocutor", "")).strip(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("jsonl 第 %d 行解析失败 %s：%s", i + 1, path.name, exc)
    return chunks


def load_lore(lore_dir: Path) -> list[LoreChunk]:
    """递归加载 lore_dir 下所有 *.md 与 *.jsonl，组成单一共享语料。

    按干员分文件（如 operators/exusiai.jsonl）只是便于维护——加载后是一个
    打了 character/type 标签的统一索引，检索时按标签过滤/加权。
    """
    chunks: list[LoreChunk] = []
    if not lore_dir.exists():
        logger.warning("lore 目录不存在：%s", lore_dir)
        return chunks
    for path in sorted(lore_dir.rglob("*.md")):
        chunks.extend(_parse_md(path))
    for path in sorted(lore_dir.rglob("*.jsonl")):
        chunks.extend(_parse_jsonl(path))
    logger.info("已加载 %d 条 lore（来自 %s）", len(chunks), lore_dir)
    return chunks
