"""RAG 检索器与 lore 加载。"""

from __future__ import annotations

from app.config import ROOT
from app.rag.retriever import LexicalRetriever, render_lore_block
from app.rag.store import load_lore


def test_lore_loads():
    chunks = load_lore(ROOT / "data" / "lore")
    assert chunks, "种子 lore 应能加载"
    assert any(c.character == "凯尔希" for c in chunks)


def test_retrieve_returns_relevant():
    chunks = load_lore(ROOT / "data" / "lore")
    r = LexicalRetriever(chunks)
    hits = r.retrieve("矿石病和源石技艺", "凯尔希", k=3)
    assert hits
    joined = " ".join(c.text for c in hits)
    assert "矿石病" in joined or "源石" in joined


def test_character_boost():
    chunks = load_lore(ROOT / "data" / "lore")
    r = LexicalRetriever(chunks)
    hits = r.retrieve("介绍一下你自己", "能天使", k=1)
    assert hits
    # 角色加权应让能天使相关 chunk 排到最前
    assert hits[0].character in ("能天使", "")


def test_render_lore_block():
    from app.rag.store import LoreChunk
    block = render_lore_block([LoreChunk(text="泰拉有天灾。"), LoreChunk(text="源石供能。")])
    assert "- 泰拉有天灾。" in block
    assert "- 源石供能。" in block


def test_empty_query_safe():
    r = LexicalRetriever([])
    assert r.retrieve("任意", "凯尔希", k=3) == []


def test_operator_kb_tagged_and_loaded():
    # data/lore/operators/*.jsonl 应被递归加载，且带 type/character 标签
    chunks = load_lore(ROOT / "data" / "lore")
    typed = {c.type for c in chunks}
    assert {"world", "archive", "voice", "story"} & typed
    exu = [c for c in chunks if c.character == "能天使"]
    assert any(c.type == "story" for c in exu)
    assert any(c.type == "voice" for c in exu)


def test_story_topic_retrieval():
    chunks = load_lore(ROOT / "data" / "lore")
    r = LexicalRetriever(chunks)
    hits = r.retrieve("叙拉古的黑手党", "能天使", k=3)
    assert hits
    assert any("叙拉古" in h.text for h in hits)
