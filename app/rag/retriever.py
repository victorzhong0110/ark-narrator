"""轻量词法检索器。

中文无需分词依赖：用字符 bigram 重合度（Jaccard）打分，叠加「角色匹配」与
「标签命中」加权。纯 Python、零重依赖、可在 24GB Mac 上零成本跑。

接口是 Protocol——日后要换成向量检索（embedding）直接实现 Retriever 即可，
编排层不用改。
"""

from __future__ import annotations

from typing import Protocol

from app.rag.store import LoreChunk

_CHAR_BOOST = 0.35   # chunk 关联角色 == 当前角色时的加分
_TAG_BOOST = 0.15    # 每个命中标签的加分（上限 1 个计入）


def _bigrams(text: str) -> set[str]:
    s = "".join(ch for ch in text if not ch.isspace())
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class Retriever(Protocol):
    def retrieve(self, query: str, character: str, k: int) -> list[LoreChunk]:
        ...

    def register_exemplars(
        self, character: str, register: str, query: str, k: int
    ) -> list[LoreChunk]:
        ...


class LexicalRetriever:
    def __init__(self, chunks: list[LoreChunk]):
        self._chunks = chunks
        self._bigrams = [_bigrams(c.text) for c in chunks]

    def retrieve(self, query: str, character: str, k: int = 3) -> list[LoreChunk]:
        if not self._chunks or k <= 0:
            return []
        qb = _bigrams(query)
        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(self._chunks):
            score = _jaccard(qb, self._bigrams[i])
            if character and chunk.character == character:
                score += _CHAR_BOOST
            if chunk.tags and any(t and t in query for t in chunk.tags):
                score += _TAG_BOOST
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._chunks[i] for _, i in scored[:k]]

    def register_exemplars(
        self, character: str, register: str, query: str = "", k: int = 3
    ) -> list[LoreChunk]:
        """取某干员在指定语气档位下的真实剧情台词，按与 query 的相关度排序。

        档位内无料时退回该干员任意剧情台词（保证有示范）。
        """
        if k <= 0:
            return []
        in_reg = [
            (i, c) for i, c in enumerate(self._chunks)
            if c.character == character and c.type == "story" and c.register == register
        ]
        pool = in_reg or [
            (i, c) for i, c in enumerate(self._chunks)
            if c.character == character and c.type == "story"
        ]
        if not pool:
            return []
        qb = _bigrams(query)
        pool.sort(key=lambda ic: _jaccard(qb, self._bigrams[ic[0]]), reverse=True)
        return [c for _, c in pool[:k]]


def render_lore_block(chunks: list[LoreChunk]) -> str:
    """把检索到的 lore 渲染成注入 system prompt 的文本块。"""
    return "\n".join(f"- {c.text}" for c in chunks)
