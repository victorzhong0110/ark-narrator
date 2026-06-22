"""测试夹具：用脚本化后端 + 仓库内种子数据装配各层，无需任何模型权重。"""

from __future__ import annotations

import pytest

from app.build import _system_markers
from app.characters.cards import load_characters
from app.config import ROOT, Settings
from app.guard import rules
from app.guard.cloud_audit import DisabledAuditor
from app.guard.input_guard import InputGuard
from app.guard.output_guard import OutputGuard
from app.logging_store import AuditLog
from app.orchestrator import DialogueOrchestrator
from app.rag.retriever import LexicalRetriever
from app.rag.store import load_lore


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(backend="scripted", audit_log_path=tmp_path / "audit.jsonl")


@pytest.fixture
def t3_terms() -> tuple[str, ...]:
    return rules.load_t3_terms(ROOT / "data" / "guard" / "politics_t3.txt")


@pytest.fixture
def characters(settings):
    cards = load_characters(settings.characters_dir)
    assert cards, "种子角色卡应能加载"
    return cards


@pytest.fixture
def input_guard(settings, t3_terms) -> InputGuard:
    return InputGuard(settings, t3_terms=t3_terms)


@pytest.fixture
def output_guard(settings, t3_terms) -> OutputGuard:
    from app.world import DEFAULT_WORLD
    return OutputGuard(
        settings, system_markers=_system_markers(DEFAULT_WORLD), t3_terms=t3_terms,
        cloud_auditor=DisabledAuditor(),
    )


@pytest.fixture
def make_orchestrator(settings, characters, input_guard, output_guard, t3_terms):
    """返回一个工厂：传入 responder，得到用该脚本后端装配的编排器。"""
    from app.llm.scripted_backend import ScriptedBackend

    def _factory(responder=None):
        chunks = load_lore(settings.lore_dir)
        retriever = LexicalRetriever(chunks) if chunks else None
        backend = ScriptedBackend(responder)
        audit = AuditLog(settings.audit_log_path)
        return DialogueOrchestrator(
            settings, backend, retriever, characters,
            input_guard, output_guard, audit,
        )

    return _factory


# T3 测试哨兵（与 data/guard/politics_t3.txt 中保持一致）
T3_SENTINEL = "__T3_TEST_SENTINEL__"
