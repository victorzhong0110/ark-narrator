"""装配：从 Settings 一把组装出可用的 DialogueOrchestrator。"""

from __future__ import annotations

import logging

from app.characters.cards import load_characters
from app.config import Settings, load_settings
from app.guard import rules
from app.guard.cloud_audit import DisabledAuditor, get_cloud_auditor
from app.guard.input_guard import InputGuard
from app.guard.output_guard import OutputGuard
from app.llm.factory import get_backend
from app.logging_store import AuditLog
from app.orchestrator import DialogueOrchestrator
from app.rag.retriever import LexicalRetriever
from app.rag.store import load_lore
from app.scene import get_scene_tagger

logger = logging.getLogger(__name__)

# 出现在 system prompt 中、绝不应出现在模型输出里的标记（泄露检测用）
SYSTEM_MARKERS: tuple[str, ...] = (
    "[扮演规则",
    "你正在扮演明日方舟干员",
    "扮演规则 · 必须严格遵守",
)


def build_orchestrator(settings: Settings | None = None) -> DialogueOrchestrator:
    s = settings or load_settings()

    characters = load_characters(s.characters_dir)
    if not characters:
        logger.warning("未加载到任何角色卡，请检查 %s", s.characters_dir)

    chunks = load_lore(s.lore_dir) if s.rag_enabled else []
    retriever = LexicalRetriever(chunks) if chunks else None

    t3_terms = rules.load_t3_terms(s.lore_dir.parent / "guard" / "politics_t3.txt")

    backend = get_backend(s)

    cloud = get_cloud_auditor(s) or DisabledAuditor()
    in_guard = InputGuard(s, t3_terms=t3_terms)
    out_guard = OutputGuard(
        s, system_markers=SYSTEM_MARKERS, t3_terms=t3_terms, cloud_auditor=cloud
    )
    audit = AuditLog(s.audit_log_path)
    scene_tagger = get_scene_tagger(s, backend)

    logger.info(
        "编排就绪：后端=%s 角色=%d lore=%d T3词=%d 云端审核=%s RAG=%s 场景判定=%s",
        backend.label, len(characters), len(chunks), len(t3_terms),
        getattr(cloud, "name", "?"), s.rag_enabled, s.scene_tagger,
    )
    return DialogueOrchestrator(
        s, backend, retriever, characters, in_guard, out_guard, audit,
        scene_tagger=scene_tagger,
    )
