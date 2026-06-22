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
from app.memory import MemoryManager
from app.orchestrator import DialogueOrchestrator
from app.rag.retriever import LexicalRetriever
from app.rag.store import load_lore
from app.scene import get_scene_tagger
from app.store import get_store
from app.world import WorldProfile, load_world

logger = logging.getLogger(__name__)


def _system_markers(world: WorldProfile) -> tuple[str, ...]:
    """system prompt 中绝不应出现在模型输出里的标记（泄露检测用），按 IP 派生。"""
    return (
        "[扮演规则",
        "扮演规则 · 必须严格遵守",
        f"你正在扮演{world.work}{world.role_term}",
    )


def build_orchestrator(settings: Settings | None = None) -> DialogueOrchestrator:
    s = settings or load_settings()
    world = load_world(s.world_config)

    characters = load_characters(s.characters_dir)
    if not characters:
        logger.warning("未加载到任何角色卡，请检查 %s", s.characters_dir)

    chunks = load_lore(s.lore_dir) if s.rag_enabled else []
    retriever = LexicalRetriever(chunks) if chunks else None

    t3_terms = rules.load_t3_terms(s.lore_dir.parent / "guard" / "politics_t3.txt")

    from app.store.resilient import ResilientStore
    store = ResilientStore(get_store(s))    # 存储抖动不 500，优雅降级
    backend = get_backend(s, store=store)   # pool 后端靠 store 做节点服务发现

    cloud = get_cloud_auditor(s) or DisabledAuditor()
    in_guard = InputGuard(s, t3_terms=t3_terms, store=store)
    out_guard = OutputGuard(
        s, system_markers=_system_markers(world), t3_terms=t3_terms, cloud_auditor=cloud
    )
    audit = AuditLog(s.audit_log_path)
    scene_tagger = get_scene_tagger(s, backend)
    memory = MemoryManager(
        backend, store, every=s.memory_every, max_chars=s.memory_max_chars
    )

    logger.info(
        "编排就绪：IP=%s 后端=%s 存储=%s 角色=%d lore=%d 云端审核=%s 场景判定=%s 记忆/%d轮",
        world.work, backend.label, s.store, len(characters), len(chunks),
        getattr(cloud, "name", "?"), s.scene_tagger, s.memory_every,
    )
    return DialogueOrchestrator(
        s, backend, retriever, characters, in_guard, out_guard, audit,
        scene_tagger=scene_tagger, world=world, store=store, memory_manager=memory,
    )
