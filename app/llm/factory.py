"""后端工厂。按配置选择 mlx / scripted。"""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLMBackend

logger = logging.getLogger(__name__)


def get_backend(settings: Settings, store=None) -> LLMBackend:
    backend = settings.backend.lower()
    if backend == "scripted":
        from app.llm.scripted_backend import ScriptedBackend
        logger.info("使用 ScriptedBackend（无需模型权重）")
        return ScriptedBackend()
    if backend == "mlx":
        from app.llm.mlx_backend import MLXBackend
        return MLXBackend(settings.model_path, settings.adapter_dir)
    if backend == "api":
        from app.llm.api_backend import APIBackend
        if not settings.api_base_url or not settings.api_key:
            raise ValueError("api 后端需要 ARK_API_BASE_URL 与 ARK_API_KEY")
        logger.info("使用 APIBackend：%s @ %s", settings.model_path, settings.api_base_url)
        return APIBackend(settings.model_path, settings.api_base_url, settings.api_key,
                          disable_thinking=settings.disable_thinking)
    if backend == "pool":
        from app.llm.pool_backend import PooledAPIBackend
        if store is None:
            raise ValueError("pool 后端需要 store（节点服务发现走存储）")
        allowlist = [p.strip() for p in settings.node_allowlist.split(",") if p.strip()]
        logger.info("使用 PooledAPIBackend：组=%s 白名单=%s", settings.node_group, allowlist or "(无)")
        return PooledAPIBackend(store, settings.model_path,
                                group=settings.node_group, api_key=settings.api_key or "EMPTY",
                                disable_thinking=settings.disable_thinking or True,
                                allowlist=allowlist)
    raise ValueError(f"未知后端：{settings.backend}（应为 mlx / api / pool / scripted）")


def get_internal_backend(settings: Settings, main_backend: LLMBackend) -> LLMBackend:
    """内部消费者(场景判定/记忆摘要)的后端：api 模式且配了 internal_api_key 时，
    用单独 token 指向同一网关(各自计量/配额，可限内部机队)；否则复用主后端。"""
    if settings.backend.lower() == "api" and settings.internal_api_key:
        from app.llm.api_backend import APIBackend
        logger.info("内部消费者用独立网关 token（与主调用分账）：%s", settings.api_base_url)
        return APIBackend(settings.model_path, settings.api_base_url, settings.internal_api_key,
                          disable_thinking=settings.disable_thinking)
    return main_backend
