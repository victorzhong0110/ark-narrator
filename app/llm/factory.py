"""后端工厂。按配置选择 mlx / scripted。"""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLMBackend

logger = logging.getLogger(__name__)


def get_backend(settings: Settings) -> LLMBackend:
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
        return APIBackend(settings.model_path, settings.api_base_url, settings.api_key)
    raise ValueError(f"未知后端：{settings.backend}（应为 mlx / api / scripted）")
