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
    raise ValueError(f"未知后端：{settings.backend}（应为 mlx 或 scripted）")
