"""存储工厂：按配置选 memory / sqlite / redis。"""

from __future__ import annotations

import logging
import os

from app.config import ROOT, Settings
from app.store.base import Store

logger = logging.getLogger(__name__)


def get_store(settings: Settings) -> Store:
    kind = settings.store.lower()
    if kind == "memory":
        from app.store.memory import InMemoryStore
        return InMemoryStore()
    if kind == "sqlite":
        from pathlib import Path
        from app.store.sqlite import SQLiteStore
        path = ROOT / os.getenv("ARK_STORE_PATH", "data/state.sqlite")
        logger.info("使用 SQLiteStore：%s", path)
        return SQLiteStore(Path(path))
    if kind == "redis":
        from app.store.redis import RedisStore
        url = os.getenv("ARK_REDIS_URL", "redis://localhost:6379/0")
        logger.info("使用 RedisStore：%s", url)
        return RedisStore(url)
    raise ValueError(f"未知存储：{settings.store}（应为 memory / sqlite / redis）")
