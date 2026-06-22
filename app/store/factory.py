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
        sentinels = [s.strip() for s in settings.redis_sentinels.split(",") if s.strip()]
        if sentinels:
            logger.info("使用 RedisStore（Sentinel HA）：%s master=%s", sentinels, settings.redis_master)
            return RedisStore(sentinels=sentinels, master=settings.redis_master)
        logger.info("使用 RedisStore：%s", settings.redis_url)
        return RedisStore(settings.redis_url)
    raise ValueError(f"未知存储：{settings.store}（应为 memory / sqlite / redis）")
