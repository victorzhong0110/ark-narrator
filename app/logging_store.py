"""审计日志（合规要求：输入输出 + 拦截记录可追溯）。

逐行 JSONL 追加写。仅记录运营所需字段；matched 特征只留类别级，不回显敏感原文。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("无法创建日志目录 %s：%s", path.parent, exc)

    def record(self, event: dict) -> None:
        event.setdefault("ts", time.time())
        line = json.dumps(event, ensure_ascii=False)
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            logger.warning("写审计日志失败：%s", exc)
