"""SQLiteStore：单机持久（budget 档）。重启不丢，零外部依赖。"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

from app.store.base import Turn
from app.store.memory import fixed_window_allow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history(
  session_id TEXT, role TEXT, content TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS idx_hist ON history(session_id);
CREATE TABLE IF NOT EXISTS memory(
  user_id TEXT, character TEXT, text TEXT, PRIMARY KEY(user_id, character));
CREATE TABLE IF NOT EXISTS pair_turns(
  user_id TEXT, character TEXT, n INTEGER, PRIMARY KEY(user_id, character));
CREATE TABLE IF NOT EXISTS counters(
  key TEXT PRIMARY KEY, value INTEGER, expire_at REAL);
CREATE TABLE IF NOT EXISTS nodes(
  grp TEXT, addr TEXT, expire_at REAL, PRIMARY KEY(grp, addr));
"""


class SQLiteStore:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self._now = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def append_turn(self, session_id: str, role: str, content: str,
                    cap: int = 0, ttl: float = 0.0) -> None:
        with self._lock:
            self._db.execute("INSERT INTO history VALUES(?,?,?,?)",
                             (session_id, role, content, self._now()))
            if cap > 0:     # 写时裁剪：只保留该会话最近 cap 条
                self._db.execute(
                    "DELETE FROM history WHERE session_id=? AND rowid NOT IN "
                    "(SELECT rowid FROM history WHERE session_id=? ORDER BY rowid DESC LIMIT ?)",
                    (session_id, session_id, cap))
            self._db.commit()

    def history(self, session_id: str, limit: int = 50) -> list[Turn]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content, ts FROM history WHERE session_id=? "
                "ORDER BY rowid DESC LIMIT ?", (session_id, max(limit, 0) or 1_000_000),
            ).fetchall()
        return [Turn(r[0], r[1], r[2]) for r in reversed(rows)]

    def get_memory(self, user_id: str, character: str) -> str:
        with self._lock:
            row = self._db.execute(
                "SELECT text FROM memory WHERE user_id=? AND character=?",
                (user_id, character)).fetchone()
        return row[0] if row else ""

    def set_memory(self, user_id: str, character: str, text: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO memory VALUES(?,?,?) ON CONFLICT(user_id,character) "
                "DO UPDATE SET text=excluded.text", (user_id, character, text))
            self._db.commit()

    def bump_pair_turns(self, user_id: str, character: str) -> int:
        with self._lock:
            self._db.execute(
                "INSERT INTO pair_turns VALUES(?,?,1) ON CONFLICT(user_id,character) "
                "DO UPDATE SET n=n+1", (user_id, character))
            self._db.commit()
            row = self._db.execute(
                "SELECT n FROM pair_turns WHERE user_id=? AND character=?",
                (user_id, character)).fetchone()
        return row[0] if row else 0

    def _live_value(self, key: str) -> tuple[int, float | None] | None:
        row = self._db.execute(
            "SELECT value, expire_at FROM counters WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value, exp = row
        if exp is not None and self._now() >= exp:
            self._db.execute("DELETE FROM counters WHERE key=?", (key,))
            return None
        return value, exp

    def incr(self, key: str, delta: int = 1, ttl: float | None = None) -> int:
        with self._lock:
            cur = self._live_value(key)
            if cur is None:
                value, exp = 0, (self._now() + ttl) if ttl else None
            else:
                value, exp = cur
            value += delta
            self._db.execute(
                "INSERT INTO counters VALUES(?,?,?) ON CONFLICT(key) "
                "DO UPDATE SET value=excluded.value, expire_at=excluded.expire_at",
                (key, value, exp))
            self._db.commit()
        return value

    def get_int(self, key: str) -> int:
        with self._lock:
            cur = self._live_value(key)
        return cur[0] if cur else 0

    def delete(self, key: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM counters WHERE key=?", (key,))
            self._db.commit()

    def rate_allow(self, user_id: str, limit: int, window_s: float) -> bool:
        return fixed_window_allow(self, user_id, limit, window_s, self._now())

    def register_node(self, group: str, addr: str, ttl: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO nodes VALUES(?,?,?) ON CONFLICT(grp,addr) "
                "DO UPDATE SET expire_at=excluded.expire_at",
                (group, addr, self._now() + ttl))
            self._db.commit()

    def live_nodes(self, group: str) -> list[str]:
        with self._lock:
            self._db.execute("DELETE FROM nodes WHERE expire_at<=?", (self._now(),))
            rows = self._db.execute(
                "SELECT addr FROM nodes WHERE grp=? ORDER BY addr", (group,)).fetchall()
            self._db.commit()
        return [r[0] for r in rows]
