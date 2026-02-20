"""L2 disk cache backed by SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from doc2md.cache.stats import CacheEntry
from doc2md.types import TokenUsage

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE_MB = 5000
_DEFAULT_DB_PATH = Path.home() / ".doc2md" / "cache.db"


class DiskCache:
    """SQLite-backed persistent cache with TTL and LRU eviction."""

    def __init__(
        self,
        db_path: Path | None = None,
        max_size_mb: float = _DEFAULT_MAX_SIZE_MB,
    ) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def get(self, key: str) -> CacheEntry | None:
        row = self._conn.execute(
            "SELECT * FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        entry = self._row_to_entry(row)
        if entry.is_expired:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        # Update last_accessed for LRU
        self._conn.execute(
            "UPDATE cache SET last_accessed = ? WHERE key = ?",
            (time.time(), key),
        )
        self._conn.commit()
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        self._evict_if_needed(entry.size_bytes)
        self._conn.execute(
            """INSERT OR REPLACE INTO cache
               (key, created_at, ttl_seconds, last_accessed,
                pipeline_name, step_name, agent_name, agent_version,
                markdown, blackboard_writes, confidence,
                prompt_tokens, completion_tokens, total_tokens,
                model_used, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key, entry.created_at, entry.ttl_seconds, time.time(),
                entry.pipeline_name, entry.step_name, entry.agent_name,
                entry.agent_version, entry.markdown,
                json.dumps(entry.blackboard_writes),
                entry.confidence,
                entry.token_usage.prompt_tokens,
                entry.token_usage.completion_tokens,
                entry.token_usage.total_tokens,
                entry.model_used, entry.size_bytes,
            ),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def invalidate(
        self,
        pipeline: str | None = None,
        agent: str | None = None,
        step: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[str] = []
        if pipeline:
            conditions.append("pipeline_name = ?")
            params.append(pipeline)
        if agent:
            conditions.append("agent_name = ?")
            params.append(agent)
        if step:
            conditions.append("step_name = ?")
            params.append(step)

        if not conditions:
            return 0

        where = " AND ".join(conditions)
        cursor = self._conn.execute(f"DELETE FROM cache WHERE {where}", params)
        self._conn.commit()
        return cursor.rowcount

    @property
    def entry_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return row[0]

    @property
    def size_mb(self) -> float:
        row = self._conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM cache").fetchone()
        return row[0] / (1024 * 1024)

    def close(self) -> None:
        self._conn.close()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                created_at REAL,
                ttl_seconds REAL,
                last_accessed REAL,
                pipeline_name TEXT,
                step_name TEXT,
                agent_name TEXT,
                agent_version TEXT,
                markdown TEXT,
                blackboard_writes TEXT,
                confidence REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                model_used TEXT,
                size_bytes INTEGER
            )
        """)
        self._conn.commit()

    def _evict_if_needed(self, new_entry_size: int) -> None:
        # First remove expired entries
        self._conn.execute(
            "DELETE FROM cache WHERE created_at + ttl_seconds < ?",
            (time.time(),),
        )
        self._conn.commit()

        # Then LRU evict if still over limit
        while True:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM cache"
            ).fetchone()
            current_size = row[0]
            if current_size + new_entry_size <= self._max_size_bytes:
                break
            # Remove oldest accessed
            oldest = self._conn.execute(
                "SELECT key FROM cache ORDER BY last_accessed ASC LIMIT 1"
            ).fetchone()
            if oldest is None:
                break
            self._conn.execute("DELETE FROM cache WHERE key = ?", (oldest[0],))
            self._conn.commit()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
        bb_writes = {}
        try:
            bb_writes = json.loads(row["blackboard_writes"]) if row["blackboard_writes"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return CacheEntry(
            key=row["key"],
            created_at=row["created_at"],
            ttl_seconds=row["ttl_seconds"],
            pipeline_name=row["pipeline_name"] or "",
            step_name=row["step_name"] or "",
            agent_name=row["agent_name"] or "",
            agent_version=row["agent_version"] or "",
            markdown=row["markdown"] or "",
            blackboard_writes=bb_writes,
            confidence=row["confidence"],
            token_usage=TokenUsage(
                prompt_tokens=row["prompt_tokens"] or 0,
                completion_tokens=row["completion_tokens"] or 0,
                total_tokens=row["total_tokens"] or 0,
            ),
            model_used=row["model_used"] or "",
        )
