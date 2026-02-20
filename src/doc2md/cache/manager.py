"""Cache manager — orchestrates L1 (memory) and L2 (disk) tiers."""

from __future__ import annotations

import logging
from pathlib import Path

from doc2md.cache.disk import DiskCache
from doc2md.cache.memory import MemoryCache
from doc2md.cache.stats import CacheEntry, CacheStats

logger = logging.getLogger(__name__)


class CacheManager:
    """Two-tier cache: L1 in-memory → L2 on-disk (SQLite)."""

    def __init__(
        self,
        memory_max_mb: float = 500,
        disk_max_mb: float = 5000,
        disk_path: Path | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._l1 = MemoryCache(max_size_mb=memory_max_mb)
        self._l2 = DiskCache(db_path=disk_path, max_size_mb=disk_max_mb) if enabled else None
        self._stats = CacheStats()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def lookup(self, key: str) -> CacheEntry | None:
        """Look up a cache key. L1 first, then L2 (with promotion)."""
        if not self._enabled:
            self._stats.misses += 1
            return None

        # L1
        entry = self._l1.get(key)
        if entry is not None:
            self._stats.hits += 1
            self._stats.tokens_saved += entry.token_usage.total_tokens
            return entry

        # L2
        if self._l2:
            entry = self._l2.get(key)
            if entry is not None:
                # Promote to L1
                self._l1.set(key, entry)
                self._stats.hits += 1
                self._stats.tokens_saved += entry.token_usage.total_tokens
                return entry

        self._stats.misses += 1
        return None

    def store(self, key: str, entry: CacheEntry) -> None:
        """Store in L1 and L2."""
        if not self._enabled:
            return
        self._l1.set(key, entry)
        if self._l2:
            self._l2.set(key, entry)

    def invalidate(
        self,
        pipeline: str | None = None,
        agent: str | None = None,
        step: str | None = None,
    ) -> int:
        """Remove entries matching filters from both tiers."""
        count = self._l1.invalidate(pipeline=pipeline, agent=agent, step=step)
        if self._l2:
            count += self._l2.invalidate(pipeline=pipeline, agent=agent, step=step)
        return count

    def clear(self) -> None:
        """Clear all caches."""
        self._l1.clear()
        if self._l2:
            self._l2.clear()
        self._stats = CacheStats()

    def stats(self) -> CacheStats:
        """Return aggregate cache statistics."""
        l1_size = self._l1.size_mb
        l2_size = self._l2.size_mb if self._l2 else 0.0
        return CacheStats(
            entries=len(self._l1) + (self._l2.entry_count if self._l2 else 0),
            size_mb=l1_size + l2_size,
            hits=self._stats.hits,
            misses=self._stats.misses,
            tokens_saved=self._stats.tokens_saved,
        )

    def close(self) -> None:
        if self._l2:
            self._l2.close()
