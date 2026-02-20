"""L1 in-memory LRU cache."""

from __future__ import annotations

from collections import OrderedDict

from doc2md.cache.stats import CacheEntry

_DEFAULT_MAX_SIZE_MB = 500


class MemoryCache:
    """In-memory LRU cache with size-based eviction."""

    def __init__(self, max_size_mb: float = _DEFAULT_MAX_SIZE_MB) -> None:
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._current_size_bytes = 0

    def get(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            self._remove(key)
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        if key in self._store:
            self._remove(key)
        entry_size = entry.size_bytes
        # Evict until there's room
        while self._current_size_bytes + entry_size > self._max_size_bytes and self._store:
            self._evict_oldest()
        self._store[key] = entry
        self._current_size_bytes += entry_size

    def clear(self) -> None:
        self._store.clear()
        self._current_size_bytes = 0

    def invalidate(
        self,
        pipeline: str | None = None,
        agent: str | None = None,
        step: str | None = None,
    ) -> int:
        """Remove entries matching the given filters. Returns count deleted."""
        to_remove = [
            key
            for key, entry in self._store.items()
            if _matches_filter(entry, pipeline, agent, step)
        ]
        for key in to_remove:
            self._remove(key)
        return len(to_remove)

    @property
    def size_mb(self) -> float:
        return self._current_size_bytes / (1024 * 1024)

    def __len__(self) -> int:
        return len(self._store)

    def _remove(self, key: str) -> None:
        entry = self._store.pop(key, None)
        if entry:
            self._current_size_bytes -= entry.size_bytes

    def _evict_oldest(self) -> None:
        if self._store:
            key, _ = self._store.popitem(last=False)
            # Recalculate; entry already removed from dict
            self._recalculate_size()

    def _recalculate_size(self) -> None:
        self._current_size_bytes = sum(e.size_bytes for e in self._store.values())


def _matches_filter(
    entry: CacheEntry,
    pipeline: str | None,
    agent: str | None,
    step: str | None,
) -> bool:
    if pipeline and entry.pipeline_name != pipeline:
        return False
    if agent and entry.agent_name != agent:
        return False
    return not (step and entry.step_name != step)
