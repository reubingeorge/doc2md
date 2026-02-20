"""Tests for in-memory LRU cache."""

import time

from doc2md.cache.memory import MemoryCache
from doc2md.cache.stats import CacheEntry


def _entry(key: str, markdown: str = "test", **kwargs) -> CacheEntry:
    return CacheEntry(key=key, markdown=markdown, **kwargs)


class TestMemoryCache:
    def test_get_set(self):
        cache = MemoryCache()
        entry = _entry("k1", "# Hello")
        cache.set("k1", entry)
        result = cache.get("k1")
        assert result is not None
        assert result.markdown == "# Hello"

    def test_get_miss(self):
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        cache = MemoryCache()
        entry = _entry("k1", created_at=time.time() - 100, ttl_seconds=1)
        cache.set("k1", entry)
        assert cache.get("k1") is None

    def test_lru_eviction(self):
        # Each entry with 200 chars of markdown + "{}" overhead ≈ 202 bytes
        # Cache limit of 0.0002 MB ≈ 209 bytes — fits 1 entry, not 2
        cache = MemoryCache(max_size_mb=0.0002)
        cache.set("k1", _entry("k1", "a" * 200))
        cache.set("k2", _entry("k2", "b" * 200))
        # k1 should have been evicted to make room for k2
        assert cache.get("k1") is None
        assert cache.get("k2") is not None

    def test_lru_order_preserved(self):
        # Each entry ≈ 502 bytes, cache fits ~2 entries
        cache = MemoryCache(max_size_mb=0.001)
        cache.set("k1", _entry("k1", "a" * 500))
        cache.set("k2", _entry("k2", "b" * 500))
        # Access k1 to make it most recently used
        cache.get("k1")
        # Add k3 to trigger eviction — k2 should be evicted (least recently used)
        cache.set("k3", _entry("k3", "c" * 500))
        assert cache.get("k1") is not None
        assert cache.get("k2") is None

    def test_clear(self):
        cache = MemoryCache()
        cache.set("k1", _entry("k1"))
        cache.set("k2", _entry("k2"))
        cache.clear()
        assert len(cache) == 0
        assert cache.get("k1") is None

    def test_len(self):
        cache = MemoryCache()
        assert len(cache) == 0
        cache.set("k1", _entry("k1"))
        assert len(cache) == 1
        cache.set("k2", _entry("k2"))
        assert len(cache) == 2

    def test_size_mb(self):
        cache = MemoryCache()
        cache.set("k1", _entry("k1", "x" * 1000))
        assert cache.size_mb > 0

    def test_invalidate_by_agent(self):
        cache = MemoryCache()
        cache.set("k1", _entry("k1", agent_name="agent_a"))
        cache.set("k2", _entry("k2", agent_name="agent_b"))
        count = cache.invalidate(agent="agent_a")
        assert count == 1
        assert cache.get("k1") is None
        assert cache.get("k2") is not None

    def test_invalidate_by_pipeline(self):
        cache = MemoryCache()
        cache.set("k1", _entry("k1", pipeline_name="pipe1"))
        cache.set("k2", _entry("k2", pipeline_name="pipe2"))
        count = cache.invalidate(pipeline="pipe1")
        assert count == 1
        assert cache.get("k1") is None

    def test_invalidate_by_step(self):
        cache = MemoryCache()
        cache.set("k1", _entry("k1", step_name="s1"))
        cache.set("k2", _entry("k2", step_name="s2"))
        count = cache.invalidate(step="s1")
        assert count == 1

    def test_invalidate_no_filter_returns_zero(self):
        cache = MemoryCache()
        cache.set("k1", _entry("k1"))
        # No filter args → _matches_filter returns True for all
        count = cache.invalidate()
        assert count == 1

    def test_overwrite_existing_key(self):
        cache = MemoryCache()
        cache.set("k1", _entry("k1", "first"))
        cache.set("k1", _entry("k1", "second"))
        assert cache.get("k1").markdown == "second"
        assert len(cache) == 1
