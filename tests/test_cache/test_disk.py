"""Tests for SQLite disk cache."""

import time

from doc2md.cache.disk import DiskCache
from doc2md.cache.stats import CacheEntry
from doc2md.types import TokenUsage


def _entry(key: str, markdown: str = "test", **kwargs) -> CacheEntry:
    return CacheEntry(key=key, markdown=markdown, **kwargs)


class TestDiskCache:
    def test_get_set(self, tmp_path):
        db_path = tmp_path / "cache.db"
        cache = DiskCache(db_path=db_path)
        try:
            entry = _entry("k1", "# Hello")
            cache.set("k1", entry)
            result = cache.get("k1")
            assert result is not None
            assert result.markdown == "# Hello"
        finally:
            cache.close()

    def test_get_miss(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            assert cache.get("nonexistent") is None
        finally:
            cache.close()

    def test_expired_entry_returns_none(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            entry = _entry("k1", created_at=time.time() - 100, ttl_seconds=1)
            cache.set("k1", entry)
            assert cache.get("k1") is None
        finally:
            cache.close()

    def test_clear(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            cache.set("k1", _entry("k1"))
            cache.set("k2", _entry("k2"))
            cache.clear()
            assert cache.entry_count == 0
        finally:
            cache.close()

    def test_entry_count(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            assert cache.entry_count == 0
            cache.set("k1", _entry("k1"))
            assert cache.entry_count == 1
            cache.set("k2", _entry("k2"))
            assert cache.entry_count == 2
        finally:
            cache.close()

    def test_size_mb(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            cache.set("k1", _entry("k1", "x" * 1000))
            assert cache.size_mb > 0
        finally:
            cache.close()

    def test_invalidate_by_agent(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            cache.set("k1", _entry("k1", agent_name="agent_a"))
            cache.set("k2", _entry("k2", agent_name="agent_b"))
            count = cache.invalidate(agent="agent_a")
            assert count == 1
            assert cache.get("k1") is None
            assert cache.get("k2") is not None
        finally:
            cache.close()

    def test_invalidate_by_pipeline(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            cache.set("k1", _entry("k1", pipeline_name="pipe1"))
            cache.set("k2", _entry("k2", pipeline_name="pipe2"))
            count = cache.invalidate(pipeline="pipe1")
            assert count == 1
        finally:
            cache.close()

    def test_invalidate_no_filter_returns_zero(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            cache.set("k1", _entry("k1"))
            count = cache.invalidate()
            assert count == 0
        finally:
            cache.close()

    def test_overwrite_existing_key(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            cache.set("k1", _entry("k1", "first"))
            cache.set("k1", _entry("k1", "second"))
            result = cache.get("k1")
            assert result.markdown == "second"
            assert cache.entry_count == 1
        finally:
            cache.close()

    def test_preserves_token_usage(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            entry = _entry(
                "k1", "md",
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                model_used="gpt-4.1",
            )
            cache.set("k1", entry)
            result = cache.get("k1")
            assert result.token_usage.prompt_tokens == 100
            assert result.token_usage.completion_tokens == 50
            assert result.token_usage.total_tokens == 150
            assert result.model_used == "gpt-4.1"
        finally:
            cache.close()

    def test_preserves_blackboard_writes(self, tmp_path):
        cache = DiskCache(db_path=tmp_path / "cache.db")
        try:
            writes = {"page_observations": {"1": {"quality": 0.9}}}
            entry = _entry("k1", blackboard_writes=writes)
            cache.set("k1", entry)
            result = cache.get("k1")
            assert result.blackboard_writes == writes
        finally:
            cache.close()

    def test_lru_eviction(self, tmp_path):
        # Very small max size to trigger eviction
        cache = DiskCache(db_path=tmp_path / "cache.db", max_size_mb=0.0001)
        try:
            cache.set("k1", _entry("k1", "a" * 200))
            cache.set("k2", _entry("k2", "b" * 200))
            # k1 should be evicted
            assert cache.get("k1") is None
            assert cache.get("k2") is not None
        finally:
            cache.close()

    def test_persistence(self, tmp_path):
        db_path = tmp_path / "cache.db"
        cache1 = DiskCache(db_path=db_path)
        cache1.set("k1", _entry("k1", "persistent data"))
        cache1.close()

        # Reopen
        cache2 = DiskCache(db_path=db_path)
        try:
            result = cache2.get("k1")
            assert result is not None
            assert result.markdown == "persistent data"
        finally:
            cache2.close()
