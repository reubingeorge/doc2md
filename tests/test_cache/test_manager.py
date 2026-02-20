"""Tests for CacheManager (L1 + L2 orchestration)."""

from doc2md.cache.manager import CacheManager
from doc2md.cache.stats import CacheEntry
from doc2md.types import TokenUsage


def _entry(key: str, markdown: str = "test", **kwargs) -> CacheEntry:
    return CacheEntry(key=key, markdown=markdown, **kwargs)


class TestCacheManager:
    def test_store_and_lookup(self, tmp_path):
        mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            entry = _entry("k1", "# Hello", token_usage=TokenUsage(total_tokens=100))
            mgr.store("k1", entry)
            result = mgr.lookup("k1")
            assert result is not None
            assert result.markdown == "# Hello"
        finally:
            mgr.close()

    def test_lookup_miss(self, tmp_path):
        mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            assert mgr.lookup("nonexistent") is None
        finally:
            mgr.close()

    def test_disabled_cache_returns_none(self, tmp_path):
        mgr = CacheManager(enabled=False, disk_path=tmp_path / "cache.db")
        mgr.store("k1", _entry("k1"))
        assert mgr.lookup("k1") is None

    def test_l2_promotion_to_l1(self, tmp_path):
        mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            entry = _entry("k1", "promoted", token_usage=TokenUsage(total_tokens=50))
            mgr.store("k1", entry)
            # Clear L1 only
            mgr._l1.clear()
            # L2 should still have it, and looking it up promotes to L1
            result = mgr.lookup("k1")
            assert result is not None
            assert result.markdown == "promoted"
            # Now it should be in L1
            assert mgr._l1.get("k1") is not None
        finally:
            mgr.close()

    def test_stats_tracks_hits_and_misses(self, tmp_path):
        mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            entry = _entry("k1", token_usage=TokenUsage(total_tokens=100))
            mgr.store("k1", entry)
            mgr.lookup("k1")  # hit
            mgr.lookup("k1")  # hit
            mgr.lookup("k2")  # miss
            stats = mgr.stats()
            assert stats.hits == 2
            assert stats.misses == 1
            assert stats.tokens_saved == 200
        finally:
            mgr.close()

    def test_stats_entries_and_size(self, tmp_path):
        mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            mgr.store("k1", _entry("k1", "x" * 100))
            mgr.store("k2", _entry("k2", "y" * 200))
            stats = mgr.stats()
            # Entries counted from L1 + L2 (both have both)
            assert stats.entries >= 2
            assert stats.size_mb > 0
        finally:
            mgr.close()

    def test_clear(self, tmp_path):
        mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            mgr.store("k1", _entry("k1"))
            mgr.clear()
            assert mgr.lookup("k1") is None
            stats = mgr.stats()
            assert stats.entries == 0
        finally:
            mgr.close()

    def test_invalidate(self, tmp_path):
        mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            mgr.store("k1", _entry("k1", agent_name="a1"))
            mgr.store("k2", _entry("k2", agent_name="a2"))
            count = mgr.invalidate(agent="a1")
            assert count >= 1
            assert mgr.lookup("k1") is None
            assert mgr.lookup("k2") is not None
        finally:
            mgr.close()

    def test_enabled_property(self, tmp_path):
        mgr1 = CacheManager(enabled=True, disk_path=tmp_path / "c1.db")
        mgr2 = CacheManager(enabled=False, disk_path=tmp_path / "c2.db")
        assert mgr1.enabled is True
        assert mgr2.enabled is False
        mgr1.close()

    def test_disabled_store_is_noop(self, tmp_path):
        mgr = CacheManager(enabled=False, disk_path=tmp_path / "cache.db")
        mgr.store("k1", _entry("k1"))
        # Should not store anything
        assert mgr.lookup("k1") is None
