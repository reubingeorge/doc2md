"""Tests for cache entry and stats models."""

import time

from doc2md.cache.stats import CacheEntry, CacheStats
from doc2md.types import TokenUsage


class TestCacheEntry:
    def test_defaults(self):
        entry = CacheEntry(key="k1")
        assert entry.pipeline_name == ""
        assert entry.step_name == ""
        assert entry.markdown == ""
        assert entry.ttl_seconds == 7 * 24 * 3600
        assert entry.blackboard_writes == {}
        assert entry.token_usage.total_tokens == 0

    def test_is_expired_false_when_fresh(self):
        entry = CacheEntry(key="k1")
        assert not entry.is_expired

    def test_is_expired_true_when_old(self):
        entry = CacheEntry(key="k1", created_at=time.time() - 999999, ttl_seconds=1)
        assert entry.is_expired

    def test_size_bytes(self):
        entry = CacheEntry(key="k1", markdown="hello world")
        assert entry.size_bytes > 0

    def test_size_bytes_includes_blackboard_writes(self):
        entry_plain = CacheEntry(key="k1", markdown="x")
        entry_bb = CacheEntry(
            key="k2", markdown="x",
            blackboard_writes={"region": {"key": "a long value" * 100}},
        )
        assert entry_bb.size_bytes > entry_plain.size_bytes


class TestCacheStats:
    def test_defaults(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.entries == 0

    def test_hit_rate_zero_when_no_requests(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_hit_rate_all_hits(self):
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0
