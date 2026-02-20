"""Tests for token-bucket rate limiter."""

from doc2md.concurrency.rate_limiter import RateLimiter


class TestRateLimiter:
    async def test_acquire_within_limit(self):
        limiter = RateLimiter(rpm_limit=100, tpm_limit=100_000)
        wait = await limiter.acquire(estimated_tokens=100)
        assert wait == 0.0  # Should not wait

    async def test_acquire_tracks_requests(self):
        limiter = RateLimiter(rpm_limit=100, tpm_limit=100_000)
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.stats["total_requests"] == 2

    async def test_record_usage(self):
        limiter = RateLimiter()
        limiter.record_usage(100, 50)
        assert limiter.stats["total_tokens_used"] == 150

    async def test_reset(self):
        limiter = RateLimiter(rpm_limit=100, tpm_limit=100_000)
        await limiter.acquire()
        limiter.record_usage(100, 50)
        limiter.reset()
        assert limiter.stats["total_requests"] == 0
        assert limiter.stats["total_tokens_used"] == 0

    async def test_stats(self):
        limiter = RateLimiter(rpm_limit=3500, tpm_limit=100_000)
        stats = limiter.stats
        assert "rpm_available" in stats
        assert "tpm_available" in stats
        assert stats["total_requests"] == 0

    async def test_bucket_deduction(self):
        limiter = RateLimiter(rpm_limit=10, tpm_limit=10000)
        initial_rpm = limiter.stats["rpm_available"]
        await limiter.acquire(estimated_tokens=100)
        assert limiter.stats["rpm_available"] < initial_rpm
