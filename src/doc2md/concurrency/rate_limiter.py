"""Token-bucket rate limiter for RPM + TPM against OpenAI limits."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Dual token-bucket rate limiter for requests/min and tokens/min.

    Both RPM and TPM buckets must have capacity before a request proceeds.
    Buckets refill continuously at their respective rates.
    """

    def __init__(
        self,
        rpm_limit: int = 3500,
        tpm_limit: int = 100_000,
    ) -> None:
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit

        # Bucket state
        self._rpm_tokens = float(rpm_limit)
        self._tpm_tokens = float(tpm_limit)
        self._last_refill = time.monotonic()

        self._lock = asyncio.Lock()

        # Stats
        self._total_requests = 0
        self._total_tokens_used = 0
        self._total_wait_seconds = 0.0

    async def acquire(self, estimated_tokens: int = 1000) -> float:
        """Wait until capacity is available, then deduct.

        Returns the time spent waiting (seconds).
        """
        wait_total = 0.0

        async with self._lock:
            while True:
                self._refill()

                # Check both buckets
                if self._rpm_tokens >= 1 and self._tpm_tokens >= estimated_tokens:
                    self._rpm_tokens -= 1
                    self._tpm_tokens -= estimated_tokens
                    self._total_requests += 1
                    break

                # Compute wait time until both buckets have capacity
                rpm_wait = 0.0
                if self._rpm_tokens < 1:
                    rpm_wait = (1 - self._rpm_tokens) / (self._rpm_limit / 60.0)

                tpm_wait = 0.0
                if self._tpm_tokens < estimated_tokens:
                    tpm_wait = (estimated_tokens - self._tpm_tokens) / (self._tpm_limit / 60.0)

                wait_time = max(rpm_wait, tpm_wait, 0.01)
                wait_total += wait_time

                # Release lock during sleep so other coroutines aren't blocked
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()

        self._total_wait_seconds += wait_total
        return wait_total

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record actual token usage after a VLM response (for stats)."""
        self._total_tokens_used += input_tokens + output_tokens

    @property
    def stats(self) -> dict:
        """Return current rate limiter statistics."""
        self._refill()
        return {
            "rpm_available": self._rpm_tokens,
            "tpm_available": self._tpm_tokens,
            "total_requests": self._total_requests,
            "total_tokens_used": self._total_tokens_used,
            "total_wait_seconds": self._total_wait_seconds,
        }

    def reset(self) -> None:
        """Reset all state (for testing)."""
        self._rpm_tokens = float(self._rpm_limit)
        self._tpm_tokens = float(self._tpm_limit)
        self._last_refill = time.monotonic()
        self._total_requests = 0
        self._total_tokens_used = 0
        self._total_wait_seconds = 0.0

    def _refill(self) -> None:
        """Refill buckets based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        self._rpm_tokens = min(
            float(self._rpm_limit),
            self._rpm_tokens + elapsed * (self._rpm_limit / 60.0),
        )
        self._tpm_tokens = min(
            float(self._tpm_limit),
            self._tpm_tokens + elapsed * (self._tpm_limit / 60.0),
        )
