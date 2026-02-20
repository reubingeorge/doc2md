"""Two-tier async concurrency pool for batch document processing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from doc2md.concurrency.rate_limiter import RateLimiter
from doc2md.errors.exceptions import PageLevelError

if TYPE_CHECKING:
    from doc2md.types import ConversionResult

logger = logging.getLogger(__name__)


class ConcurrencyPool:
    """Two-tier async dispatcher: files → pages, with semaphore + rate limiting.

    Tier 1: File workers process documents concurrently.
    Tier 2: Page workers within each file run concurrently (bounded by semaphore).
    """

    def __init__(
        self,
        max_file_workers: int = 5,
        max_page_workers: int = 10,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._max_file_workers = max_file_workers
        self._max_page_workers = max_page_workers
        self._rate_limiter = rate_limiter or RateLimiter()
        self._semaphore = asyncio.Semaphore(max_page_workers)

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    async def process_batch(
        self,
        convert_fn: object,
        file_paths: list[str | Path],
        **kwargs: object,
    ) -> list[ConversionResult]:
        """Process a batch of files concurrently.

        Args:
            convert_fn: Async callable(path, **kwargs) -> ConversionResult.
            file_paths: List of document paths to process.
            **kwargs: Additional args passed to convert_fn.

        Returns list of ConversionResult (one per file, in input order).
        """
        # Limit concurrent files
        file_semaphore = asyncio.Semaphore(self._max_file_workers)

        async def worker(path: str | Path) -> ConversionResult:
            async with file_semaphore:
                # Rate limit at the file level
                await self._rate_limiter.acquire()
                return await convert_fn(path, **kwargs)  # type: ignore[operator]

        tasks = [worker(p) for p in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        from doc2md.types import ConversionResult as CR

        final: list[ConversionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("File %s failed: %s", file_paths[i], result)
                final.append(CR(
                    markdown="",
                    pages_processed=0,
                    pages_failed=[0],
                ))
            else:
                final.append(result)

        return final
