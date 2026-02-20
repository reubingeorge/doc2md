"""Tests for concurrency pool."""

import asyncio

from doc2md.concurrency.pool import ConcurrencyPool
from doc2md.types import ConversionResult, TokenUsage


class TestConcurrencyPool:
    async def test_process_batch(self):
        """Process a batch of 'files' concurrently."""
        call_count = 0

        async def mock_convert(path, **kwargs):
            nonlocal call_count
            call_count += 1
            return ConversionResult(
                markdown=f"# Converted {path}",
                pages_processed=1,
            )

        pool = ConcurrencyPool(max_file_workers=3)
        results = await pool.process_batch(
            mock_convert, ["file1.png", "file2.png", "file3.png"],
        )

        assert len(results) == 3
        assert call_count == 3
        assert all(r.markdown.startswith("# Converted") for r in results)

    async def test_process_batch_with_failure(self):
        """Failed files should produce empty results, not crash."""
        async def mock_convert(path, **kwargs):
            if "bad" in str(path):
                raise ValueError("bad file")
            return ConversionResult(markdown="# OK", pages_processed=1)

        pool = ConcurrencyPool(max_file_workers=2)
        results = await pool.process_batch(
            mock_convert, ["good.png", "bad.png", "good2.png"],
        )

        assert len(results) == 3
        assert results[0].markdown == "# OK"
        assert results[1].markdown == ""  # Failed
        assert results[2].markdown == "# OK"

    async def test_respects_max_workers(self):
        """Should not exceed max concurrent workers."""
        concurrent = 0
        max_concurrent = 0

        async def mock_convert(path, **kwargs):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1
            return ConversionResult(markdown="ok", pages_processed=1)

        pool = ConcurrencyPool(max_file_workers=2)
        await pool.process_batch(
            mock_convert, [f"f{i}.png" for i in range(6)],
        )

        assert max_concurrent <= 2

    async def test_empty_batch(self):
        async def mock_convert(path, **kwargs):
            return ConversionResult(markdown="", pages_processed=0)

        pool = ConcurrencyPool()
        results = await pool.process_batch(mock_convert, [])
        assert results == []

    async def test_passes_kwargs(self):
        received_kwargs = {}

        async def mock_convert(path, **kwargs):
            received_kwargs.update(kwargs)
            return ConversionResult(markdown="ok", pages_processed=1)

        pool = ConcurrencyPool(max_file_workers=1)
        await pool.process_batch(
            mock_convert, ["f.png"],
            agent="custom", pipeline="receipt",
        )

        assert received_kwargs["agent"] == "custom"
        assert received_kwargs["pipeline"] == "receipt"
