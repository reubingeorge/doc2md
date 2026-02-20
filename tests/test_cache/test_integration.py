"""Integration test: cache with agent engine pipeline execution."""

from unittest.mock import AsyncMock, patch

from doc2md.agents.engine import AgentEngine
from doc2md.blackboard.board import Blackboard
from doc2md.cache.manager import CacheManager
from doc2md.types import AgentConfig, ModelConfig, PromptConfig, TokenUsage, VLMResponse


def _make_agent_config(name: str = "test_agent") -> AgentConfig:
    return AgentConfig(
        name=name,
        version="1.0",
        prompt=PromptConfig(system="System prompt", user="User prompt"),
        model=ModelConfig(preferred="gpt-4.1-mini"),
    )


def _mock_vlm_response(content: str = "# Output") -> VLMResponse:
    return VLMResponse(
        content=content,
        model="gpt-4.1-mini",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


class TestCacheWithAgentEngine:
    async def test_cache_miss_then_hit(self, tmp_path, sample_image_bytes):
        """First call should be a miss (calls VLM), second call should be a hit."""
        cache_mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=_mock_vlm_response())
            engine = AgentEngine(mock_client)

            config = _make_agent_config()

            # First call — cache miss
            result1 = await engine.execute(
                agent_config=config,
                image_bytes=sample_image_bytes,
                cache_manager=cache_mgr,
                pipeline_name="test_pipe",
                step_name="extract",
            )
            assert result1.cached is False
            assert result1.markdown == "# Output"
            assert mock_client.send_request.call_count == 1

            # Second call — cache hit (same inputs)
            result2 = await engine.execute(
                agent_config=config,
                image_bytes=sample_image_bytes,
                cache_manager=cache_mgr,
                pipeline_name="test_pipe",
                step_name="extract",
            )
            assert result2.cached is True
            assert result2.markdown == "# Output"
            # VLM should NOT have been called again
            assert mock_client.send_request.call_count == 1
        finally:
            cache_mgr.close()

    async def test_different_images_different_cache_keys(self, tmp_path):
        """Different image bytes should produce different cache keys."""
        cache_mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(
                side_effect=[_mock_vlm_response("first"), _mock_vlm_response("second")]
            )
            engine = AgentEngine(mock_client)
            config = _make_agent_config()

            result1 = await engine.execute(
                agent_config=config,
                image_bytes=b"image_a",
                cache_manager=cache_mgr,
                pipeline_name="p",
                step_name="s",
            )
            result2 = await engine.execute(
                agent_config=config,
                image_bytes=b"image_b",
                cache_manager=cache_mgr,
                pipeline_name="p",
                step_name="s",
            )

            assert result1.markdown == "first"
            assert result2.markdown == "second"
            assert mock_client.send_request.call_count == 2
        finally:
            cache_mgr.close()

    async def test_no_cache_manager_always_calls_vlm(self, tmp_path, sample_image_bytes):
        """Without a cache manager, every call goes to VLM."""
        mock_client = AsyncMock()
        mock_client.send_request = AsyncMock(return_value=_mock_vlm_response())
        engine = AgentEngine(mock_client)
        config = _make_agent_config()

        await engine.execute(
            agent_config=config, image_bytes=sample_image_bytes, step_name="s",
        )
        await engine.execute(
            agent_config=config, image_bytes=sample_image_bytes, step_name="s",
        )
        assert mock_client.send_request.call_count == 2

    async def test_cached_result_replays_blackboard_writes(self, tmp_path, sample_image_bytes):
        """Cached results should re-apply blackboard writes."""
        cache_mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            response = _mock_vlm_response(
                "# Output\n\n<blackboard>\npage_observations:\n  1:\n    quality_score: 0.9\n</blackboard>"
            )
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=response)
            engine = AgentEngine(mock_client)
            config = _make_agent_config()

            # First call — writes to blackboard
            bb1 = Blackboard()
            await engine.execute(
                agent_config=config, image_bytes=sample_image_bytes,
                blackboard=bb1, cache_manager=cache_mgr,
                pipeline_name="p", step_name="s",
            )

            # Second call — cached, should replay writes to fresh blackboard
            bb2 = Blackboard()
            result2 = await engine.execute(
                agent_config=config, image_bytes=sample_image_bytes,
                blackboard=bb2, cache_manager=cache_mgr,
                pipeline_name="p", step_name="s",
            )
            assert result2.cached is True
            assert result2.blackboard_writes != {}
        finally:
            cache_mgr.close()

    async def test_cache_stats_after_usage(self, tmp_path, sample_image_bytes):
        """Cache stats should reflect actual usage."""
        cache_mgr = CacheManager(disk_path=tmp_path / "cache.db")
        try:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=_mock_vlm_response())
            engine = AgentEngine(mock_client)
            config = _make_agent_config()

            # Miss
            await engine.execute(
                agent_config=config, image_bytes=sample_image_bytes,
                cache_manager=cache_mgr, pipeline_name="p", step_name="s",
            )
            # Hit
            await engine.execute(
                agent_config=config, image_bytes=sample_image_bytes,
                cache_manager=cache_mgr, pipeline_name="p", step_name="s",
            )

            stats = cache_mgr.stats()
            assert stats.hits == 1
            assert stats.misses == 1
        finally:
            cache_mgr.close()
