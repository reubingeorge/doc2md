"""Tests for agent execution engine with mocked VLM."""

from unittest.mock import AsyncMock

import pytest

from doc2md.agents.engine import AgentEngine
from doc2md.types import AgentConfig, InputMode, PromptConfig, TokenUsage, VLMResponse


def _make_agent(input_mode: InputMode = InputMode.IMAGE) -> AgentConfig:
    return AgentConfig(
        name="test_agent",
        prompt=PromptConfig(system="Extract text.", user="Extract."),
        input=input_mode,
    )


def _make_vlm_response(content: str = "# Extracted\n\nText.") -> VLMResponse:
    return VLMResponse(
        content=content,
        model="gpt-4.1-mini",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


class TestAgentEngine:
    @pytest.fixture
    def mock_vlm(self):
        client = AsyncMock()
        client.send_request = AsyncMock(return_value=_make_vlm_response())
        return client

    @pytest.fixture
    def engine(self, mock_vlm):
        return AgentEngine(mock_vlm)

    async def test_execute_returns_step_result(self, engine, sample_image_bytes):
        result = await engine.execute(
            agent_config=_make_agent(),
            image_bytes=sample_image_bytes,
        )
        assert result.agent_name == "test_agent"
        assert "# Extracted" in result.markdown
        assert result.token_usage.total_tokens == 150
        assert result.model_used == "gpt-4.1-mini"

    async def test_execute_sends_image_for_image_mode(self, engine, mock_vlm, sample_image_bytes):
        await engine.execute(
            agent_config=_make_agent(InputMode.IMAGE),
            image_bytes=sample_image_bytes,
        )
        call_kwargs = mock_vlm.send_request.call_args.kwargs
        assert call_kwargs["image_b64"] is not None

    async def test_execute_no_image_for_text_only(self, engine, mock_vlm):
        await engine.execute(
            agent_config=_make_agent(InputMode.PREVIOUS_OUTPUT_ONLY),
            previous_output="Some text",
        )
        call_kwargs = mock_vlm.send_request.call_args.kwargs
        assert call_kwargs["image_b64"] is None

    async def test_execute_with_confidence_tag(self, mock_vlm):
        mock_vlm.send_request = AsyncMock(
            return_value=_make_vlm_response("Result [confidence: HIGH]")
        )
        engine = AgentEngine(mock_vlm)
        result = await engine.execute(
            agent_config=_make_agent(),
            image_bytes=b"fake",
        )
        assert result.confidence_level is not None
        assert "[confidence:" not in result.markdown

    async def test_custom_step_name(self, engine, sample_image_bytes):
        result = await engine.execute(
            agent_config=_make_agent(),
            image_bytes=sample_image_bytes,
            step_name="custom_step",
        )
        assert result.step_name == "custom_step"

    async def test_execute_passes_model_params(self, engine, mock_vlm, sample_image_bytes):
        agent = _make_agent()
        agent.model.preferred = "gpt-4.1"
        agent.model.max_tokens = 2048
        agent.model.temperature = 0.5
        await engine.execute(agent_config=agent, image_bytes=sample_image_bytes)
        call_kwargs = mock_vlm.send_request.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1"
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.5
