"""Tests for step executor dispatching."""

from unittest.mock import AsyncMock

import pytest

from doc2md.blackboard.board import Blackboard
from doc2md.config.schema import StepConfig, StepType
from doc2md.pipeline.data_flow import StepInput
from doc2md.pipeline.step_executor import (
    execute_step,
    register_code_step,
    get_code_step,
)
from doc2md.types import (
    AgentConfig,
    InputMode,
    PromptConfig,
    StepResult,
    TokenUsage,
    VLMResponse,
)


def _make_agent_config(name: str = "test_agent") -> AgentConfig:
    return AgentConfig(
        name=name,
        prompt=PromptConfig(system="System", user="User"),
    )


def _mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.execute = AsyncMock(return_value=StepResult(
        step_name="extract",
        agent_name="test_agent",
        markdown="# Extracted",
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model_used="gpt-4.1-mini",
    ))
    return engine


class TestExecuteAgentStep:
    async def test_agent_step_with_image(self):
        step = StepConfig(name="extract", type=StepType.AGENT, agent="test_agent")
        step_input = StepInput(images=[b"img1"])
        bb = Blackboard()
        engine = _mock_engine()
        configs = {"test_agent": _make_agent_config()}

        result = await execute_step(step, step_input, bb, engine, configs)

        assert result.markdown == "# Extracted"
        assert bb.step_outputs["extract"] == "# Extracted"

    async def test_agent_step_text_only(self):
        step = StepConfig(
            name="summarize", type=StepType.AGENT, agent="test_agent",
            input=InputMode.PREVIOUS_OUTPUT_ONLY,
        )
        step_input = StepInput(previous_output="Some content")
        bb = Blackboard()
        engine = _mock_engine()
        configs = {"test_agent": _make_agent_config()}

        result = await execute_step(step, step_input, bb, engine, configs)
        assert result.markdown == "# Extracted"

    async def test_agent_not_found_raises(self):
        step = StepConfig(name="x", type=StepType.AGENT, agent="missing")
        bb = Blackboard()
        engine = _mock_engine()

        with pytest.raises(ValueError, match="Agent 'missing' not found"):
            await execute_step(step, StepInput(), bb, engine, {})


class TestCodeStep:
    def test_register_and_get(self):
        @register_code_step("test_fn")
        def test_fn(text: str) -> str:
            return text.upper()

        assert get_code_step("test_fn") is not None

    async def test_code_step_execution(self):
        @register_code_step("upper_fn")
        def upper_fn(text: str) -> str:
            return text.upper()

        step = StepConfig(name="transform", type=StepType.CODE, function="upper_fn")
        step_input = StepInput(previous_output="hello")
        bb = Blackboard()
        engine = _mock_engine()

        result = await execute_step(step, step_input, bb, engine, {})
        assert result.markdown == "HELLO"
        assert result.agent_name == "code:upper_fn"

    async def test_code_step_missing_function_raises(self):
        step = StepConfig(name="bad", type=StepType.CODE)
        with pytest.raises(ValueError, match="missing 'function'"):
            await execute_step(step, StepInput(), Blackboard(), _mock_engine(), {})

    async def test_code_step_unknown_function_raises(self):
        step = StepConfig(name="bad", type=StepType.CODE, function="nonexistent_fn")
        with pytest.raises(ValueError, match="not registered"):
            await execute_step(step, StepInput(), Blackboard(), _mock_engine(), {})
