"""Tests for parallel step execution with blackboard merge."""

from unittest.mock import AsyncMock

import pytest

from doc2md.blackboard.board import Blackboard
from doc2md.config.schema import StepConfig, StepType
from doc2md.pipeline.data_flow import StepInput
from doc2md.pipeline.step_executor import execute_step
from doc2md.types import (
    AgentConfig,
    PromptConfig,
    StepResult,
    TokenUsage,
)


def _make_agent(name: str) -> AgentConfig:
    return AgentConfig(name=name, prompt=PromptConfig(system="S", user="U"))


class TestParallelExecution:
    async def test_parallel_step_executes_sub_steps(self):
        call_count = 0

        async def _mock_execute(agent_config, **kwargs):
            nonlocal call_count
            call_count += 1
            return StepResult(
                step_name=kwargs.get("step_name", agent_config.name),
                agent_name=agent_config.name,
                markdown=f"Result from {agent_config.name}",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model_used="gpt-4.1-mini",
            )

        engine = AsyncMock()
        engine.execute = _mock_execute

        parallel_step = StepConfig(
            name="enrichment",
            type=StepType.PARALLEL,
            steps=[
                StepConfig(name="summary", type=StepType.AGENT, agent="summarize"),
                StepConfig(name="key_terms", type=StepType.AGENT, agent="key_terms_agent"),
            ],
        )

        configs = {
            "summarize": _make_agent("summarize"),
            "key_terms_agent": _make_agent("key_terms_agent"),
        }

        bb = Blackboard()
        result = await execute_step(
            parallel_step, StepInput(), bb, engine, configs
        )

        assert call_count == 2
        assert "Result from summarize" in result.markdown
        assert "Result from key_terms_agent" in result.markdown

    async def test_parallel_step_merges_blackboard(self):
        async def _mock_execute(agent_config, **kwargs):
            bb = kwargs.get("blackboard")
            if bb and agent_config.name == "agent_a":
                bb.write("document_metadata", "language", "fr", writer="agent_a")
            return StepResult(
                step_name=kwargs.get("step_name", agent_config.name),
                agent_name=agent_config.name,
                markdown=f"From {agent_config.name}",
                model_used="gpt-4.1-mini",
            )

        engine = AsyncMock()
        engine.execute = _mock_execute

        parallel_step = StepConfig(
            name="parallel",
            type=StepType.PARALLEL,
            steps=[
                StepConfig(name="a", type=StepType.AGENT, agent="agent_a"),
                StepConfig(name="b", type=StepType.AGENT, agent="agent_b"),
            ],
        )

        configs = {
            "agent_a": _make_agent("agent_a"),
            "agent_b": _make_agent("agent_b"),
        }

        bb = Blackboard()
        await execute_step(parallel_step, StepInput(), bb, engine, configs)

        # Blackboard should have the merged write from branch A
        assert bb.document_metadata.language == "fr"

    async def test_parallel_no_sub_steps_raises(self):
        step = StepConfig(name="empty", type=StepType.PARALLEL, steps=[])
        with pytest.raises(ValueError, match="no sub-steps"):
            await execute_step(step, StepInput(), Blackboard(), AsyncMock(), {})
