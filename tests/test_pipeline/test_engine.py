"""Tests for the full pipeline engine."""

from unittest.mock import AsyncMock

from doc2md.blackboard.board import Blackboard
from doc2md.config.schema import PipelineConfig, StepConfig
from doc2md.pipeline.engine import PipelineEngine
from doc2md.types import (
    AgentConfig,
    InputMode,
    PromptConfig,
    StepResult,
    TokenUsage,
)


def _make_agent(name: str, input_mode: InputMode = InputMode.IMAGE) -> AgentConfig:
    return AgentConfig(
        name=name,
        input=input_mode,
        prompt=PromptConfig(system="S", user="U"),
    )


def _mock_engine():
    engine = AsyncMock()

    async def _execute(agent_config, **kwargs):
        return StepResult(
            step_name=kwargs.get("step_name", agent_config.name),
            agent_name=agent_config.name,
            markdown=f"Output from {agent_config.name}",
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
            model_used="gpt-4.1-mini",
        )

    engine.execute = _execute
    return engine


class TestPipelineEngine:
    async def test_single_step_pipeline(self):
        config = PipelineConfig(
            name="simple",
            steps=[StepConfig(name="extract", agent="generic")],
        )
        agents = {"generic": _make_agent("generic")}
        engine = PipelineEngine(_mock_engine(), agents)

        result = await engine.execute(config, [b"img"])

        assert "Output from generic" in result.markdown
        assert "extract" in result.steps
        assert result.pipeline_name == "simple"

    async def test_two_step_pipeline(self):
        config = PipelineConfig(
            name="extract_validate",
            steps=[
                StepConfig(name="extract", agent="text_extract"),
                StepConfig(
                    name="validate",
                    agent="validator",
                    input=InputMode.IMAGE_AND_PREVIOUS,
                    depends_on=["extract"],
                ),
            ],
        )
        agents = {
            "text_extract": _make_agent("text_extract"),
            "validator": _make_agent("validator", InputMode.IMAGE_AND_PREVIOUS),
        }
        engine = PipelineEngine(_mock_engine(), agents)

        result = await engine.execute(config, [b"img"])

        assert "extract" in result.steps
        assert "validate" in result.steps
        assert result.token_usage.total_tokens == 150  # 75 * 2 steps

    async def test_conditional_step_skipped(self):
        config = PipelineConfig(
            name="conditional",
            steps=[
                StepConfig(name="extract", agent="generic", depends_on=[]),
                StepConfig(
                    name="handwriting",
                    agent="handwriting",
                    depends_on=["extract"],
                    condition="'handwriting' in bb.document_metadata.content_types",
                ),
            ],
        )
        agents = {
            "generic": _make_agent("generic"),
            "handwriting": _make_agent("handwriting"),
        }
        engine = PipelineEngine(_mock_engine(), agents)
        bb = Blackboard()  # content_types is empty, so condition is False

        result = await engine.execute(config, [b"img"], bb)

        assert "extract" in result.steps
        assert "handwriting" not in result.steps  # Skipped

    async def test_conditional_step_runs(self):
        config = PipelineConfig(
            name="conditional",
            steps=[
                StepConfig(name="extract", agent="generic", depends_on=[]),
                StepConfig(
                    name="handwriting",
                    agent="handwriting",
                    depends_on=["extract"],
                    condition="'handwriting' in bb.document_metadata.content_types",
                ),
            ],
        )
        agents = {
            "generic": _make_agent("generic"),
            "handwriting": _make_agent("handwriting"),
        }
        engine = PipelineEngine(_mock_engine(), agents)
        bb = Blackboard()
        bb.write("document_metadata", "content_types", ["handwriting"], writer="test")

        result = await engine.execute(config, [b"img"], bb)

        assert "handwriting" in result.steps  # Should run

    async def test_page_selection(self):
        config = PipelineConfig(
            name="selective",
            steps=[
                StepConfig(name="metadata", agent="meta", pages=[1], depends_on=[]),
            ],
        )
        agents = {"meta": _make_agent("meta")}
        engine = PipelineEngine(_mock_engine(), agents)

        result = await engine.execute(config, [b"p1", b"p2", b"p3"])

        assert "metadata" in result.steps

    async def test_blackboard_propagation(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "de", writer="pre")

        config = PipelineConfig(
            name="bb_test",
            steps=[StepConfig(name="extract", agent="generic", depends_on=[])],
        )
        agents = {"generic": _make_agent("generic")}
        engine = PipelineEngine(_mock_engine(), agents)

        result = await engine.execute(config, [b"img"], bb)

        # Blackboard should persist through execution
        assert result.blackboard.document_metadata.language == "de"
