"""Integration test: confidence scoring through pipeline execution."""

from unittest.mock import AsyncMock

from doc2md.agents.engine import AgentEngine
from doc2md.blackboard.board import Blackboard
from doc2md.config.schema import PipelineConfig, StepConfig, StepType
from doc2md.pipeline.engine import PipelineEngine
from doc2md.types import AgentConfig, ModelConfig, PromptConfig, TokenUsage, VLMResponse, ValidationRule, ConfidenceConfig


def _mock_response(content: str = "# Title\n\nContent here [confidence: HIGH]") -> VLMResponse:
    return VLMResponse(
        content=content,
        model="gpt-4.1-mini",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


class TestConfidencePipelineIntegration:
    async def test_single_step_gets_confidence(self, sample_image_bytes):
        mock_client = AsyncMock()
        mock_client.send_request = AsyncMock(return_value=_mock_response())
        engine = AgentEngine(mock_client)

        agent_config = AgentConfig(
            name="test",
            prompt=PromptConfig(system="sys", user="usr"),
            model=ModelConfig(preferred="gpt-4.1-mini"),
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment"],
                weights={"vlm_self_assessment": 1.0},
            ),
        )

        pipeline_config = PipelineConfig(
            name="test_pipe",
            steps=[StepConfig(name="extract", type=StepType.AGENT, agent="test")],
        )

        pipeline_engine = PipelineEngine(engine, {"test": agent_config})
        result = await pipeline_engine.execute(pipeline_config, [sample_image_bytes])

        # Step should have confidence
        assert result.steps["extract"].confidence is not None
        assert result.steps["extract"].confidence > 0

        # Pipeline should have confidence report
        assert result.confidence_report is not None
        assert result.confidence_report.overall > 0

    async def test_confidence_signals_written_to_blackboard(self, sample_image_bytes):
        mock_client = AsyncMock()
        mock_client.send_request = AsyncMock(return_value=_mock_response())
        engine = AgentEngine(mock_client)

        agent_config = AgentConfig(
            name="test",
            prompt=PromptConfig(system="sys", user="usr"),
            model=ModelConfig(preferred="gpt-4.1-mini"),
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment"],
                weights={"vlm_self_assessment": 1.0},
            ),
        )

        pipeline_config = PipelineConfig(
            name="test_pipe",
            steps=[StepConfig(name="extract", type=StepType.AGENT, agent="test")],
        )

        bb = Blackboard()
        pipeline_engine = PipelineEngine(engine, {"test": agent_config})
        await pipeline_engine.execute(pipeline_config, [sample_image_bytes], blackboard=bb)

        # Confidence signals should be in blackboard
        assert "extract" in bb.confidence_signals

    async def test_two_step_pipeline_aggregates_confidence(self, sample_image_bytes):
        mock_client = AsyncMock()
        mock_client.send_request = AsyncMock(return_value=_mock_response())
        engine = AgentEngine(mock_client)

        agent_config = AgentConfig(
            name="test",
            prompt=PromptConfig(system="sys", user="usr"),
            model=ModelConfig(preferred="gpt-4.1-mini"),
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment"],
                weights={"vlm_self_assessment": 1.0},
            ),
        )

        pipeline_config = PipelineConfig(
            name="test_pipe",
            steps=[
                StepConfig(name="extract", type=StepType.AGENT, agent="test"),
                StepConfig(name="validate", type=StepType.AGENT, agent="test",
                           depends_on=["extract"]),
            ],
        )

        pipeline_engine = PipelineEngine(engine, {"test": agent_config})
        result = await pipeline_engine.execute(pipeline_config, [sample_image_bytes])

        report = result.confidence_report
        assert report is not None
        assert len(report.per_step) == 2
        assert "extract" in report.per_step
        assert "validate" in report.per_step
