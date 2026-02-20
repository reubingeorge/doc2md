"""Pipeline execution engine — DAG-based multi-step orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Auto-register built-in transforms on import
import doc2md.transforms  # noqa: F401
from doc2md.blackboard.board import Blackboard
from doc2md.confidence.engine import ConfidenceEngine
from doc2md.confidence.report import (
    ConfidenceReport,
    StepConfidenceReport,
)
from doc2md.config.schema import PipelineConfig, StepConfig, StepType
from doc2md.pipeline.data_flow import resolve_step_input
from doc2md.pipeline.graph import parse_pipeline
from doc2md.pipeline.merger import merge_outputs
from doc2md.pipeline.postprocessor import run_postprocessing
from doc2md.pipeline.step_executor import execute_step
from doc2md.types import AgentConfig, StepResult, TokenUsage

if TYPE_CHECKING:
    from doc2md.agents.engine import AgentEngine
    from doc2md.cache.manager import CacheManager

logger = logging.getLogger(__name__)


class PipelineResult:
    """Result of a full pipeline execution."""

    def __init__(
        self,
        markdown: str,
        steps: dict[str, StepResult],
        blackboard: Blackboard,
        pipeline_name: str,
        confidence_report: ConfidenceReport | None = None,
    ) -> None:
        self.markdown = markdown
        self.steps = steps
        self.blackboard = blackboard
        self.pipeline_name = pipeline_name
        self.confidence_report = confidence_report

    @property
    def token_usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=sum(s.token_usage.prompt_tokens for s in self.steps.values()),
            completion_tokens=sum(s.token_usage.completion_tokens for s in self.steps.values()),
            total_tokens=sum(s.token_usage.total_tokens for s in self.steps.values()),
        )


class PipelineEngine:
    """Execute a pipeline: parse YAML → build graph → topological sort → run steps."""

    def __init__(
        self,
        agent_engine: AgentEngine,
        agent_configs: dict[str, AgentConfig],
        cache_manager: CacheManager | None = None,
    ) -> None:
        self._agent_engine = agent_engine
        self._agent_configs = agent_configs
        self._cache_manager = cache_manager
        self._confidence_engine = ConfidenceEngine()

    async def execute(
        self,
        pipeline_config: PipelineConfig,
        images: list[bytes],
        blackboard: Blackboard | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline."""
        if blackboard is None:
            blackboard = Blackboard()

        graph = parse_pipeline(pipeline_config)
        execution_order = graph.topological_sort()
        step_results: dict[str, StepResult] = {}
        step_confidence_reports: dict[str, StepConfidenceReport] = {}

        for step_name in execution_order:
            step_config = graph.get_step(step_name)

            # Evaluate condition
            if not self._evaluate_condition(step_config, blackboard):
                logger.info("Skipping step '%s': condition not met", step_name)
                continue

            # Resolve page selection
            page_images = self._select_pages(step_config, images)

            # Resolve dependencies
            deps = graph.dependencies_of(step_name)

            # Resolve input
            step_input = resolve_step_input(
                input_mode=step_config.input,
                images=page_images,
                depends_on=deps,
                step_results=step_results,
            )

            logger.info("Executing step '%s' (type=%s)", step_name, step_config.type.value)

            result = await execute_step(
                step_config=step_config,
                step_input=step_input,
                blackboard=blackboard,
                agent_engine=self._agent_engine,
                agent_configs=self._agent_configs,
                cache_manager=self._cache_manager,
                pipeline_name=pipeline_config.name,
            )

            # Compute step-level confidence (agent steps only)
            if step_config.type == StepType.AGENT and step_config.agent:
                agent_config = self._agent_configs.get(step_config.agent)
                if agent_config:
                    report = self._confidence_engine.compute_step_confidence(
                        step_result=result,
                        agent_config=agent_config,
                        image_bytes=page_images[0] if page_images else None,
                    )
                    result.confidence = report.calibrated_score
                    result.confidence_level = report.level
                    step_confidence_reports[step_name] = report

                    # Write signals to blackboard
                    signal_data = {s.name: s.score for s in report.signals if s.available}
                    if signal_data:
                        blackboard.write(
                            "confidence_signals",
                            step_name,
                            signal_data,
                            writer=step_name,
                        )

            step_results[step_name] = result

        # Aggregate pipeline-level confidence
        confidence_report = None
        if step_confidence_reports:
            conf_config = pipeline_config.confidence
            strategy = conf_config.strategy if conf_config else "weighted_average"
            step_weights = conf_config.step_weights if conf_config else None
            confidence_report = self._confidence_engine.aggregate_pipeline(
                step_confidence_reports,
                strategy=strategy,
                step_weights=step_weights,
            )

        # Final merge
        final_markdown = self._final_merge(step_results, pipeline_config)

        # Pipeline-level postprocessing
        if pipeline_config.postprocessing:
            final_markdown = run_postprocessing(final_markdown, pipeline_config.postprocessing)

        return PipelineResult(
            markdown=final_markdown,
            steps=step_results,
            blackboard=blackboard,
            pipeline_name=pipeline_config.name,
            confidence_report=confidence_report,
        )

    @staticmethod
    def _evaluate_condition(step_config: StepConfig, blackboard: Blackboard) -> bool:
        """Evaluate a step's condition expression against the blackboard."""
        if not step_config.condition:
            return True
        try:
            bb = blackboard  # noqa: F841 — used in eval
            return bool(
                eval(
                    step_config.condition,
                    {
                        "bb": blackboard,
                        "__builtins__": {
                            "any": any,
                            "all": all,
                            "len": len,
                            "str": str,
                            "int": int,
                            "float": float,
                        },
                    },
                )
            )
        except Exception:
            logger.warning("Condition eval failed for step '%s', running anyway", step_config.name)
            return True

    @staticmethod
    def _select_pages(step_config: StepConfig, images: list[bytes]) -> list[bytes]:
        """Select page images based on step's page selector."""
        selector = step_config.get_page_selector()
        page_nums = selector.resolve(len(images))
        return [images[p - 1] for p in page_nums if 0 < p <= len(images)]

    @staticmethod
    def _final_merge(
        step_results: dict[str, StepResult],
        pipeline_config: PipelineConfig,
    ) -> str:
        """Merge all step outputs into the final document."""
        outputs = {name: r.markdown for name, r in step_results.items() if r.markdown}
        if not outputs:
            return ""

        # If only one step, return its output directly
        if len(outputs) == 1:
            return next(iter(outputs.values()))

        return merge_outputs(outputs, pipeline_config.page_merge)
