"""Confidence engine — orchestrates signal computation, combination, and calibration."""

from __future__ import annotations

import logging

from doc2md.confidence.calibration import calibrate
from doc2md.confidence.combiner import SignalResult, combine_signals
from doc2md.confidence.report import (
    ConfidenceReport,
    StepConfidenceReport,
    aggregate_step_scores,
    needs_human_review,
    score_to_level,
)
from doc2md.confidence.signals.completeness import compute_completeness
from doc2md.confidence.signals.image_quality import compute_image_quality
from doc2md.confidence.signals.logprobs import compute_logprobs
from doc2md.confidence.signals.validation import compute_validation_pass_rate
from doc2md.confidence.signals.vlm_assessment import compute_vlm_self_assessment
from doc2md.types import AgentConfig, StepResult, VLMResponse

logger = logging.getLogger(__name__)

# Default weights when agent YAML doesn't specify any
_DEFAULT_WEIGHTS: dict[str, float] = {
    "vlm_self_assessment": 0.30,
    "logprobs_analysis": 0.20,
    "validation_pass_rate": 0.20,
    "completeness_check": 0.15,
    "image_quality": 0.15,
}


class ConfidenceEngine:
    """Orchestrate signal computation → combination → calibration → threshold."""

    def compute_step_confidence(
        self,
        step_result: StepResult,
        agent_config: AgentConfig,
        vlm_response: VLMResponse | None = None,
        image_bytes: bytes | None = None,
    ) -> StepConfidenceReport:
        """Compute confidence for a single step execution.

        Collects all available signals, combines with adaptive weights,
        applies calibration, and returns a StepConfidenceReport.
        """
        signals = self._collect_signals(
            step_result=step_result,
            agent_config=agent_config,
            vlm_response=vlm_response,
            image_bytes=image_bytes,
        )

        # Get configured weights (or defaults)
        weights = agent_config.confidence.weights or _DEFAULT_WEIGHTS

        # Combine signals
        raw_score, effective_weights = combine_signals(signals, weights)

        # Calibrate
        cal_config = agent_config.confidence.calibration
        method = cal_config.get("method", "none") if cal_config else "none"
        manual_curve = cal_config.get("manual_curve") if cal_config else None
        calibrated = calibrate(raw_score, method=method, manual_curve=manual_curve)

        level = score_to_level(calibrated)

        reasoning_parts = [
            f"raw={raw_score:.3f}",
            f"calibrated={calibrated:.3f}",
            f"level={level.value}",
        ]
        available_signals = [s for s in signals if s.available]
        if available_signals:
            reasoning_parts.append(
                f"signals: {', '.join(s.name + '=' + f'{s.score:.2f}' for s in available_signals)}"
            )

        return StepConfidenceReport(
            step_name=step_result.step_name,
            agent_name=step_result.agent_name,
            raw_score=raw_score,
            calibrated_score=calibrated,
            level=level,
            signals=signals,
            effective_weights=effective_weights,
            reasoning="; ".join(reasoning_parts),
        )

    def aggregate_pipeline(
        self,
        step_reports: dict[str, StepConfidenceReport],
        strategy: str = "weighted_average",
        step_weights: dict[str, float] | None = None,
    ) -> ConfidenceReport:
        """Aggregate step-level confidence into a document-level report."""
        step_scores = {name: report.calibrated_score for name, report in step_reports.items()}

        overall = aggregate_step_scores(step_scores, strategy, step_weights)
        level = score_to_level(overall)

        return ConfidenceReport(
            overall=overall,
            level=level,
            needs_human_review=needs_human_review(overall),
            per_step=step_reports,
            strategy=strategy,
            reasoning=f"Aggregated {len(step_reports)} steps via {strategy}: {overall:.3f}",
        )

    def _collect_signals(
        self,
        step_result: StepResult,
        agent_config: AgentConfig,
        vlm_response: VLMResponse | None,
        image_bytes: bytes | None,
    ) -> list[SignalResult]:
        """Collect all configured confidence signals."""
        configured = (
            set(agent_config.confidence.signals) if agent_config.confidence.signals else set()
        )
        # If no signals configured, use all available
        use_all = not configured

        signals: list[SignalResult] = []

        # VLM self-assessment
        if use_all or "vlm_self_assessment" in configured:
            score, available, reasoning = compute_vlm_self_assessment(
                step_result.confidence_level,
            )
            signals.append(
                SignalResult(
                    name="vlm_self_assessment",
                    score=score,
                    available=available,
                    reasoning=reasoning,
                )
            )

        # Logprobs
        if use_all or "logprobs_analysis" in configured:
            logprobs_data = vlm_response.logprobs if vlm_response else None
            score, available, reasoning = compute_logprobs(logprobs_data)
            signals.append(
                SignalResult(
                    name="logprobs_analysis",
                    score=score,
                    available=available,
                    reasoning=reasoning,
                )
            )

        # Validation pass rate
        if use_all or "validation_pass_rate" in configured:
            score, available, reasoning = compute_validation_pass_rate(
                step_result.markdown,
                agent_config.validation,
            )
            signals.append(
                SignalResult(
                    name="validation_pass_rate",
                    score=score,
                    available=available,
                    reasoning=reasoning,
                )
            )

        # Completeness check
        if use_all or "completeness_check" in configured:
            score, available, reasoning = compute_completeness(
                step_result.markdown,
                agent_config.confidence.expected_fields,
            )
            signals.append(
                SignalResult(
                    name="completeness_check",
                    score=score,
                    available=available,
                    reasoning=reasoning,
                )
            )

        # Image quality
        if use_all or "image_quality" in configured:
            score, available, reasoning = compute_image_quality(image_bytes)
            signals.append(
                SignalResult(
                    name="image_quality",
                    score=score,
                    available=available,
                    reasoning=reasoning,
                )
            )

        return signals
