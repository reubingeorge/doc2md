"""Confidence report generation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from doc2md.confidence.combiner import SignalResult
from doc2md.types import ConfidenceLevel


# Decision thresholds
_THRESHOLDS: list[tuple[float, ConfidenceLevel]] = [
    (0.8, ConfidenceLevel.HIGH),
    (0.6, ConfidenceLevel.MEDIUM),
    (0.3, ConfidenceLevel.LOW),
]


class StepConfidenceReport(BaseModel):
    """Confidence report for a single step execution."""

    step_name: str
    agent_name: str = ""
    raw_score: float = 0.0
    calibrated_score: float = 0.0
    level: ConfidenceLevel = ConfidenceLevel.FAILED
    signals: list[SignalResult] = Field(default_factory=list)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


class ConfidenceReport(BaseModel):
    """Full confidence report for a document conversion."""

    overall: float = 0.0
    level: ConfidenceLevel = ConfidenceLevel.FAILED
    needs_human_review: bool = True
    per_step: dict[str, StepConfidenceReport] = Field(default_factory=dict)
    per_page: dict[int, float] = Field(default_factory=dict)
    strategy: str = "weighted_average"
    reasoning: str = ""


def score_to_level(score: float) -> ConfidenceLevel:
    """Convert a numeric confidence score to a ConfidenceLevel."""
    for threshold, level in _THRESHOLDS:
        if score >= threshold:
            return level
    return ConfidenceLevel.FAILED


def needs_human_review(score: float) -> bool:
    """Determine if a confidence score warrants human review."""
    return score < 0.6


def aggregate_step_scores(
    step_scores: dict[str, float],
    strategy: str = "weighted_average",
    step_weights: dict[str, float] | None = None,
) -> float:
    """Aggregate step-level confidence into a document-level score.

    Strategies:
      - weighted_average: Weighted mean of step scores
      - minimum: Minimum score across steps
      - last_step: Score of the last step only
    """
    if not step_scores:
        return 0.0

    if strategy == "minimum":
        return min(step_scores.values())

    if strategy == "last_step":
        # Use the last step's score (dict is ordered in Python 3.7+)
        return list(step_scores.values())[-1]

    # weighted_average (default)
    if step_weights:
        total_weight = sum(
            step_weights.get(name, 0.0) for name in step_scores
        )
        if total_weight > 0:
            return sum(
                score * step_weights.get(name, 0.0)
                for name, score in step_scores.items()
            ) / total_weight

    # Equal-weight average fallback
    return sum(step_scores.values()) / len(step_scores)
