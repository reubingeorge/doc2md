"""Confidence scoring — multi-signal weighted scoring with calibration."""

from doc2md.confidence.combiner import SignalResult, combine_signals
from doc2md.confidence.engine import ConfidenceEngine
from doc2md.confidence.report import (
    ConfidenceReport,
    StepConfidenceReport,
    aggregate_step_scores,
    needs_human_review,
    score_to_level,
)

__all__ = [
    "ConfidenceEngine",
    "ConfidenceReport",
    "StepConfidenceReport",
    "SignalResult",
    "combine_signals",
    "aggregate_step_scores",
    "needs_human_review",
    "score_to_level",
]
