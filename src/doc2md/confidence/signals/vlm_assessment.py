"""VLM self-assessment signal — parsed from VLM output at zero extra cost."""

from __future__ import annotations

from doc2md.types import ConfidenceLevel


# Map VLM self-assessment levels to numeric scores
_LEVEL_SCORES: dict[ConfidenceLevel, float] = {
    ConfidenceLevel.HIGH: 0.90,
    ConfidenceLevel.MEDIUM: 0.65,
    ConfidenceLevel.LOW: 0.35,
    ConfidenceLevel.FAILED: 0.10,
}


def compute_vlm_self_assessment(
    confidence_level: ConfidenceLevel | None,
) -> tuple[float, bool, str]:
    """Compute VLM self-assessment signal.

    Returns (score, available, reasoning).
    """
    if confidence_level is None:
        return 0.0, False, "VLM did not provide a self-assessment"

    score = _LEVEL_SCORES.get(confidence_level, 0.5)
    return score, True, f"VLM self-assessed as {confidence_level.value}"
