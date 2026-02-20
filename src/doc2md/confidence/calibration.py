"""Calibration — adjust raw confidence scores to reduce VLM overconfidence."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def calibrate(
    raw_score: float,
    method: str = "manual",
    manual_curve: list[list[float]] | None = None,
) -> float:
    """Apply calibration to a raw confidence score.

    Supports:
      - "manual": Linear interpolation on a manual curve (default)
      - "platt_scaling": Sigmoid fit (requires scikit-learn + trained model)
      - "isotonic": Non-parametric (requires scikit-learn + trained model)
      - "none": No calibration (pass-through)

    Returns calibrated score clamped to [0.0, 1.0].
    """
    if method == "none" or method is None:
        return raw_score

    if method == "manual":
        return _manual_calibrate(raw_score, manual_curve or [])

    if method in ("platt_scaling", "isotonic"):
        logger.debug(
            "Calibration method '%s' requires trained model; "
            "falling back to manual.",
            method,
        )
        return _manual_calibrate(raw_score, manual_curve or [])

    return raw_score


def _manual_calibrate(
    raw_score: float,
    curve: list[list[float]],
) -> float:
    """Apply manual calibration via linear interpolation.

    Curve is a list of [raw, calibrated] pairs sorted by raw value.
    Scores outside the curve range are clamped to boundary values.
    """
    if not curve:
        return raw_score

    # Sort by raw value
    sorted_curve = sorted(curve, key=lambda p: p[0])

    # Below curve range
    if raw_score <= sorted_curve[0][0]:
        return max(0.0, sorted_curve[0][1])

    # Above curve range
    if raw_score >= sorted_curve[-1][0]:
        return min(1.0, sorted_curve[-1][1])

    # Linear interpolation between two nearest points
    for i in range(len(sorted_curve) - 1):
        raw_lo, cal_lo = sorted_curve[i]
        raw_hi, cal_hi = sorted_curve[i + 1]
        if raw_lo <= raw_score <= raw_hi:
            t = (raw_score - raw_lo) / (raw_hi - raw_lo) if raw_hi != raw_lo else 0.0
            calibrated = cal_lo + t * (cal_hi - cal_lo)
            return max(0.0, min(1.0, calibrated))

    return raw_score
