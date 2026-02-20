"""Signal combination with adaptive weight redistribution."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SignalResult(BaseModel):
    """Result from a single confidence signal."""

    name: str
    score: float = 0.0
    available: bool = False
    reasoning: str = ""


def combine_signals(
    signals: list[SignalResult],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Combine signal scores using weighted average with adaptive redistribution.

    When a signal is unavailable, its weight is redistributed proportionally
    to the remaining available signals.

    Returns (combined_score, effective_weights).
    """
    if not signals:
        return 0.0, {}

    # Identify available signals
    available = {s.name: s for s in signals if s.available}

    if not available:
        return 0.0, {}

    # Compute effective weights (redistribute unavailable weight)
    effective = _redistribute_weights(weights, set(available.keys()))

    # Weighted average
    combined = sum(
        available[name].score * weight
        for name, weight in effective.items()
        if name in available
    )

    return combined, effective


def _redistribute_weights(
    original: dict[str, float],
    available_names: set[str],
) -> dict[str, float]:
    """Redistribute weights from unavailable signals to available ones.

    Proportional redistribution: each available signal gets its share
    of the unavailable weight, proportional to its original weight.
    """
    if not available_names:
        return {}

    # Calculate total weight of available signals
    available_weight_sum = sum(
        original.get(name, 0.0) for name in available_names
    )

    if available_weight_sum <= 0:
        # Equal weights if no original weights defined for available signals
        equal = 1.0 / len(available_names)
        return {name: equal for name in available_names}

    # Normalize: each available signal gets weight / sum(available_weights)
    return {
        name: original.get(name, 0.0) / available_weight_sum
        for name in available_names
    }
