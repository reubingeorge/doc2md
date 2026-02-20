"""Consistency check signal — compare two independent extractions.

This signal has 2x VLM cost and is optional. Full dual-extraction
triggering deferred until Phase 7 (concurrency). The similarity
computation itself is implemented here.
"""

from __future__ import annotations


def compute_consistency(
    markdown_a: str | None = None,
    markdown_b: str | None = None,
) -> tuple[float, bool, str]:
    """Compare two independent extractions for consistency.

    When both extractions are provided, computes token-level similarity.
    Returns (score, available, reasoning).
    """
    if markdown_a is None or markdown_b is None:
        return 0.0, False, "Consistency check requires two extractions (2x cost)"

    # Simple token-level Jaccard similarity
    tokens_a = set(markdown_a.lower().split())
    tokens_b = set(markdown_b.lower().split())

    if not tokens_a and not tokens_b:
        return 1.0, True, "Both extractions empty"

    if not tokens_a or not tokens_b:
        return 0.0, True, "One extraction empty"

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    score = len(intersection) / len(union)

    return score, True, f"Jaccard similarity: {score:.3f} ({len(intersection)}/{len(union)} tokens)"
