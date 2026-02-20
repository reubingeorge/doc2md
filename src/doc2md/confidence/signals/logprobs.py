"""Logprobs analysis signal — geometric mean of token probabilities."""

from __future__ import annotations

import math
from typing import Any


def compute_logprobs(
    logprobs: list[dict[str, Any]] | None,
) -> tuple[float, bool, str]:
    """Compute logprobs-based confidence from VLM response.

    Uses geometric mean of token probabilities (excluding special tokens).
    Only available on models that return logprobs (GPT-4.1/4o family).

    Returns (score, available, reasoning).
    """
    if not logprobs:
        return 0.0, False, "Model did not return logprobs"

    log_probs: list[float] = []
    for token_data in logprobs:
        lp = token_data.get("logprob")
        if lp is None:
            continue
        # Skip special tokens
        token = token_data.get("token", "")
        if token in ("<|endoftext|>", "<|begin_of_text|>", "<|end_of_text|>"):
            continue
        log_probs.append(lp)

    if not log_probs:
        return 0.0, False, "No usable logprob tokens found"

    # Geometric mean: exp(mean(log(p))) = exp(mean(logprob))
    mean_logprob = sum(log_probs) / len(log_probs)
    score = math.exp(mean_logprob)
    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    return score, True, f"Geometric mean of {len(log_probs)} token logprobs"
