"""Model fallback chain — cycle through models on terminal errors."""

from __future__ import annotations

import logging

from doc2md.errors.exceptions import TerminalError

logger = logging.getLogger(__name__)


class FallbackChain:
    """Manages model fallback on terminal errors (401, 404, model-not-found).

    Cycles through the model list in order. Tracks which models have been tried.
    """

    def __init__(self, preferred: str, fallbacks: list[str] | None = None) -> None:
        self._models = [preferred] + (fallbacks or [])
        self._tried: set[str] = set()
        self._current_index = 0

    @property
    def current_model(self) -> str:
        return self._models[self._current_index]

    @property
    def exhausted(self) -> bool:
        return len(self._tried) >= len(self._models)

    def next_model(self) -> str:
        """Advance to the next untried model.

        Raises TerminalError if all models have been exhausted.
        """
        self._tried.add(self.current_model)

        for i in range(self._current_index + 1, len(self._models)):
            if self._models[i] not in self._tried:
                self._current_index = i
                logger.info(
                    "Falling back to model '%s' (tried: %s)",
                    self.current_model,
                    ", ".join(sorted(self._tried)),
                )
                return self.current_model

        raise TerminalError(
            f"All models exhausted: {', '.join(self._models)}",
            error_type="model_not_found",
            recoverable_with_fallback=False,
        )

    def mark_tried(self, model: str) -> None:
        """Mark a model as tried (e.g., after failure)."""
        self._tried.add(model)

    def reset(self) -> None:
        """Reset the chain for a new request."""
        self._tried.clear()
        self._current_index = 0
