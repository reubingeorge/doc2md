"""Model discovery — validate models against the curated allowlist."""

from __future__ import annotations

import logging

from doc2md.models.allowlist import ModelAllowlist, ModelInfo

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """Validate and select models from the curated allowlist.

    Uses the hardcoded models.yaml allowlist as the single source of truth.
    No dynamic API discovery — only models explicitly listed in the
    allowlist are accepted. This keeps the system predictable and avoids
    surprises from unknown models.
    """

    def __init__(self, allowlist: ModelAllowlist | None = None) -> None:
        self._allowlist = allowlist or ModelAllowlist()

    def validate_model(self, model_id: str) -> tuple[bool, str]:
        """Validate that a model is in the curated allowlist.

        Returns (is_valid, message).
        """
        if self._allowlist.is_allowed(model_id):
            return True, "OK"

        return False, f"Model '{model_id}' is not in the supported models list"

    def get_best_available(self, preferred: str, fallbacks: list[str] | None = None) -> str:
        """Get the best available model from preferred + fallbacks."""
        candidates = [preferred] + (fallbacks or [])

        for model_id in candidates:
            valid, _ = self.validate_model(model_id)
            if valid:
                return model_id

        logger.warning("No validated models available, using '%s' anyway", preferred)
        return preferred

    @property
    def available_models(self) -> list[ModelInfo]:
        """All models from the curated allowlist."""
        return self._allowlist.list_models()

    def supports_logprobs(self, model_id: str) -> bool:
        """Check if a model supports logprobs."""
        return self._allowlist.supports_logprobs(model_id)

    def get_by_tier(self, tier: str) -> list[ModelInfo]:
        """Get all models in a specific tier."""
        return self._allowlist.get_by_tier(tier)
