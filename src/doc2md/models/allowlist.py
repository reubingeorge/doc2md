"""Model allowlist — curated list of supported models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODELS_YAML = Path(__file__).parent / "models.yaml"


class ModelInfo(BaseModel):
    """Information about a supported model."""

    name: str
    tier: str = "standard"
    priority: int = 99
    logprobs: bool = False
    max_tokens: int = 4096
    description: str = ""


class ModelAllowlist:
    """Curated list of supported models loaded from models.yaml."""

    def __init__(self, models_path: Path | None = None) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._load(models_path or _MODELS_YAML)

    def _load(self, path: Path) -> None:
        """Load models from YAML file."""
        if not path.exists():
            logger.warning("Models YAML not found: %s", path)
            return

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "models" not in data:
            logger.warning("Invalid models YAML: missing 'models' key")
            return

        for name, info in data["models"].items():
            if isinstance(info, dict):
                self._models[name] = ModelInfo(name=name, **info)

    def is_allowed(self, model_id: str) -> bool:
        """Check if a model is in the allowlist."""
        return model_id in self._models

    def get(self, model_id: str) -> ModelInfo | None:
        """Get info for a model, or None if not in allowlist."""
        return self._models.get(model_id)

    def list_models(self) -> list[ModelInfo]:
        """Return all models sorted by priority."""
        return sorted(self._models.values(), key=lambda m: m.priority)

    def supports_logprobs(self, model_id: str) -> bool:
        """Check if a model supports logprobs."""
        info = self._models.get(model_id)
        return info.logprobs if info else False

    def get_by_tier(self, tier: str) -> list[ModelInfo]:
        """Get all models in a specific tier."""
        return [m for m in self._models.values() if m.tier == tier]

    @property
    def model_names(self) -> list[str]:
        """All model names in the allowlist."""
        return list(self._models.keys())
