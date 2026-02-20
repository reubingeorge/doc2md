"""YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from doc2md.config.schema import PipelineConfig
from doc2md.types import AgentConfig


def load_agent_yaml(path: str | Path) -> AgentConfig:
    """Load an agent YAML file and return a validated AgentConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Agent YAML not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "agent" not in raw:
        raise ValueError(f"Invalid agent YAML: missing top-level 'agent' key in {path}")

    return AgentConfig(**raw["agent"])


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load any YAML file safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping, got {type(raw).__name__} in {path}")

    return raw


def load_pipeline_yaml(path: str | Path) -> PipelineConfig:
    """Load a pipeline YAML file and return a validated PipelineConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline YAML not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "pipeline" not in raw:
        raise ValueError(f"Invalid pipeline YAML: missing top-level 'pipeline' key in {path}")

    return PipelineConfig(**raw["pipeline"])
