"""Agent and pipeline registry — discover, register, and look up configs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import yaml

from doc2md.config.loader import load_agent_yaml, load_pipeline_yaml
from doc2md.config.schema import PipelineConfig
from doc2md.types import AgentConfig

logger = logging.getLogger(__name__)

_BUILTIN_AGENTS_DIR = Path(__file__).parent / "builtin" / "agents"
_BUILTIN_PIPELINES_DIR = Path(__file__).parent / "builtin" / "pipelines"


class AgentInfo(NamedTuple):
    name: str
    version: str
    description: str
    builtin: bool


class PipelineInfo(NamedTuple):
    name: str
    version: str
    description: str
    builtin: bool
    step_count: int


class AgentRegistry:
    """Discovers and stores agent configs from builtin + user directories."""

    def __init__(self, user_dirs: list[Path] | None = None) -> None:
        self._agents: dict[str, AgentConfig] = {}
        self._sources: dict[str, bool] = {}  # name → is_builtin
        self._scan(_BUILTIN_AGENTS_DIR, builtin=True)
        for d in user_dirs or []:
            self._scan(d, builtin=False)

    def get(self, name: str) -> AgentConfig:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found in registry")
        return self._agents[name]

    def has(self, name: str) -> bool:
        return name in self._agents

    def list_agents(self) -> list[AgentInfo]:
        return [
            AgentInfo(
                name=c.name,
                version=c.version,
                description=c.description,
                builtin=self._sources[c.name],
            )
            for c in self._agents.values()
        ]

    def all_configs(self) -> dict[str, AgentConfig]:
        return dict(self._agents)

    def register(self, config: AgentConfig, builtin: bool = False) -> None:
        self._agents[config.name] = config
        self._sources[config.name] = builtin

    def _scan(self, directory: Path, builtin: bool) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    raw = yaml.safe_load(f)
                if not isinstance(raw, dict) or "agent" not in raw:
                    continue  # Not an agent YAML, skip silently
                config = load_agent_yaml(path)
                # User agents override builtins
                if config.name not in self._agents or not builtin:
                    self._agents[config.name] = config
                    self._sources[config.name] = builtin
            except Exception as e:
                logger.warning("Failed to load agent %s: %s", path, e)


class PipelineRegistry:
    """Discovers and stores pipeline configs from builtin + user directories."""

    def __init__(self, user_dirs: list[Path] | None = None) -> None:
        self._pipelines: dict[str, PipelineConfig] = {}
        self._sources: dict[str, bool] = {}
        self._scan(_BUILTIN_PIPELINES_DIR, builtin=True)
        for d in user_dirs or []:
            self._scan(d, builtin=False)

    def get(self, name: str) -> PipelineConfig:
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found in registry")
        return self._pipelines[name]

    def has(self, name: str) -> bool:
        return name in self._pipelines

    def list_pipelines(self) -> list[PipelineInfo]:
        return [
            PipelineInfo(
                name=c.name,
                version=c.version,
                description=c.description,
                builtin=self._sources[c.name],
                step_count=len(c.steps),
            )
            for c in self._pipelines.values()
        ]

    def register(self, config: PipelineConfig, builtin: bool = False) -> None:
        self._pipelines[config.name] = config
        self._sources[config.name] = builtin

    def _scan(self, directory: Path, builtin: bool) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    raw = yaml.safe_load(f)
                if not isinstance(raw, dict) or "pipeline" not in raw:
                    continue  # Not a pipeline YAML, skip silently
                config = load_pipeline_yaml(path)
                if config.name not in self._pipelines or not builtin:
                    self._pipelines[config.name] = config
                    self._sources[config.name] = builtin
            except Exception as e:
                logger.warning("Failed to load pipeline %s: %s", path, e)
