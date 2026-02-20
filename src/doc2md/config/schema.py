"""Pydantic models for pipeline configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from doc2md.types import InputMode


class StepType(StrEnum):
    AGENT = "agent"
    PARALLEL = "parallel"
    PAGE_ROUTE = "page_route"
    CODE = "code"


class RouterStrategy(StrEnum):
    VLM = "vlm"
    RULES = "rules"
    HYBRID = "hybrid"


class PageSelector(BaseModel):
    """Selects which pages a step operates on.

    Supports: [1], [1, 3], [2:], [:5], [-1], None (all pages).
    Stored as raw value and resolved at runtime.
    """

    raw: list[int | str] | None = None

    def resolve(self, total_pages: int) -> list[int]:
        """Resolve page selector to concrete 1-based page numbers."""
        if self.raw is None:
            return list(range(1, total_pages + 1))

        result: list[int] = []
        for item in self.raw:
            if isinstance(item, int):
                idx = item if item > 0 else total_pages + item + 1
                if 1 <= idx <= total_pages:
                    result.append(idx)
            elif isinstance(item, str) and ":" in item:
                parts = item.split(":")
                start = int(parts[0]) if parts[0] else 1
                end = int(parts[1]) if parts[1] else total_pages
                if start < 0:
                    start = total_pages + start + 1
                if end < 0:
                    end = total_pages + end + 1
                for p in range(max(1, start), min(total_pages, end) + 1):
                    result.append(p)
        return sorted(set(result))


class RouterRule(BaseModel):
    pages: list[int | str] = Field(default_factory=list)
    agent: str


class VLMFallbackConfig(BaseModel):
    model: str = "gpt-4.1-nano"
    batch_size: int = 8
    categories: dict[str, dict[str, str]] = Field(default_factory=dict)


class RouterConfig(BaseModel):
    strategy: RouterStrategy = RouterStrategy.RULES
    rules: list[RouterRule] = Field(default_factory=list)
    vlm_fallback: VLMFallbackConfig | None = None
    default_agent: str = "generic"


class MergeConfig(BaseModel):
    strategy: str = "concatenate"  # "concatenate" or "agent"
    agent: str | None = None


class ConfidenceStrategyConfig(BaseModel):
    strategy: str = "weighted_average"  # minimum, weighted_average, last_step
    step_weights: dict[str, float] = Field(default_factory=dict)


class StepConfig(BaseModel):
    name: str
    type: StepType = StepType.AGENT
    agent: str | None = None
    input: InputMode = InputMode.IMAGE
    pages: list[int | str] | None = None
    depends_on: list[str] | None = None  # None = implicit (previous step)
    condition: str | None = None
    cross_page_aware: bool = False

    # For parallel steps
    steps: list[StepConfig] | None = None
    merge: MergeConfig | None = None

    # For page_route steps
    router: RouterConfig | None = None

    # For code steps
    function: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    def get_page_selector(self) -> PageSelector:
        return PageSelector(raw=self.pages)


class PipelineConfig(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    steps: list[StepConfig]
    page_merge: MergeConfig | None = None
    confidence: ConfidenceStrategyConfig | None = None
    postprocessing: list[str] = Field(default_factory=list)
