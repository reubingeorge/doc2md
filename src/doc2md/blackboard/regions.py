"""Pydantic models for the 5 typed blackboard regions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UncertainRegion(BaseModel):
    """A region of a page where extraction confidence is low."""

    page: int | None = None
    area: str = ""  # e.g. "bottom_right", "top_left"
    reason: str = ""
    confidence: str = "low"


class PageObservation(BaseModel):
    """Per-page observations written by agents during pipeline execution."""

    content_types: list[str] = Field(default_factory=list)
    rotation: float = 0.0
    continues_on_next_page: bool = False
    continues_from_previous: bool = False
    quality_score: float | None = None
    table_count: int | None = None
    uncertain_regions: list[UncertainRegion] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Document-level metadata, typically written once by an early agent."""

    language: str | None = None
    date_format: str | None = None
    layout: str | None = None  # e.g. "single_column", "two_column"
    page_count: int | None = None
    content_types: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


# The 5 region names and their types, for validation
REGION_TYPES: dict[str, type] = {
    "document_metadata": DocumentMetadata,
    "page_observations": dict,  # Dict[int, PageObservation]
    "step_outputs": dict,  # Dict[str, str]
    "agent_notes": dict,  # Dict[str, Dict[str, Any]]
    "confidence_signals": dict,  # Dict[str, Dict[str, Any]]
}

VALID_REGIONS = frozenset(REGION_TYPES.keys())
