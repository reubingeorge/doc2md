"""Shared Pydantic models for doc2md."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ──


class InputMode(StrEnum):
    IMAGE = "image"
    PREVIOUS_OUTPUT = "previous_output"
    IMAGE_AND_PREVIOUS = "image_and_previous"
    PREVIOUS_OUTPUTS = "previous_outputs"
    PREVIOUS_OUTPUT_ONLY = "previous_output_only"


class RetryStrategy(StrEnum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


class WritesVia(StrEnum):
    PROMPT_ELICITED = "prompt_elicited"
    CODE_COMPUTED = "code_computed"
    HYBRID = "hybrid"


class ConfidenceLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    FAILED = "FAILED"


# ── Config models ──


class ModelConfig(BaseModel):
    preferred: str = "gpt-4.1-mini"
    fallback: list[str] = Field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.0


class PromptConfig(BaseModel):
    system: str
    user: str


class PreprocessStep(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ValidationRule(BaseModel):
    rule: str
    params: dict[str, Any] = Field(default_factory=dict)


class RetryConfig(BaseModel):
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on: list[str] = Field(default_factory=list)


class CodeWriter(BaseModel):
    function: str
    input: str = "markdown"
    output_key: str


class BlackboardConfig(BaseModel):
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    writes_via: WritesVia = WritesVia.PROMPT_ELICITED
    write_schema: dict[str, Any] = Field(default_factory=dict)
    code_writers: list[CodeWriter] = Field(default_factory=list)


class ConfidenceConfig(BaseModel):
    signals: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    expected_fields: list[str] = Field(default_factory=list)
    calibration: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)
    input: InputMode = InputMode.IMAGE
    preprocessing: list[PreprocessStep] = Field(default_factory=list)
    prompt: PromptConfig
    blackboard: BlackboardConfig = Field(default_factory=BlackboardConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    validation: list[ValidationRule] = Field(default_factory=list)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    output_format: str | None = None
    postprocessing: list[str] = Field(default_factory=list)


# ── Runtime models ──


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ImageQuality(BaseModel):
    blur_score: float = 1.0
    contrast_score: float = 1.0
    resolution_dpi: int = 300
    noise_score: float = 1.0
    overall: float = 1.0


class VLMResponse(BaseModel):
    content: str
    model: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    logprobs: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


class StepResult(BaseModel):
    step_name: str
    agent_name: str
    markdown: str
    page_markdowns: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float | None = None
    confidence_level: ConfidenceLevel | None = None
    blackboard_writes: dict[str, Any] = Field(default_factory=dict)
    model_used: str = ""
    cached: bool = False


class PageTask(BaseModel):
    page_number: int
    image_bytes: bytes
    agent_name: str | None = None


class ConversionResult(BaseModel):
    markdown: str
    page_markdowns: list[str] = Field(default_factory=list)
    classified_as: str | None = None
    steps: dict[str, StepResult] = Field(default_factory=dict)
    confidence: float | None = None
    confidence_level: ConfidenceLevel | None = None
    needs_human_review: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    pages_processed: int = 0
    pages_failed: list[int] = Field(default_factory=list)
    model_config = {"arbitrary_types_allowed": True}

    def save(
        self,
        path: str | Path,
        per_page: bool = False,
    ) -> list[Path]:
        """Save markdown output to file(s).

        Args:
            path: Output file path (single file) or directory (per-page).
            per_page: If True, save each page as a separate file.

        Returns:
            List of paths written.
        """
        path = Path(path)
        written: list[Path] = []

        if per_page and self.page_markdowns:
            # Save per-page files into a directory
            path.mkdir(parents=True, exist_ok=True)
            for i, page_md in enumerate(self.page_markdowns, 1):
                page_path = path / f"page_{i:03d}.md"
                page_path.write_text(page_md, encoding="utf-8")
                written.append(page_path)
        else:
            # Save as a single file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.markdown, encoding="utf-8")
            written.append(path)

        return written
