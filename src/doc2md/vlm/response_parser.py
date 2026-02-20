"""Parse VLM response text into markdown and metadata."""

from __future__ import annotations

import re
from typing import Any

import yaml

from doc2md.types import ConfidenceLevel

# Pattern to extract <blackboard>...</blackboard> blocks
_BLACKBOARD_PATTERN = re.compile(
    r"<blackboard>\s*(.*?)\s*</blackboard>",
    re.DOTALL,
)

# Confidence tags the VLM may embed
_CONFIDENCE_PATTERN = re.compile(
    r"\[confidence:\s*(HIGH|MEDIUM|LOW)\]",
    re.IGNORECASE,
)


def parse_response(raw_text: str) -> tuple[str, dict[str, Any]]:
    """Parse VLM output into clean markdown and extracted metadata.

    Returns (markdown, metadata_dict).
    """
    metadata: dict[str, Any] = {}

    # Extract and parse blackboard blocks
    blackboard_data = _extract_blackboard(raw_text)
    if blackboard_data is not None:
        metadata["blackboard_writes"] = blackboard_data

    # Strip blackboard blocks from markdown
    markdown = _BLACKBOARD_PATTERN.sub("", raw_text).strip()

    # Extract confidence self-assessments
    confidence_level = _extract_confidence(markdown)
    if confidence_level is not None:
        metadata["confidence_level"] = confidence_level
        markdown = _CONFIDENCE_PATTERN.sub("", markdown).strip()

    # Clean residual artifacts
    markdown = _strip_artifacts(markdown)

    return markdown, metadata


def _extract_blackboard(raw_text: str) -> dict[str, Any] | None:
    """Extract and YAML-parse <blackboard> blocks from VLM output."""
    match = _BLACKBOARD_PATTERN.search(raw_text)
    if not match:
        return None

    raw_yaml = match.group(1)
    try:
        parsed = yaml.safe_load(raw_yaml)
    except yaml.YAMLError:
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


def _extract_confidence(text: str) -> ConfidenceLevel | None:
    match = _CONFIDENCE_PATTERN.search(text)
    if match:
        return ConfidenceLevel(match.group(1).upper())
    return None


def _strip_artifacts(markdown: str) -> str:
    """Remove common VLM output artifacts."""
    # Strip leading/trailing code fences that wrap entire output
    if markdown.startswith("```markdown"):
        markdown = markdown[len("```markdown") :]
    elif markdown.startswith("```md"):
        markdown = markdown[len("```md") :]
    elif markdown.startswith("```"):
        markdown = markdown[3:]

    if markdown.rstrip().endswith("```"):
        markdown = markdown.rstrip()[:-3]

    return markdown.strip()
