"""Markdown postprocessing — cleanup transforms applied after VLM output."""

from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

# Registry of postprocessing functions
_POSTPROCESS_REGISTRY: dict[str, Callable[..., str]] = {}


def _register(name: str) -> Callable:
    """Decorator to register a markdown postprocessing function."""
    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        _POSTPROCESS_REGISTRY[name] = fn
        return fn
    return decorator


def get_postprocess_fn(name: str) -> Callable[..., str] | None:
    """Look up a postprocessing function by name."""
    return _POSTPROCESS_REGISTRY.get(name)


# ── Individual transforms ──


@_register("normalize_headings")
def normalize_headings(markdown: str) -> str:
    """Ensure headings are properly formatted with consistent spacing.

    - Ensures a space after # characters
    - Normalizes heading levels (no jumps like h1 → h3)
    - Ensures blank lines before/after headings
    """
    lines = markdown.split("\n")
    result: list[str] = []
    prev_level = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Fix missing space after #
        heading_match = re.match(r"^(#{1,6})([^ #])", stripped)
        if heading_match:
            stripped = heading_match.group(1) + " " + stripped[len(heading_match.group(1)):]

        # Check if it's a heading
        heading_match = re.match(r"^(#{1,6})\s", stripped)
        if heading_match:
            level = len(heading_match.group(1))

            # Normalize level to avoid jumps > 1
            if prev_level > 0 and level > prev_level + 1:
                new_level = prev_level + 1
                stripped = "#" * new_level + stripped[level:]
                level = new_level

            prev_level = level

            # Ensure blank line before heading (if not at start)
            if result and result[-1].strip():
                result.append("")

            result.append(stripped)

            # Ensure blank line after heading
            if i + 1 < len(lines) and lines[i + 1].strip():
                result.append("")
            continue

        result.append(line)

    return "\n".join(result)


@_register("fix_table_alignment")
def fix_table_alignment(markdown: str) -> str:
    """Fix markdown table alignment by padding columns consistently."""
    lines = markdown.split("\n")
    result: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        if not table_lines:
            return
        aligned = _align_table(table_lines)
        result.extend(aligned)
        table_lines.clear()

    for line in lines:
        stripped = line.strip()
        if "|" in stripped and (stripped.startswith("|") or "---" in stripped):
            table_lines.append(stripped)
        else:
            flush_table()
            result.append(line)

    flush_table()
    return "\n".join(result)


def _align_table(table_lines: list[str]) -> list[str]:
    """Align a group of table lines to consistent column widths."""
    if len(table_lines) < 2:
        return table_lines

    # Parse cells
    rows: list[list[str]] = []
    separator_idx: int | None = None
    for i, line in enumerate(table_lines):
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
        if all(re.match(r":?-+:?$", c.strip()) for c in cells if c.strip()):
            separator_idx = i

    if not rows:
        return table_lines

    # Find max width per column
    max_cols = max(len(r) for r in rows)
    col_widths = [0] * max_cols
    for row in rows:
        for j, cell in enumerate(row):
            if j < max_cols:
                col_widths[j] = max(col_widths[j], len(cell))

    # Ensure minimum width of 3 for separator
    col_widths = [max(w, 3) for w in col_widths]

    # Rebuild table
    aligned: list[str] = []
    for i, row in enumerate(rows):
        padded: list[str] = []
        for j in range(max_cols):
            cell = row[j] if j < len(row) else ""
            if i == separator_idx:
                padded.append("-" * col_widths[j])
            else:
                padded.append(cell.ljust(col_widths[j]))
        aligned.append("| " + " | ".join(padded) + " |")

    return aligned


@_register("strip_artifacts")
def strip_artifacts(markdown: str, patterns: list[str] | None = None) -> str:
    """Remove common VLM artifacts from markdown output.

    Default patterns remove:
    - Page break markers
    - Repeated dashes/underscores (horizontal rules from scanning)
    - OCR artifacts like stray special characters
    """
    default_patterns = [
        r"---\s*Page\s+\d+\s*---",       # Page break markers
        r"^_{10,}$",                       # Long underscore lines
        r"^-{10,}$",                       # Long dash lines
        r"^={10,}$",                       # Long equals lines
        r"\[?\[image\]\]?",                # [image] placeholders
        r"<\|endoftext\|>",               # Model artifacts
    ]
    all_patterns = (patterns or []) + default_patterns

    result = markdown
    for pattern in all_patterns:
        result = re.sub(pattern, "", result, flags=re.MULTILINE)

    # Clean up excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


@_register("dedup_content")
def dedup_content(markdown: str) -> str:
    """Remove duplicate paragraphs/blocks from markdown.

    Detects exact duplicate paragraphs (common with multi-pass extraction
    or overlapping page regions) and removes them.
    """
    blocks = re.split(r"\n{2,}", markdown)
    seen: set[str] = set()
    unique: list[str] = []

    for block in blocks:
        normalized = block.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(block)

    return "\n\n".join(unique)


@_register("embed_confidence")
def embed_confidence(markdown: str, score: float | None = None) -> str:
    """Embed a confidence score as YAML frontmatter in the markdown."""
    if score is None:
        return markdown

    level = "HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.6 else "LOW" if score >= 0.3 else "FAILED"
    frontmatter = f"---\nconfidence: {score:.2f}\nconfidence_level: {level}\n---\n\n"

    # Don't double-add if frontmatter already exists
    if markdown.startswith("---\n"):
        return markdown

    return frontmatter + markdown


def validate_markdown(markdown: str) -> bool:
    """Validate that markdown output meets basic quality checks.

    Returns True if the markdown passes validation.
    """
    if not markdown or not markdown.strip():
        return False

    # Must have some actual content (not just whitespace/symbols)
    content = re.sub(r"[#\-_=|*>\s]", "", markdown)
    if len(content) < 10:
        return False

    # Check for unclosed code blocks
    code_blocks = markdown.count("```")
    if code_blocks % 2 != 0:
        return False

    return True


# ── Orchestrator ──


def run_postprocessing(markdown: str, steps: list[str]) -> str:
    """Run a sequence of postprocessing steps on markdown.

    Each step is identified by its registered name.
    """
    current = markdown
    for step_name in steps:
        fn = _POSTPROCESS_REGISTRY.get(step_name)
        if fn is None:
            logger.warning("Unknown postprocessing step '%s', skipping", step_name)
            continue
        try:
            current = fn(current)
        except Exception as e:
            logger.warning("Postprocessing step '%s' failed: %s, skipping", step_name, e)

    return current
