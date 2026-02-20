"""Code step: add YAML frontmatter to markdown."""

from __future__ import annotations

import re

from doc2md.pipeline.step_executor import register_code_step


@register_code_step("add_frontmatter")
def add_frontmatter(markdown: str, **kwargs: str) -> str:
    """Add YAML frontmatter to the beginning of markdown.

    Any keyword arguments become frontmatter fields.
    Skips if frontmatter already exists.
    """
    if markdown.startswith("---\n"):
        return markdown

    if not kwargs:
        return markdown

    # Build frontmatter
    lines = ["---"]
    for key, value in kwargs.items():
        # Sanitize: only allow simple string/number values
        safe_value = re.sub(r"[^\w\s\-_./]", "", str(value))
        lines.append(f"{key}: {safe_value}")
    lines.append("---")
    lines.append("")

    return "\n".join(lines) + markdown
