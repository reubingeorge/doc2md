"""Code step: strip page numbers from markdown."""

from __future__ import annotations

import re

from doc2md.pipeline.step_executor import register_code_step


@register_code_step("strip_page_numbers")
def strip_page_numbers(markdown: str, **kwargs: str) -> str:
    """Remove standalone page numbers from markdown.

    Matches patterns like:
    - "Page 3" or "page 3" on its own line
    - "- 3 -" centered page numbers
    - Bare numbers on their own line (1-4 digits)
    """
    patterns = [
        r"^\s*[Pp]age\s+\d+\s*$",  # "Page 3"
        r"^\s*-\s*\d+\s*-\s*$",  # "- 3 -"
        r"^\s*\d{1,4}\s*$",  # bare "3"
    ]

    lines = markdown.split("\n")
    result: list[str] = []
    for line in lines:
        if any(re.match(p, line) for p in patterns):
            continue
        result.append(line)

    return "\n".join(result)
