"""Completeness check signal — expected fields present in output."""

from __future__ import annotations

import re


def compute_completeness(
    markdown: str,
    expected_fields: list[str],
) -> tuple[float, bool, str]:
    """Check how many expected fields appear in the markdown output.

    Fields are matched case-insensitively as substrings or markdown headers.

    Returns (score, available, reasoning).
    """
    if not expected_fields:
        return 0.0, False, "No expected fields configured"

    found: list[str] = []
    missing: list[str] = []
    lower_md = markdown.lower()

    for field in expected_fields:
        # Check as substring (case-insensitive)
        if field.lower() in lower_md:
            found.append(field)
        # Also check as a markdown header variant
        elif re.search(rf"#{1,6}\s+.*{re.escape(field)}", markdown, re.IGNORECASE):
            found.append(field)
        else:
            missing.append(field)

    score = len(found) / len(expected_fields)
    reasoning = f"{len(found)}/{len(expected_fields)} expected fields found"
    if missing:
        reasoning += f"; missing: {', '.join(missing)}"

    return score, True, reasoning
