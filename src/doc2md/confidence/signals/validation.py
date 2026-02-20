"""Validation pass rate signal — % of validation rules that pass."""

from __future__ import annotations

import re
from typing import Any

from doc2md.types import ValidationRule


# Built-in validation rules
def _rule_has_header(markdown: str, **params: Any) -> bool:
    """Check that the markdown contains at least one header."""
    return bool(re.search(r"^#{1,6}\s", markdown, re.MULTILINE))


def _rule_min_length(markdown: str, min_chars: int = 50, **params: Any) -> bool:
    """Check markdown meets a minimum character length."""
    return len(markdown.strip()) >= min_chars


def _rule_no_empty_output(markdown: str, **params: Any) -> bool:
    """Check markdown is not empty or whitespace-only."""
    return bool(markdown.strip())


def _rule_has_content_after_header(markdown: str, **params: Any) -> bool:
    """Check that headers are followed by content."""
    lines = markdown.strip().split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s", line) and i == len(lines) - 1:
            return False  # Header is last line with no content after
    return True


_BUILTIN_RULES: dict[str, Any] = {
    "has_header": _rule_has_header,
    "min_length": _rule_min_length,
    "no_empty_output": _rule_no_empty_output,
    "has_content_after_header": _rule_has_content_after_header,
}


def compute_validation_pass_rate(
    markdown: str,
    rules: list[ValidationRule],
) -> tuple[float, bool, str]:
    """Run validation rules and return pass rate.

    Returns (score, available, reasoning).
    """
    if not rules:
        return 0.0, False, "No validation rules configured"

    passed = 0
    total = len(rules)
    failed_names: list[str] = []

    for rule in rules:
        fn = _BUILTIN_RULES.get(rule.rule)
        if fn is None:
            total -= 1  # Unknown rule doesn't count
            continue
        try:
            if fn(markdown, **rule.params):
                passed += 1
            else:
                failed_names.append(rule.rule)
        except Exception:
            failed_names.append(rule.rule)

    if total == 0:
        return 0.0, False, "No recognized validation rules"

    score = passed / total
    reasoning = f"{passed}/{total} rules passed"
    if failed_names:
        reasoning += f"; failed: {', '.join(failed_names)}"

    return score, True, reasoning
