"""Built-in code-computed blackboard writers.

These run after a VLM call to derive deterministic blackboard values
from the extracted markdown or image, without additional VLM cost.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Registry of code-computed writers: function_name → (callable, output_key)
_WRITER_REGISTRY: dict[str, tuple[Callable[..., Any], str]] = {}


def blackboard_writer(output_key: str) -> Callable:
    """Decorator to register a code-computed blackboard writer."""
    def decorator(fn: Callable) -> Callable:
        _WRITER_REGISTRY[fn.__name__] = (fn, output_key)
        return fn
    return decorator


def get_writer(name: str) -> tuple[Callable[..., Any], str] | None:
    """Look up a registered writer by function name."""
    return _WRITER_REGISTRY.get(name)


def list_writers() -> list[str]:
    """Return all registered writer names."""
    return list(_WRITER_REGISTRY.keys())


# ── Built-in writers ──


@blackboard_writer("page_observations.{page_num}.continues_on_next_page")
def detect_continuations(markdown: str, page_num: int) -> bool:
    """Heuristic: does this page's markdown end mid-table or mid-sentence?"""
    stripped = markdown.rstrip()
    if not stripped:
        return False
    # Ends mid-table row
    if stripped.endswith("|"):
        return True
    # Ends without sentence-terminating punctuation
    if stripped[-1] not in ".!?\"')":
        return True
    return False


@blackboard_writer("page_observations.{page_num}.table_count")
def count_tables(markdown: str, page_num: int = 0) -> int:
    """Count the number of Markdown tables in the output."""
    # A table starts with a header row followed by a separator row
    table_sep_pattern = re.compile(r"^\|[\s:|-]+\|$", re.MULTILINE)
    return len(table_sep_pattern.findall(markdown))
