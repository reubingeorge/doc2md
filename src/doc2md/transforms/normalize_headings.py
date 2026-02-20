"""Code step: normalize headings in markdown."""

from __future__ import annotations

from doc2md.pipeline.postprocessor import normalize_headings as _normalize
from doc2md.pipeline.step_executor import register_code_step


@register_code_step("normalize_headings")
def normalize_headings(markdown: str, **kwargs: str) -> str:
    """Normalize heading formatting and levels.

    Delegates to the postprocessor's normalize_headings function.
    """
    return _normalize(markdown)
