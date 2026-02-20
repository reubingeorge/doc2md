"""Code step: deduplicate content in markdown."""

from __future__ import annotations

from doc2md.pipeline.postprocessor import dedup_content
from doc2md.pipeline.step_executor import register_code_step


@register_code_step("deduplicate_content")
def deduplicate_content(markdown: str, **kwargs: str) -> str:
    """Remove duplicate paragraphs from markdown.

    Delegates to the postprocessor's dedup_content function.
    """
    return dedup_content(markdown)
