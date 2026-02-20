"""Code step: fix table alignment in markdown."""

from __future__ import annotations

from doc2md.pipeline.postprocessor import fix_table_alignment as _fix_tables
from doc2md.pipeline.step_executor import register_code_step


@register_code_step("fix_table_alignment")
def fix_table_alignment(markdown: str, **kwargs: str) -> str:
    """Align markdown table columns consistently.

    Delegates to the postprocessor's fix_table_alignment function.
    """
    return _fix_tables(markdown)
