"""Built-in code step transforms — auto-registered on import."""

from doc2md.transforms.add_frontmatter import add_frontmatter
from doc2md.transforms.deduplicate_content import deduplicate_content
from doc2md.transforms.fix_table_alignment import fix_table_alignment
from doc2md.transforms.normalize_headings import normalize_headings
from doc2md.transforms.strip_page_numbers import strip_page_numbers

__all__ = [
    "strip_page_numbers",
    "normalize_headings",
    "fix_table_alignment",
    "deduplicate_content",
    "add_frontmatter",
]
