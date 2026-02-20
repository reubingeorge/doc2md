"""Tests for built-in code step transforms."""

# Import to trigger registration
import doc2md.transforms  # noqa: F401
from doc2md.pipeline.step_executor import get_code_step


class TestStripPageNumbers:
    def test_registered(self):
        assert get_code_step("strip_page_numbers") is not None

    def test_removes_page_n(self):
        fn = get_code_step("strip_page_numbers")
        result = fn("Content\nPage 3\nMore content")
        assert "Page 3" not in result
        assert "Content" in result
        assert "More content" in result

    def test_removes_dash_numbers(self):
        fn = get_code_step("strip_page_numbers")
        result = fn("Content\n- 5 -\nMore")
        assert "- 5 -" not in result

    def test_removes_bare_numbers(self):
        fn = get_code_step("strip_page_numbers")
        result = fn("Content\n42\nMore")
        assert "\n42\n" not in result

    def test_preserves_numbers_in_text(self):
        fn = get_code_step("strip_page_numbers")
        result = fn("There are 42 items in the list")
        assert "42" in result  # Not on its own line


class TestNormalizeHeadings:
    def test_registered(self):
        assert get_code_step("normalize_headings") is not None

    def test_normalizes(self):
        fn = get_code_step("normalize_headings")
        result = fn("#Title")
        assert "# Title" in result


class TestFixTableAlignment:
    def test_registered(self):
        assert get_code_step("fix_table_alignment") is not None

    def test_aligns(self):
        fn = get_code_step("fix_table_alignment")
        result = fn("| a | b |\n| --- | --- |\n| long | x |")
        assert "|" in result


class TestDeduplicateContent:
    def test_registered(self):
        assert get_code_step("deduplicate_content") is not None

    def test_deduplicates(self):
        fn = get_code_step("deduplicate_content")
        result = fn("Para\n\nPara")
        assert result.count("Para") == 1


class TestAddFrontmatter:
    def test_registered(self):
        assert get_code_step("add_frontmatter") is not None

    def test_adds_frontmatter(self):
        fn = get_code_step("add_frontmatter")
        result = fn("# Content", title="Test", author="Me")
        assert result.startswith("---\n")
        assert "title: Test" in result
        assert "author: Me" in result
        assert "# Content" in result

    def test_no_kwargs_passes_through(self):
        fn = get_code_step("add_frontmatter")
        result = fn("# Content")
        assert result == "# Content"

    def test_no_double_frontmatter(self):
        fn = get_code_step("add_frontmatter")
        result = fn("---\nexisting: true\n---\n# Content", title="Test")
        assert result.count("---") == 2  # Original frontmatter only
