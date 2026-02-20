"""Tests for markdown postprocessing pipeline."""

from doc2md.pipeline.postprocessor import (
    dedup_content,
    embed_confidence,
    fix_table_alignment,
    normalize_headings,
    run_postprocessing,
    strip_artifacts,
    validate_markdown,
)


class TestNormalizeHeadings:
    def test_adds_space_after_hash(self):
        result = normalize_headings("#Title")
        assert "# Title" in result

    def test_blank_lines_around_heading(self):
        result = normalize_headings("text\n# Heading\nmore text")
        lines = result.split("\n")
        heading_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("# "))
        # Blank line before heading
        assert lines[heading_idx - 1].strip() == ""
        # Blank line after heading
        assert lines[heading_idx + 1].strip() == ""

    def test_no_heading_jump(self):
        result = normalize_headings("# H1\n\n### H3")
        assert "## H3" in result  # Should normalize to h2

    def test_preserves_non_heading_content(self):
        text = "Just a paragraph\n\nAnother paragraph"
        result = normalize_headings(text)
        assert "Just a paragraph" in result
        assert "Another paragraph" in result


class TestFixTableAlignment:
    def test_aligns_columns(self):
        table = "| a | b |\n| --- | --- |\n| long text | x |"
        result = fix_table_alignment(table)
        lines = result.strip().split("\n")
        # All rows should have same number of | characters
        pipe_counts = [line.count("|") for line in lines]
        assert len(set(pipe_counts)) == 1

    def test_preserves_non_table_content(self):
        text = "# Heading\n\nSome text\n"
        result = fix_table_alignment(text)
        assert "# Heading" in result
        assert "Some text" in result

    def test_handles_empty_cells(self):
        table = "| a | |\n| --- | --- |\n| | b |"
        result = fix_table_alignment(table)
        assert "|" in result


class TestStripArtifacts:
    def test_removes_page_markers(self):
        text = "Content\n--- Page 1 ---\nMore content"
        result = strip_artifacts(text)
        assert "Page 1" not in result
        assert "Content" in result
        assert "More content" in result

    def test_removes_long_dash_lines(self):
        text = "Content\n--------------\nMore"
        result = strip_artifacts(text)
        assert "----------" not in result

    def test_removes_image_placeholders(self):
        text = "Text [image] more text"
        result = strip_artifacts(text)
        assert "[image]" not in result

    def test_removes_endoftext(self):
        text = "Content <|endoftext|>"
        result = strip_artifacts(text)
        assert "<|endoftext|>" not in result

    def test_custom_patterns(self):
        text = "Content WATERMARK more"
        result = strip_artifacts(text, patterns=["WATERMARK"])
        assert "WATERMARK" not in result

    def test_cleans_excessive_blank_lines(self):
        text = "A\n\n\n\n\nB"
        result = strip_artifacts(text)
        assert "\n\n\n" not in result


class TestDedupContent:
    def test_removes_duplicate_paragraphs(self):
        text = "Para 1\n\nPara 2\n\nPara 1"
        result = dedup_content(text)
        assert result.count("Para 1") == 1
        assert "Para 2" in result

    def test_preserves_unique_content(self):
        text = "First\n\nSecond\n\nThird"
        result = dedup_content(text)
        assert "First" in result
        assert "Second" in result
        assert "Third" in result

    def test_empty_input(self):
        assert dedup_content("") == ""


class TestEmbedConfidence:
    def test_adds_frontmatter(self):
        result = embed_confidence("# Content", score=0.85)
        assert result.startswith("---\n")
        assert "confidence: 0.85" in result
        assert "confidence_level: HIGH" in result

    def test_no_double_frontmatter(self):
        text = "---\ntitle: test\n---\n\n# Content"
        result = embed_confidence(text, score=0.5)
        assert result == text  # Should not add again

    def test_none_score_passes_through(self):
        text = "# Content"
        result = embed_confidence(text)
        assert result == text

    def test_level_thresholds(self):
        assert "HIGH" in embed_confidence("x", score=0.9)
        assert "MEDIUM" in embed_confidence("x", score=0.7)
        assert "LOW" in embed_confidence("x", score=0.4)
        assert "FAILED" in embed_confidence("x", score=0.1)


class TestValidateMarkdown:
    def test_valid_markdown(self):
        assert validate_markdown("# Title\n\nSome content here with text") is True

    def test_empty_fails(self):
        assert validate_markdown("") is False
        assert validate_markdown("   ") is False

    def test_symbols_only_fails(self):
        assert validate_markdown("### ---") is False

    def test_unclosed_code_block_fails(self):
        assert validate_markdown("```python\ncode here\nmore content text") is False

    def test_closed_code_block_passes(self):
        assert validate_markdown("```python\ncode here\n```\nSome real content text") is True


class TestRunPostprocessing:
    def test_empty_steps(self):
        text = "# Content"
        result = run_postprocessing(text, [])
        assert result == text

    def test_single_step(self):
        text = "#Title"
        result = run_postprocessing(text, ["normalize_headings"])
        assert "# Title" in result

    def test_multiple_steps(self):
        text = "#Title\n\nPara\n\nPara"
        result = run_postprocessing(text, ["normalize_headings", "dedup_content"])
        assert "# Title" in result
        assert result.count("Para") == 1

    def test_unknown_step_skipped(self):
        text = "# Content"
        result = run_postprocessing(text, ["nonexistent"])
        assert result == text

    def test_chain_preserves_order(self):
        text = "Content <|endoftext|>\n\nContent <|endoftext|>"
        result = run_postprocessing(text, ["strip_artifacts", "dedup_content"])
        assert "<|endoftext|>" not in result
        # After strip + dedup, should have just one "Content"
        assert result.count("Content") == 1
