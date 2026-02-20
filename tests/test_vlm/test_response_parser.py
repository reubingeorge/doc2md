"""Tests for VLM response parser."""

from doc2md.types import ConfidenceLevel
from doc2md.vlm.response_parser import parse_response


class TestParseResponse:
    def test_plain_markdown(self):
        md, meta = parse_response("# Title\n\nSome text.")
        assert md == "# Title\n\nSome text."
        assert "blackboard_writes" not in meta

    def test_extracts_blackboard_block(self):
        raw = (
            "# Title\n\n"
            "<blackboard>\n"
            "page_observations:\n"
            "  3:\n"
            "    quality_score: 0.4\n"
            "</blackboard>\n"
            "\nMore text."
        )
        md, meta = parse_response(raw)
        assert "<blackboard>" not in md
        assert "blackboard_writes" in meta
        assert "page_observations" in meta["blackboard_writes"]
        assert meta["blackboard_writes"]["page_observations"][3]["quality_score"] == 0.4
        assert "# Title" in md
        assert "More text." in md

    def test_extracts_confidence_level_high(self):
        raw = "# Title\n[confidence: HIGH]\nContent."
        md, meta = parse_response(raw)
        assert meta["confidence_level"] == ConfidenceLevel.HIGH
        assert "[confidence:" not in md

    def test_extracts_confidence_level_low(self):
        _, meta = parse_response("Text [confidence: LOW] more")
        assert meta["confidence_level"] == ConfidenceLevel.LOW

    def test_strips_markdown_code_fences(self):
        raw = "```markdown\n# Title\nContent\n```"
        md, _ = parse_response(raw)
        assert md == "# Title\nContent"

    def test_strips_md_code_fences(self):
        raw = "```md\n# Title\n```"
        md, _ = parse_response(raw)
        assert md == "# Title"

    def test_strips_generic_code_fences(self):
        raw = "```\n# Title\n```"
        md, _ = parse_response(raw)
        assert md == "# Title"

    def test_no_metadata_when_none_present(self):
        _, meta = parse_response("Just plain text.")
        assert meta == {}
