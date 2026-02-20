"""Tests for blackboard serializer."""

from doc2md.blackboard.board import Blackboard
from doc2md.blackboard.serializer import serialize_for_prompt


class TestSerializeForPrompt:
    def test_full_region(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "fr", writer="a")
        bb.write("document_metadata", "layout", "two_column", writer="a")
        result = serialize_for_prompt(bb, ["document_metadata"])
        assert result["document_metadata"]["language"] == "fr"
        assert result["document_metadata"]["layout"] == "two_column"

    def test_single_field_subscription(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "fr", writer="a")
        bb.write("document_metadata", "layout", "two_column", writer="a")
        result = serialize_for_prompt(bb, ["document_metadata.language"])
        assert result["document_metadata"]["language"] == "fr"
        # Layout should not be present since we only subscribed to language
        assert "layout" not in result["document_metadata"]

    def test_wildcard_page_observations(self):
        bb = Blackboard()
        bb.write("page_observations", "1.quality_score", 0.9, writer="a")
        bb.write("page_observations", "2.quality_score", 0.3, writer="a")
        result = serialize_for_prompt(bb, ["page_observations.*.quality_score"])
        assert result["page_observations"][1]["quality_score"] == 0.9
        assert result["page_observations"][2]["quality_score"] == 0.3

    def test_step_outputs_subscription(self):
        bb = Blackboard()
        bb.write("step_outputs", "extract", "# Content", writer="a")
        bb.write("step_outputs", "validate", "OK", writer="a")
        result = serialize_for_prompt(bb, ["step_outputs.extract"])
        assert result["step_outputs"]["extract"] == "# Content"
        assert "validate" not in result["step_outputs"]

    def test_empty_subscriptions(self):
        bb = Blackboard()
        result = serialize_for_prompt(bb, [])
        assert result == {}

    def test_missing_data_no_error(self):
        bb = Blackboard()
        result = serialize_for_prompt(bb, ["document_metadata.language"])
        # Language is None so it won't appear
        assert result.get("document_metadata", {}).get("language") is None
