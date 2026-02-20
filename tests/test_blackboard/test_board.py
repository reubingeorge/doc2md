"""Tests for the core Blackboard class."""

import pytest

from doc2md.blackboard.board import Blackboard, BlackboardView
from doc2md.blackboard.regions import PageObservation


class TestBlackboardReadWrite:
    def test_write_and_read_document_metadata(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "fr", writer="meta_agent")
        assert bb.read("document_metadata", "language", reader="text_agent") == "fr"

    def test_write_and_read_page_observation(self):
        bb = Blackboard()
        bb.write("page_observations", "3.quality_score", 0.4, writer="agent")
        val = bb.read("page_observations", "3.quality_score", reader="reader")
        assert val == 0.4

    def test_write_page_observation_creates_page(self):
        bb = Blackboard()
        bb.write("page_observations", "5.continues_on_next_page", True, writer="agent")
        assert 5 in bb.page_observations
        assert bb.page_observations[5].continues_on_next_page is True

    def test_write_step_output(self):
        bb = Blackboard()
        bb.write("step_outputs", "extract", "# Title\nContent", writer="agent")
        assert bb.read("step_outputs", "extract") == "# Title\nContent"

    def test_write_agent_notes(self):
        bb = Blackboard()
        bb.write("agent_notes", "text_extract.unusual", "Roman numerals", writer="agent")
        assert bb.agent_notes["text_extract"]["unusual"] == "Roman numerals"

    def test_write_confidence_signals(self):
        bb = Blackboard()
        signals = {"self_assessment": 0.85, "validation": 0.9}
        bb.write("confidence_signals", "page_1.extract", signals, writer="engine")
        assert bb.confidence_signals["page_1.extract"]["self_assessment"] == 0.85

    def test_invalid_region_raises(self):
        bb = Blackboard()
        with pytest.raises(ValueError, match="Invalid blackboard region"):
            bb.write("nonexistent", "key", "value")

    def test_event_log_records_operations(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "en", writer="w")
        bb.read("document_metadata", "language", reader="r")
        assert len(bb.event_log) == 2
        writes = bb.event_log.query_writes()
        reads = bb.event_log.query_reads()
        assert len(writes) == 1
        assert len(reads) == 1


class TestBlackboardSubscribe:
    def test_subscribe_returns_view(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "fr", writer="agent")
        view = bb.subscribe(["document_metadata"])
        assert isinstance(view, BlackboardView)
        assert view.document_metadata["language"] == "fr"

    def test_subscribe_filters_regions(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "fr", writer="agent")
        bb.write("step_outputs", "extract", "content", writer="agent")
        view = bb.subscribe(["document_metadata"])
        with pytest.raises(AttributeError):
            _ = view.step_outputs

    def test_view_is_read_only_copy(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "fr", writer="agent")
        view = bb.subscribe(["document_metadata"])
        data = view.to_dict()
        data["document_metadata"]["language"] = "de"
        assert bb.document_metadata.language == "fr"  # Unchanged


class TestBlackboardSnapshot:
    def test_snapshot_contains_all_regions(self):
        bb = Blackboard()
        snap = bb.snapshot()
        assert "document_metadata" in snap
        assert "page_observations" in snap
        assert "step_outputs" in snap
        assert "agent_notes" in snap
        assert "confidence_signals" in snap

    def test_snapshot_reflects_writes(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "en", writer="agent")
        snap = bb.snapshot()
        assert snap["document_metadata"]["language"] == "en"

    def test_snapshot_is_frozen(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "en", writer="agent")
        snap = bb.snapshot()
        bb.write("document_metadata", "language", "fr", writer="agent")
        assert snap["document_metadata"]["language"] == "en"  # Snapshot unchanged


class TestBlackboardCopy:
    def test_copy_is_independent(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "en", writer="agent")
        copy = bb.copy()
        copy.write("document_metadata", "language", "fr", writer="agent")
        assert bb.document_metadata.language == "en"
        assert copy.document_metadata.language == "fr"

    def test_copy_has_own_event_log(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "en", writer="agent")
        copy = bb.copy()
        copy.write("step_outputs", "x", "y", writer="agent")
        assert len(bb.event_log) == 1
        assert len(copy.event_log) == 1  # Only the new write


class TestBlackboardQuery:
    def test_query_page_observations(self):
        bb = Blackboard()
        bb.write("page_observations", "1.quality_score", 0.9, writer="a")
        bb.write("page_observations", "2.quality_score", 0.3, writer="a")
        low_quality = bb.query(
            "page_observations",
            lambda obs: obs.quality_score is not None and obs.quality_score < 0.5,
        )
        assert len(low_quality) == 1


class TestBlackboardJinjaContext:
    def test_to_jinja_context(self):
        bb = Blackboard()
        bb.write("document_metadata", "language", "fr", writer="agent")
        ctx = bb.to_jinja_context(["document_metadata.language"])
        assert ctx["document_metadata"]["language"] == "fr"
