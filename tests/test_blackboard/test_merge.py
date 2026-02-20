"""Tests for parallel blackboard merge logic."""

from doc2md.blackboard.board import Blackboard
from doc2md.blackboard.merge import merge_parallel
from doc2md.blackboard.regions import PageObservation, UncertainRegion


class TestMergeParallel:
    def test_merge_document_metadata(self):
        target = Blackboard()
        target.write("document_metadata", "language", "fr", writer="pre")

        branch = target.copy()
        branch.write("document_metadata", "layout", "two_column", writer="b")

        merge_parallel(target, [branch])
        assert target.document_metadata.language == "fr"
        assert target.document_metadata.layout == "two_column"

    def test_merge_page_observations_new_pages(self):
        target = Blackboard()
        branch_a = target.copy()
        branch_b = target.copy()

        branch_a.write("page_observations", "1.quality_score", 0.9, writer="a")
        branch_b.write("page_observations", "2.quality_score", 0.4, writer="b")

        merge_parallel(target, [branch_a, branch_b])
        assert target.page_observations[1].quality_score == 0.9
        assert target.page_observations[2].quality_score == 0.4

    def test_merge_page_observations_same_page_different_fields(self):
        target = Blackboard()
        branch_a = target.copy()
        branch_b = target.copy()

        branch_a.write("page_observations", "3.quality_score", 0.4, writer="a")
        branch_b.write("page_observations", "3.table_count", 2, writer="b")

        merge_parallel(target, [branch_a, branch_b])
        assert target.page_observations[3].quality_score == 0.4
        assert target.page_observations[3].table_count == 2

    def test_merge_step_outputs(self):
        target = Blackboard()
        branch_a = target.copy()
        branch_b = target.copy()

        branch_a.write("step_outputs", "text_extract", "# Text", writer="a")
        branch_b.write("step_outputs", "table_extract", "| A | B |", writer="b")

        merge_parallel(target, [branch_a, branch_b])
        assert target.step_outputs["text_extract"] == "# Text"
        assert target.step_outputs["table_extract"] == "| A | B |"

    def test_merge_agent_notes(self):
        target = Blackboard()
        branch = target.copy()
        branch.write("agent_notes", "handwriting.found_signature", True, writer="hw")

        merge_parallel(target, [branch])
        assert target.agent_notes["handwriting"]["found_signature"] is True

    def test_merge_confidence_signals(self):
        target = Blackboard()
        branch = target.copy()
        branch.write(
            "confidence_signals", "page_1.extract", {"self_assessment": 0.85}, writer="engine"
        )
        merge_parallel(target, [branch])
        assert target.confidence_signals["page_1.extract"]["self_assessment"] == 0.85

    def test_merge_uncertain_regions_appended(self):
        target = Blackboard()
        target.page_observations[1] = PageObservation(
            uncertain_regions=[UncertainRegion(area="top")]
        )
        branch = target.copy()
        branch.page_observations[1].uncertain_regions.append(UncertainRegion(area="bottom"))
        merge_parallel(target, [branch])
        assert len(target.page_observations[1].uncertain_regions) == 2
