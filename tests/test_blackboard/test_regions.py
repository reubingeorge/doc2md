"""Tests for blackboard region Pydantic models."""

from doc2md.blackboard.regions import (
    VALID_REGIONS,
    DocumentMetadata,
    PageObservation,
    UncertainRegion,
)


class TestDocumentMetadata:
    def test_defaults(self):
        meta = DocumentMetadata()
        assert meta.language is None
        assert meta.layout is None
        assert meta.content_types == []

    def test_set_fields(self):
        meta = DocumentMetadata(language="fr", layout="two_column", page_count=10)
        assert meta.language == "fr"
        assert meta.layout == "two_column"
        assert meta.page_count == 10

    def test_extra_data(self):
        meta = DocumentMetadata(extra={"custom_key": "custom_val"})
        assert meta.extra["custom_key"] == "custom_val"


class TestPageObservation:
    def test_defaults(self):
        obs = PageObservation()
        assert obs.continues_on_next_page is False
        assert obs.quality_score is None
        assert obs.uncertain_regions == []

    def test_with_uncertain_regions(self):
        region = UncertainRegion(page=3, area="bottom_right", reason="blurry")
        obs = PageObservation(uncertain_regions=[region])
        assert len(obs.uncertain_regions) == 1
        assert obs.uncertain_regions[0].area == "bottom_right"

    def test_content_types(self):
        obs = PageObservation(content_types=["table", "prose"])
        assert "table" in obs.content_types


class TestUncertainRegion:
    def test_defaults(self):
        region = UncertainRegion()
        assert region.confidence == "low"
        assert region.area == ""

    def test_full(self):
        region = UncertainRegion(page=5, area="top_left", reason="faded", confidence="medium")
        assert region.page == 5
        assert region.confidence == "medium"


class TestValidRegions:
    def test_has_all_five(self):
        assert len(VALID_REGIONS) == 5
        expected = {
            "document_metadata",
            "page_observations",
            "step_outputs",
            "agent_notes",
            "confidence_signals",
        }
        assert expected == VALID_REGIONS
