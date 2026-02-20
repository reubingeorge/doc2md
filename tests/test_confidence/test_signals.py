"""Tests for individual confidence signals."""

from doc2md.confidence.signals.completeness import compute_completeness
from doc2md.confidence.signals.consistency import compute_consistency
from doc2md.confidence.signals.image_quality import compute_image_quality
from doc2md.confidence.signals.logprobs import compute_logprobs
from doc2md.confidence.signals.validation import compute_validation_pass_rate
from doc2md.confidence.signals.vlm_assessment import compute_vlm_self_assessment
from doc2md.types import ConfidenceLevel, ValidationRule


class TestVLMSelfAssessment:
    def test_high_confidence(self):
        score, available, reasoning = compute_vlm_self_assessment(ConfidenceLevel.HIGH)
        assert available is True
        assert score == 0.90

    def test_medium_confidence(self):
        score, available, _ = compute_vlm_self_assessment(ConfidenceLevel.MEDIUM)
        assert available is True
        assert score == 0.65

    def test_low_confidence(self):
        score, available, _ = compute_vlm_self_assessment(ConfidenceLevel.LOW)
        assert available is True
        assert score == 0.35

    def test_failed_confidence(self):
        score, available, _ = compute_vlm_self_assessment(ConfidenceLevel.FAILED)
        assert available is True
        assert score == 0.10

    def test_none_returns_unavailable(self):
        score, available, reasoning = compute_vlm_self_assessment(None)
        assert available is False
        assert "did not provide" in reasoning


class TestLogprobs:
    def test_high_probability_tokens(self):
        # logprob close to 0 means probability close to 1
        logprobs = [
            {"token": "hello", "logprob": -0.01},
            {"token": "world", "logprob": -0.02},
        ]
        score, available, _ = compute_logprobs(logprobs)
        assert available is True
        assert score > 0.9

    def test_low_probability_tokens(self):
        logprobs = [
            {"token": "x", "logprob": -3.0},
            {"token": "y", "logprob": -4.0},
        ]
        score, available, _ = compute_logprobs(logprobs)
        assert available is True
        assert score < 0.1

    def test_none_returns_unavailable(self):
        _, available, _ = compute_logprobs(None)
        assert available is False

    def test_empty_list_returns_unavailable(self):
        _, available, _ = compute_logprobs([])
        assert available is False

    def test_skips_special_tokens(self):
        logprobs = [
            {"token": "<|endoftext|>", "logprob": -10.0},
            {"token": "good", "logprob": -0.1},
        ]
        score, available, reasoning = compute_logprobs(logprobs)
        assert available is True
        assert "1 token" in reasoning


class TestValidationPassRate:
    def test_all_pass(self):
        md = "# Title\n\nSome content here that is long enough to pass min_length."
        rules = [
            ValidationRule(rule="has_header"),
            ValidationRule(rule="no_empty_output"),
        ]
        score, available, _ = compute_validation_pass_rate(md, rules)
        assert available is True
        assert score == 1.0

    def test_some_fail(self):
        md = "No header here"
        rules = [
            ValidationRule(rule="has_header"),
            ValidationRule(rule="no_empty_output"),
        ]
        score, available, reasoning = compute_validation_pass_rate(md, rules)
        assert available is True
        assert score == 0.5
        assert "has_header" in reasoning

    def test_empty_rules_unavailable(self):
        _, available, _ = compute_validation_pass_rate("text", [])
        assert available is False

    def test_min_length_with_params(self):
        md = "short"
        rules = [ValidationRule(rule="min_length", params={"min_chars": 100})]
        score, available, _ = compute_validation_pass_rate(md, rules)
        assert available is True
        assert score == 0.0

    def test_unknown_rule_excluded(self):
        rules = [
            ValidationRule(rule="no_empty_output"),
            ValidationRule(rule="unknown_rule_xyz"),
        ]
        score, available, _ = compute_validation_pass_rate("text", rules)
        assert available is True
        assert score == 1.0  # 1/1 known rules pass


class TestCompleteness:
    def test_all_fields_found(self):
        md = "# Invoice\nDate: 2024-01-15\nTotal: $100"
        score, available, _ = compute_completeness(md, ["invoice", "date", "total"])
        assert available is True
        assert score == 1.0

    def test_some_fields_missing(self):
        md = "# Invoice\nTotal: $100"
        score, available, reasoning = compute_completeness(md, ["invoice", "date", "total"])
        assert available is True
        assert 0.5 < score < 1.0
        assert "date" in reasoning

    def test_no_fields_configured(self):
        _, available, _ = compute_completeness("text", [])
        assert available is False

    def test_case_insensitive(self):
        md = "INVOICE TOTAL"
        score, _, _ = compute_completeness(md, ["invoice", "total"])
        assert score == 1.0


class TestConsistency:
    def test_identical_extractions(self):
        score, available, _ = compute_consistency("hello world", "hello world")
        assert available is True
        assert score == 1.0

    def test_different_extractions(self):
        score, available, _ = compute_consistency("hello world", "goodbye moon")
        assert available is True
        assert score < 1.0

    def test_none_returns_unavailable(self):
        _, available, _ = compute_consistency(None, None)
        assert available is False

    def test_one_none_unavailable(self):
        _, available, _ = compute_consistency("hello", None)
        assert available is False

    def test_both_empty(self):
        score, available, _ = compute_consistency("", "")
        assert available is True
        assert score == 1.0


class TestImageQuality:
    def test_valid_image(self, sample_image_bytes):
        score, available, reasoning = compute_image_quality(sample_image_bytes)
        assert available is True
        assert 0.0 <= score <= 1.0
        assert "Quality scores" in reasoning

    def test_none_returns_unavailable(self):
        _, available, _ = compute_image_quality(None)
        assert available is False

    def test_invalid_bytes_returns_fallback(self):
        score, available, _ = compute_image_quality(b"not an image")
        # Should handle gracefully
        assert isinstance(score, float)
