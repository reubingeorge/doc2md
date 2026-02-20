"""Tests for confidence report and aggregation."""

from doc2md.confidence.report import (
    aggregate_step_scores,
    needs_human_review,
    score_to_level,
)
from doc2md.types import ConfidenceLevel


class TestScoreToLevel:
    def test_high(self):
        assert score_to_level(0.85) == ConfidenceLevel.HIGH
        assert score_to_level(0.80) == ConfidenceLevel.HIGH

    def test_medium(self):
        assert score_to_level(0.65) == ConfidenceLevel.MEDIUM
        assert score_to_level(0.60) == ConfidenceLevel.MEDIUM

    def test_low(self):
        assert score_to_level(0.45) == ConfidenceLevel.LOW
        assert score_to_level(0.30) == ConfidenceLevel.LOW

    def test_failed(self):
        assert score_to_level(0.1) == ConfidenceLevel.FAILED
        assert score_to_level(0.0) == ConfidenceLevel.FAILED


class TestNeedsHumanReview:
    def test_high_no_review(self):
        assert needs_human_review(0.85) is False

    def test_medium_no_review(self):
        assert needs_human_review(0.65) is False

    def test_low_needs_review(self):
        assert needs_human_review(0.45) is True

    def test_failed_needs_review(self):
        assert needs_human_review(0.1) is True


class TestAggregateStepScores:
    def test_weighted_average_equal(self):
        scores = {"s1": 0.8, "s2": 0.6}
        result = aggregate_step_scores(scores, "weighted_average")
        assert abs(result - 0.7) < 0.01

    def test_weighted_average_with_weights(self):
        scores = {"s1": 0.8, "s2": 0.4}
        weights = {"s1": 0.7, "s2": 0.3}
        result = aggregate_step_scores(scores, "weighted_average", weights)
        # (0.8*0.7 + 0.4*0.3) / 1.0 = 0.68
        assert abs(result - 0.68) < 0.01

    def test_minimum_strategy(self):
        scores = {"s1": 0.9, "s2": 0.3, "s3": 0.7}
        result = aggregate_step_scores(scores, "minimum")
        assert result == 0.3

    def test_last_step_strategy(self):
        scores = {"s1": 0.4, "s2": 0.9}
        result = aggregate_step_scores(scores, "last_step")
        assert result == 0.9

    def test_empty_scores(self):
        assert aggregate_step_scores({}, "weighted_average") == 0.0

    def test_single_step(self):
        assert aggregate_step_scores({"s1": 0.75}, "weighted_average") == 0.75
