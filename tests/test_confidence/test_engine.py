"""Tests for the confidence engine."""

from doc2md.confidence.engine import ConfidenceEngine
from doc2md.types import (
    AgentConfig,
    ConfidenceConfig,
    ConfidenceLevel,
    ModelConfig,
    PromptConfig,
    StepResult,
    TokenUsage,
    ValidationRule,
    VLMResponse,
)


def _make_agent_config(**overrides) -> AgentConfig:
    defaults = dict(
        name="test_agent",
        version="1.0",
        prompt=PromptConfig(system="sys", user="usr"),
        model=ModelConfig(preferred="gpt-4.1-mini"),
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _make_step_result(**overrides) -> StepResult:
    defaults = dict(
        step_name="extract",
        agent_name="test_agent",
        markdown="# Title\n\nSome content here that is long enough.",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        confidence_level=ConfidenceLevel.HIGH,
    )
    defaults.update(overrides)
    return StepResult(**defaults)


class TestConfidenceEngine:
    def test_compute_step_confidence_with_vlm_assessment(self):
        engine = ConfidenceEngine()
        config = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment"],
                weights={"vlm_self_assessment": 1.0},
            ),
        )
        result = _make_step_result(confidence_level=ConfidenceLevel.HIGH)
        report = engine.compute_step_confidence(result, config)
        assert report.calibrated_score == 0.90
        assert report.level == ConfidenceLevel.HIGH

    def test_compute_step_confidence_with_validation(self):
        engine = ConfidenceEngine()
        config = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["validation_pass_rate"],
                weights={"validation_pass_rate": 1.0},
            ),
            validation=[
                ValidationRule(rule="has_header"),
                ValidationRule(rule="no_empty_output"),
            ],
        )
        result = _make_step_result(markdown="# Title\n\nContent")
        report = engine.compute_step_confidence(result, config)
        assert report.calibrated_score == 1.0

    def test_compute_step_confidence_validation_fails(self):
        engine = ConfidenceEngine()
        config = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["validation_pass_rate"],
                weights={"validation_pass_rate": 1.0},
            ),
            validation=[
                ValidationRule(rule="has_header"),
                ValidationRule(rule="no_empty_output"),
            ],
        )
        result = _make_step_result(markdown="no header here")
        report = engine.compute_step_confidence(result, config)
        assert report.calibrated_score == 0.5  # 1/2 rules pass

    def test_compute_step_confidence_with_completeness(self):
        engine = ConfidenceEngine()
        config = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["completeness_check"],
                weights={"completeness_check": 1.0},
                expected_fields=["title", "date", "total"],
            ),
        )
        result = _make_step_result(markdown="# Title\nDate: 2024\nTotal: $100")
        report = engine.compute_step_confidence(result, config)
        assert report.calibrated_score == 1.0

    def test_multi_signal_combination(self):
        engine = ConfidenceEngine()
        config = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment", "validation_pass_rate"],
                weights={"vlm_self_assessment": 0.5, "validation_pass_rate": 0.5},
            ),
            validation=[ValidationRule(rule="has_header")],
        )
        result = _make_step_result(
            markdown="# Title\nContent",
            confidence_level=ConfidenceLevel.HIGH,
        )
        report = engine.compute_step_confidence(result, config)
        # vlm=0.9, validation=1.0, avg=(0.9*0.5 + 1.0*0.5) = 0.95
        assert abs(report.calibrated_score - 0.95) < 0.01

    def test_no_signals_configured_uses_defaults(self):
        engine = ConfidenceEngine()
        config = _make_agent_config()  # No confidence config
        result = _make_step_result(confidence_level=ConfidenceLevel.MEDIUM)
        report = engine.compute_step_confidence(result, config)
        # Should still work with default signals/weights
        assert isinstance(report.calibrated_score, float)

    def test_calibration_applied(self):
        engine = ConfidenceEngine()
        config = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment"],
                weights={"vlm_self_assessment": 1.0},
                calibration={
                    "method": "manual",
                    "manual_curve": [[0.5, 0.3], [0.9, 0.7], [1.0, 0.85]],
                },
            ),
        )
        result = _make_step_result(confidence_level=ConfidenceLevel.HIGH)
        report = engine.compute_step_confidence(result, config)
        # VLM HIGH → 0.90 raw, calibrated via curve
        assert report.raw_score == 0.90
        assert report.calibrated_score < 0.90  # Calibration reduces overconfidence

    def test_signals_in_report(self):
        engine = ConfidenceEngine()
        config = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment", "completeness_check"],
                weights={"vlm_self_assessment": 0.5, "completeness_check": 0.5},
                expected_fields=["title"],
            ),
        )
        result = _make_step_result(
            markdown="# Title\nContent",
            confidence_level=ConfidenceLevel.HIGH,
        )
        report = engine.compute_step_confidence(result, config)
        assert len(report.signals) == 2
        names = {s.name for s in report.signals}
        assert "vlm_self_assessment" in names
        assert "completeness_check" in names


class TestPipelineAggregation:
    def test_weighted_average(self):
        engine = ConfidenceEngine()
        config1 = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment"],
                weights={"vlm_self_assessment": 1.0},
            ),
        )
        r1 = _make_step_result(step_name="s1", confidence_level=ConfidenceLevel.HIGH)
        r2 = _make_step_result(step_name="s2", confidence_level=ConfidenceLevel.LOW)

        rep1 = engine.compute_step_confidence(r1, config1)
        rep2 = engine.compute_step_confidence(r2, config1)

        doc_report = engine.aggregate_pipeline(
            {"s1": rep1, "s2": rep2},
            strategy="weighted_average",
        )
        # (0.9 + 0.35) / 2 = 0.625
        assert abs(doc_report.overall - 0.625) < 0.01
        assert doc_report.level == ConfidenceLevel.MEDIUM

    def test_minimum_strategy(self):
        engine = ConfidenceEngine()
        config1 = _make_agent_config(
            confidence=ConfidenceConfig(
                signals=["vlm_self_assessment"],
                weights={"vlm_self_assessment": 1.0},
            ),
        )
        r1 = _make_step_result(step_name="s1", confidence_level=ConfidenceLevel.HIGH)
        r2 = _make_step_result(step_name="s2", confidence_level=ConfidenceLevel.LOW)

        rep1 = engine.compute_step_confidence(r1, config1)
        rep2 = engine.compute_step_confidence(r2, config1)

        doc_report = engine.aggregate_pipeline(
            {"s1": rep1, "s2": rep2},
            strategy="minimum",
        )
        assert doc_report.overall == 0.35
        assert doc_report.level == ConfidenceLevel.LOW
        assert doc_report.needs_human_review is True
