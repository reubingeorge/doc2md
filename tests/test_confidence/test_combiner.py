"""Tests for signal combiner with adaptive weight redistribution."""

from doc2md.confidence.combiner import SignalResult, _redistribute_weights, combine_signals


class TestCombineSignals:
    def test_all_available(self):
        signals = [
            SignalResult(name="a", score=0.8, available=True),
            SignalResult(name="b", score=0.6, available=True),
        ]
        weights = {"a": 0.5, "b": 0.5}
        score, effective = combine_signals(signals, weights)
        assert score == 0.7  # (0.8*0.5 + 0.6*0.5)
        assert effective["a"] == 0.5
        assert effective["b"] == 0.5

    def test_one_unavailable_redistributes(self):
        signals = [
            SignalResult(name="a", score=0.8, available=True),
            SignalResult(name="b", score=0.0, available=False),
            SignalResult(name="c", score=0.6, available=True),
        ]
        weights = {"a": 0.4, "b": 0.2, "c": 0.4}
        score, effective = combine_signals(signals, weights)
        # b is unavailable, its weight redistributed proportionally
        # effective a = 0.4/0.8 = 0.5, effective c = 0.4/0.8 = 0.5
        assert abs(effective["a"] - 0.5) < 0.01
        assert abs(effective["c"] - 0.5) < 0.01
        assert abs(score - 0.7) < 0.01  # 0.8*0.5 + 0.6*0.5

    def test_no_available_signals(self):
        signals = [
            SignalResult(name="a", score=0.8, available=False),
        ]
        score, effective = combine_signals(signals, {"a": 1.0})
        assert score == 0.0
        assert effective == {}

    def test_empty_signals(self):
        score, effective = combine_signals([], {})
        assert score == 0.0

    def test_missing_weight_treated_as_zero(self):
        signals = [
            SignalResult(name="a", score=0.8, available=True),
            SignalResult(name="b", score=0.6, available=True),
        ]
        # Only weight for 'a', none for 'b'
        weights = {"a": 1.0}
        score, effective = combine_signals(signals, weights)
        # b has weight 0, gets 0/1.0 = 0
        assert effective["a"] == 1.0
        assert effective["b"] == 0.0


class TestRedistributeWeights:
    def test_all_available(self):
        result = _redistribute_weights({"a": 0.5, "b": 0.5}, {"a", "b"})
        assert abs(result["a"] - 0.5) < 0.01
        assert abs(result["b"] - 0.5) < 0.01

    def test_one_removed(self):
        result = _redistribute_weights({"a": 0.25, "b": 0.25, "c": 0.50}, {"a", "c"})
        # a: 0.25/0.75 = 0.333, c: 0.50/0.75 = 0.667
        assert abs(result["a"] - 0.333) < 0.01
        assert abs(result["c"] - 0.667) < 0.01

    def test_empty_available(self):
        result = _redistribute_weights({"a": 1.0}, set())
        assert result == {}

    def test_equal_fallback_when_no_weights(self):
        result = _redistribute_weights({}, {"a", "b"})
        assert abs(result["a"] - 0.5) < 0.01
        assert abs(result["b"] - 0.5) < 0.01
