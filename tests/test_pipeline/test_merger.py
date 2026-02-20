"""Tests for output merger."""

from doc2md.config.schema import MergeConfig
from doc2md.pipeline.merger import merge_outputs


class TestMergeOutputs:
    def test_concatenation(self):
        outputs = {"text": "# Title\nContent", "tables": "| A | B |"}
        result = merge_outputs(outputs)
        assert "# Title\nContent" in result
        assert "| A | B |" in result

    def test_concatenation_with_config(self):
        config = MergeConfig(strategy="concatenate")
        outputs = {"a": "First", "b": "Second"}
        result = merge_outputs(outputs, config)
        assert "First" in result
        assert "Second" in result

    def test_single_output(self):
        result = merge_outputs({"only": "Just one"})
        assert result == "Just one"

    def test_empty_outputs(self):
        result = merge_outputs({})
        assert result == ""
