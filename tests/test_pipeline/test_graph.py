"""Tests for pipeline DAG graph: parsing, topo sort, cycle detection, implicit deps."""

import pytest

from doc2md.config.schema import PipelineConfig, StepConfig
from doc2md.pipeline.graph import CycleError, StepGraph, parse_pipeline


def _make_pipeline(steps: list[StepConfig]) -> PipelineConfig:
    return PipelineConfig(name="test", steps=steps)


class TestStepGraph:
    def test_single_step(self):
        steps = [StepConfig(name="extract", agent="generic")]
        graph = StepGraph(steps)
        assert graph.topological_sort() == ["extract"]

    def test_implicit_dependency(self):
        steps = [
            StepConfig(name="extract", agent="generic"),
            StepConfig(name="validate", agent="validator"),
        ]
        graph = StepGraph(steps)
        order = graph.topological_sort()
        assert order.index("extract") < order.index("validate")

    def test_explicit_dependency(self):
        steps = [
            StepConfig(name="metadata", agent="metadata_extract"),
            StepConfig(name="extract", agent="generic", depends_on=["metadata"]),
            StepConfig(name="validate", agent="validator", depends_on=["extract"]),
        ]
        graph = StepGraph(steps)
        order = graph.topological_sort()
        assert order.index("metadata") < order.index("extract")
        assert order.index("extract") < order.index("validate")

    def test_empty_depends_on_no_implicit(self):
        steps = [
            StepConfig(name="a", agent="generic"),
            StepConfig(name="b", agent="generic", depends_on=[]),
        ]
        graph = StepGraph(steps)
        # b has no dependencies, should still appear in sort
        order = graph.topological_sort()
        assert "a" in order
        assert "b" in order

    def test_diamond_dependency(self):
        steps = [
            StepConfig(name="a", agent="generic", depends_on=[]),
            StepConfig(name="b", agent="generic", depends_on=["a"]),
            StepConfig(name="c", agent="generic", depends_on=["a"]),
            StepConfig(name="d", agent="generic", depends_on=["b", "c"]),
        ]
        graph = StepGraph(steps)
        order = graph.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detection(self):
        steps = [
            StepConfig(name="a", agent="generic", depends_on=["b"]),
            StepConfig(name="b", agent="generic", depends_on=["a"]),
        ]
        graph = StepGraph(steps)
        with pytest.raises(CycleError):
            graph.topological_sort()

    def test_unknown_dependency_raises(self):
        steps = [
            StepConfig(name="a", agent="generic", depends_on=["nonexistent"]),
        ]
        with pytest.raises(ValueError, match="unknown step 'nonexistent'"):
            StepGraph(steps)

    def test_dependencies_of(self):
        steps = [
            StepConfig(name="a", agent="generic", depends_on=[]),
            StepConfig(name="b", agent="generic", depends_on=["a"]),
        ]
        graph = StepGraph(steps)
        assert graph.dependencies_of("a") == []
        assert graph.dependencies_of("b") == ["a"]

    def test_nodes(self):
        steps = [
            StepConfig(name="x", agent="generic"),
            StepConfig(name="y", agent="generic"),
        ]
        graph = StepGraph(steps)
        assert set(graph.nodes) == {"x", "y"}


class TestParsePipeline:
    def test_parse_valid(self):
        config = _make_pipeline([StepConfig(name="extract", agent="generic")])
        graph = parse_pipeline(config)
        assert graph.topological_sort() == ["extract"]

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="at least one step"):
            parse_pipeline(PipelineConfig(name="empty", steps=[]))
