"""Tests that all built-in agent and pipeline YAMLs load and validate."""

from pathlib import Path

import pytest

from doc2md.agents.registry import AgentRegistry, PipelineRegistry

_EXPECTED_AGENTS = [
    "_classifier", "_page_classifier", "generic", "text_extract",
    "table_extract", "handwriting", "metadata_extract", "summarize",
    "validator", "document_merger",
]

_EXPECTED_PIPELINES = [
    "generic", "receipt", "structured_pdf", "academic",
    "legal_contract", "handwritten", "mixed_document",
]


class TestBuiltinAgents:
    @pytest.fixture(scope="class")
    def registry(self):
        return AgentRegistry()

    @pytest.mark.parametrize("agent_name", _EXPECTED_AGENTS)
    def test_agent_loads(self, registry, agent_name):
        assert registry.has(agent_name), f"Agent '{agent_name}' not found"
        config = registry.get(agent_name)
        assert config.name == agent_name
        assert config.prompt.system  # Has a system prompt
        assert config.prompt.user    # Has a user prompt
        assert config.model.preferred  # Has a model set

    def test_all_expected_agents_present(self, registry):
        names = {a.name for a in registry.list_agents()}
        for expected in _EXPECTED_AGENTS:
            assert expected in names, f"Missing built-in agent: {expected}"


class TestBuiltinPipelines:
    @pytest.fixture(scope="class")
    def registry(self):
        return PipelineRegistry()

    @pytest.mark.parametrize("pipeline_name", _EXPECTED_PIPELINES)
    def test_pipeline_loads(self, registry, pipeline_name):
        assert registry.has(pipeline_name), f"Pipeline '{pipeline_name}' not found"
        config = registry.get(pipeline_name)
        assert config.name == pipeline_name
        assert len(config.steps) >= 1

    def test_all_expected_pipelines_present(self, registry):
        names = {p.name for p in registry.list_pipelines()}
        for expected in _EXPECTED_PIPELINES:
            assert expected in names, f"Missing built-in pipeline: {expected}"

    def test_generic_pipeline_single_step(self, registry):
        config = registry.get("generic")
        assert len(config.steps) == 1
        assert config.steps[0].agent == "generic"

    def test_receipt_pipeline_two_steps(self, registry):
        config = registry.get("receipt")
        assert len(config.steps) == 2
        step_names = [s.name for s in config.steps]
        assert "extract" in step_names
        assert "validate" in step_names
