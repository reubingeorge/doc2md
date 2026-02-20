"""Tests for agent and pipeline registries."""

import pytest

from doc2md.agents.registry import AgentRegistry, PipelineRegistry
from doc2md.config.schema import PipelineConfig, StepConfig
from doc2md.types import AgentConfig, PromptConfig


class TestAgentRegistry:
    def test_discovers_builtin_agents(self):
        registry = AgentRegistry()
        agents = registry.list_agents()
        names = {a.name for a in agents}
        assert "generic" in names
        assert "text_extract" in names
        assert "table_extract" in names
        assert "validator" in names

    def test_get_existing_agent(self):
        registry = AgentRegistry()
        config = registry.get("generic")
        assert config.name == "generic"
        assert config.model.preferred is not None

    def test_get_missing_raises(self):
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_has(self):
        registry = AgentRegistry()
        assert registry.has("generic") is True
        assert registry.has("nonexistent") is False

    def test_user_dir_override(self, tmp_path):
        (tmp_path / "generic.yaml").write_text("""
agent:
  name: generic
  version: "99.0"
  description: "Custom override"
  model:
    preferred: custom-model
    max_tokens: 1024
    temperature: 0.0
  prompt:
    system: "Custom system"
    user: "Custom user"
""")
        registry = AgentRegistry(user_dirs=[tmp_path])
        config = registry.get("generic")
        assert config.version == "99.0"
        assert config.model.preferred == "custom-model"

    def test_register_programmatic(self):
        registry = AgentRegistry()
        custom = AgentConfig(
            name="custom_agent",
            prompt=PromptConfig(system="S", user="U"),
        )
        registry.register(custom)
        assert registry.has("custom_agent")
        assert registry.get("custom_agent").name == "custom_agent"

    def test_all_configs(self):
        registry = AgentRegistry()
        configs = registry.all_configs()
        assert isinstance(configs, dict)
        assert "generic" in configs

    def test_builtin_flag(self):
        registry = AgentRegistry()
        agents = registry.list_agents()
        generic_info = next(a for a in agents if a.name == "generic")
        assert generic_info.builtin is True


class TestPipelineRegistry:
    def test_discovers_builtin_pipelines(self):
        registry = PipelineRegistry()
        pipelines = registry.list_pipelines()
        names = {p.name for p in pipelines}
        assert "generic" in names
        assert "receipt" in names
        assert "academic" in names

    def test_get_existing_pipeline(self):
        registry = PipelineRegistry()
        config = registry.get("generic")
        assert config.name == "generic"
        assert len(config.steps) >= 1

    def test_get_missing_raises(self):
        registry = PipelineRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_has(self):
        registry = PipelineRegistry()
        assert registry.has("generic") is True
        assert registry.has("nonexistent") is False

    def test_register_programmatic(self):
        registry = PipelineRegistry()
        custom = PipelineConfig(
            name="custom_pipeline",
            steps=[StepConfig(name="s1", agent="generic")],
        )
        registry.register(custom)
        assert registry.has("custom_pipeline")

    def test_pipeline_info_has_step_count(self):
        registry = PipelineRegistry()
        pipelines = registry.list_pipelines()
        receipt = next(p for p in pipelines if p.name == "receipt")
        assert receipt.step_count == 2  # extract + validate
