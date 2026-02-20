"""Tests for YAML config loading."""

import pytest

from doc2md.config.loader import load_agent_yaml, load_yaml
from doc2md.types import AgentConfig, InputMode


class TestLoadAgentYaml:
    def test_loads_valid_agent(self, sample_agent_yaml):
        config = load_agent_yaml(sample_agent_yaml)
        assert isinstance(config, AgentConfig)
        assert config.name == "test_agent"
        assert config.version == "1.0"
        assert config.model.preferred == "gpt-4.1-mini"
        assert config.model.max_tokens == 1024
        assert config.model.temperature == 0.0

    def test_defaults_applied(self, sample_agent_yaml):
        config = load_agent_yaml(sample_agent_yaml)
        assert config.input == InputMode.IMAGE
        assert config.retry.max_attempts == 3
        assert config.blackboard.reads == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_agent_yaml(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_no_agent_key(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("foo: bar\n")
        with pytest.raises(ValueError, match="missing top-level 'agent' key"):
            load_agent_yaml(path)

    def test_missing_required_fields(self, tmp_path):
        path = tmp_path / "incomplete.yaml"
        path.write_text("agent:\n  name: x\n")
        with pytest.raises(Exception):  # Pydantic validation
            load_agent_yaml(path)

    def test_full_agent_config(self, tmp_path):
        content = """
agent:
  name: full_agent
  version: "2.0"
  description: "Full config test"
  model:
    preferred: gpt-4.1
    fallback: [gpt-4o, gpt-4.1-mini]
    max_tokens: 8192
    temperature: 0.3
  input: image_and_previous
  prompt:
    system: "System prompt"
    user: "User prompt"
  blackboard:
    reads: [document_metadata.language]
    writes: [agent_notes.full_agent]
    writes_via: hybrid
  retry:
    max_attempts: 5
    strategy: linear
    retry_on: [rate_limit]
"""
        path = tmp_path / "full.yaml"
        path.write_text(content)
        config = load_agent_yaml(path)
        assert config.name == "full_agent"
        assert config.input == InputMode.IMAGE_AND_PREVIOUS
        assert config.model.fallback == ["gpt-4o", "gpt-4.1-mini"]
        assert "document_metadata.language" in config.blackboard.reads


class TestLoadYaml:
    def test_loads_dict(self, tmp_path):
        path = tmp_path / "test.yaml"
        path.write_text("key: value\nnested:\n  a: 1\n")
        result = load_yaml(path)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_rejects_non_dict(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            load_yaml(path)
