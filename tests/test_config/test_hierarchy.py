"""Tests for config hierarchy."""

import os

import pytest

from doc2md.config.hierarchy import (
    _coerce_env_value,
    _load_yaml_config,
    load_config_hierarchy,
)


class TestLoadConfigHierarchy:
    def test_returns_defaults(self):
        config = load_config_hierarchy()
        assert config["model"] == "gpt-4.1-mini"
        assert config["max_workers"] == 5

    def test_runtime_overrides(self):
        config = load_config_hierarchy(model="gpt-4o", max_workers=10)
        assert config["model"] == "gpt-4o"
        assert config["max_workers"] == 10

    def test_none_overrides_ignored(self):
        config = load_config_hierarchy(model=None)
        assert config["model"] == "gpt-4.1-mini"  # Default preserved

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("DOC2MD_MODEL", "gpt-4.1")
        config = load_config_hierarchy()
        assert config["model"] == "gpt-4.1"

    def test_openai_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = load_config_hierarchy()
        assert config["api_key"] == "sk-test-key"

    def test_runtime_beats_env(self, monkeypatch):
        monkeypatch.setenv("DOC2MD_MODEL", "gpt-4.1")
        config = load_config_hierarchy(model="gpt-4o")
        assert config["model"] == "gpt-4o"  # Runtime wins

    def test_env_numeric_coercion(self, monkeypatch):
        monkeypatch.setenv("DOC2MD_MAX_WORKERS", "10")
        config = load_config_hierarchy()
        assert config["max_workers"] == 10
        assert isinstance(config["max_workers"], int)

    def test_env_bool_coercion(self, monkeypatch):
        monkeypatch.setenv("DOC2MD_CACHE_DISABLED", "true")
        config = load_config_hierarchy()
        assert config["cache_disabled"] is True

    def test_project_config(self, tmp_path, monkeypatch):
        config_file = tmp_path / "doc2md.yaml"
        config_file.write_text("model: gpt-4.1\nmax_workers: 8\n")
        monkeypatch.chdir(tmp_path)
        config = load_config_hierarchy()
        assert config["model"] == "gpt-4.1"
        assert config["max_workers"] == 8


class TestLoadYamlConfig:
    def test_loads_valid_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("key: value\n")
        result = _load_yaml_config(path)
        assert result == {"key": "value"}

    def test_returns_none_for_missing(self, tmp_path):
        result = _load_yaml_config(tmp_path / "nonexistent.yaml")
        assert result is None

    def test_returns_none_for_non_dict(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- item1\n- item2\n")
        result = _load_yaml_config(path)
        assert result is None


class TestCoerceEnvValue:
    def test_bool_true(self):
        assert _coerce_env_value("cache_disabled", "true") is True
        assert _coerce_env_value("cache_disabled", "1") is True
        assert _coerce_env_value("cache_disabled", "yes") is True

    def test_bool_false(self):
        assert _coerce_env_value("cache_disabled", "false") is False
        assert _coerce_env_value("cache_disabled", "0") is False

    def test_int_coercion(self):
        assert _coerce_env_value("max_workers", "10") == 10

    def test_float_coercion(self):
        assert _coerce_env_value("cache_memory_mb", "256.5") == 256.5

    def test_string_passthrough(self):
        assert _coerce_env_value("api_key", "sk-123") == "sk-123"
