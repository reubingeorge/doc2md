"""Tests for package defaults."""

from doc2md.config.defaults import (
    DEFAULT_CACHE_DISABLED,
    DEFAULT_CACHE_DISK_MB,
    DEFAULT_CACHE_MEMORY_MB,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    get_defaults,
)


class TestDefaults:
    def test_default_model(self):
        assert DEFAULT_MODEL == "gpt-4.1-mini"

    def test_default_temperature(self):
        assert DEFAULT_TEMPERATURE == 0.0

    def test_default_cache_not_disabled(self):
        assert DEFAULT_CACHE_DISABLED is False

    def test_default_max_workers(self):
        assert DEFAULT_MAX_WORKERS == 5

    def test_default_log_level(self):
        assert DEFAULT_LOG_LEVEL == "WARNING"

    def test_get_defaults_returns_dict(self):
        d = get_defaults()
        assert isinstance(d, dict)
        assert d["model"] == "gpt-4.1-mini"
        assert d["cache_memory_mb"] == DEFAULT_CACHE_MEMORY_MB
        assert d["cache_disk_mb"] == DEFAULT_CACHE_DISK_MB
        assert d["max_workers"] == 5

    def test_get_defaults_has_all_keys(self):
        d = get_defaults()
        expected_keys = {
            "model", "classifier_model", "max_tokens", "temperature",
            "cache_memory_mb", "cache_disk_mb", "cache_disabled",
            "max_workers", "rpm_limit", "tpm_limit",
            "max_retries", "retry_strategy", "log_level",
        }
        assert expected_keys == set(d.keys())
