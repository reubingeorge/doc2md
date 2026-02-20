"""Tests for model discovery (hardcoded allowlist only)."""

from doc2md.models.allowlist import ModelAllowlist, ModelInfo
from doc2md.models.discovery import ModelDiscovery


def _custom_allowlist(*model_tuples: tuple) -> ModelAllowlist:
    """Create an allowlist with custom models for testing."""
    allowlist = ModelAllowlist.__new__(ModelAllowlist)
    allowlist._models = {}
    for name, tier, priority, logprobs in model_tuples:
        allowlist._models[name] = ModelInfo(
            name=name, tier=tier, priority=priority, logprobs=logprobs,
        )
    return allowlist


class TestModelDiscovery:
    def test_validate_known_model(self):
        discovery = ModelDiscovery()
        valid, msg = discovery.validate_model("gpt-4.1-mini")
        assert valid is True
        assert msg == "OK"

    def test_validate_unknown_model(self):
        discovery = ModelDiscovery()
        valid, msg = discovery.validate_model("totally-fake-model")
        assert valid is False
        assert "not in the supported models list" in msg

    def test_available_models_from_allowlist(self):
        discovery = ModelDiscovery()
        models = discovery.available_models
        assert len(models) > 0
        names = [m.name for m in models]
        assert "gpt-4.1-mini" in names
        assert "gpt-4.1-nano" in names

    def test_get_best_available_returns_preferred(self):
        discovery = ModelDiscovery()
        best = discovery.get_best_available("gpt-4.1-mini")
        assert best == "gpt-4.1-mini"

    def test_get_best_available_falls_back(self):
        discovery = ModelDiscovery()
        best = discovery.get_best_available("totally-fake", ["gpt-4.1-mini"])
        assert best == "gpt-4.1-mini"

    def test_get_best_available_no_valid_returns_preferred(self):
        discovery = ModelDiscovery()
        best = discovery.get_best_available("fake-1", ["fake-2"])
        assert best == "fake-1"  # Returns preferred as last resort

    def test_supports_logprobs(self):
        discovery = ModelDiscovery()
        assert discovery.supports_logprobs("gpt-4.1-mini") is True
        assert discovery.supports_logprobs("gpt-5.2") is False
        assert discovery.supports_logprobs("nonexistent") is False

    def test_get_by_tier(self):
        discovery = ModelDiscovery()
        economy = discovery.get_by_tier("economy")
        assert len(economy) > 0
        assert all(m.tier == "economy" for m in economy)

    def test_custom_allowlist(self):
        allowlist = _custom_allowlist(
            ("model-a", "standard", 1, True),
            ("model-b", "premium", 2, False),
        )
        discovery = ModelDiscovery(allowlist=allowlist)

        valid_a, _ = discovery.validate_model("model-a")
        assert valid_a is True

        valid_c, _ = discovery.validate_model("model-c")
        assert valid_c is False

        assert len(discovery.available_models) == 2

    def test_all_yaml_models_are_valid(self):
        """Every model in models.yaml should validate."""
        discovery = ModelDiscovery()
        for model in discovery.available_models:
            valid, msg = discovery.validate_model(model.name)
            assert valid is True, f"{model.name}: {msg}"
