"""Tests for model allowlist."""

from doc2md.models.allowlist import ModelAllowlist, ModelInfo


class TestModelAllowlist:
    def test_loads_builtin_models(self):
        al = ModelAllowlist()
        assert len(al.model_names) > 0

    def test_has_default_model(self):
        al = ModelAllowlist()
        assert al.is_allowed("gpt-4.1-mini")

    def test_has_classifier_model(self):
        al = ModelAllowlist()
        assert al.is_allowed("gpt-4.1-nano")

    def test_unknown_model_not_allowed(self):
        al = ModelAllowlist()
        assert al.is_allowed("nonexistent-model") is False

    def test_get_returns_model_info(self):
        al = ModelAllowlist()
        info = al.get("gpt-4.1-mini")
        assert info is not None
        assert isinstance(info, ModelInfo)
        assert info.name == "gpt-4.1-mini"
        assert info.tier == "standard"

    def test_get_returns_none_for_unknown(self):
        al = ModelAllowlist()
        assert al.get("nonexistent") is None

    def test_list_models_sorted_by_priority(self):
        al = ModelAllowlist()
        models = al.list_models()
        priorities = [m.priority for m in models]
        assert priorities == sorted(priorities)

    def test_supports_logprobs(self):
        al = ModelAllowlist()
        assert al.supports_logprobs("gpt-4.1-mini") is True
        assert al.supports_logprobs("nonexistent") is False

    def test_get_by_tier(self):
        al = ModelAllowlist()
        standard = al.get_by_tier("standard")
        assert len(standard) >= 1
        assert all(m.tier == "standard" for m in standard)

    def test_custom_models_yaml(self, tmp_path):
        yaml_path = tmp_path / "custom.yaml"
        yaml_path.write_text(
            "models:\n"
            "  test-model:\n"
            "    tier: custom\n"
            "    priority: 1\n"
            "    logprobs: false\n"
        )
        al = ModelAllowlist(models_path=yaml_path)
        assert al.is_allowed("test-model")
        assert al.get("test-model").tier == "custom"

    def test_missing_yaml_file(self, tmp_path):
        al = ModelAllowlist(models_path=tmp_path / "nope.yaml")
        assert al.model_names == []
