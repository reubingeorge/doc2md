"""Tests for Jinja2 prompt builder."""

from doc2md.types import AgentConfig, PromptConfig
from doc2md.vlm.prompt_builder import build_prompt


def _make_config(system: str, user: str) -> AgentConfig:
    return AgentConfig(
        name="test",
        prompt=PromptConfig(system=system, user=user),
    )


class TestBuildPrompt:
    def test_static_prompts(self):
        config = _make_config("You are a helper.", "Extract text.")
        system, user = build_prompt(config)
        assert system == "You are a helper."
        assert user == "Extract text."

    def test_jinja2_previous_output(self):
        config = _make_config(
            "System",
            "Previous: {{ previous_output }}",
        )
        _, user = build_prompt(config, previous_output="Hello world")
        assert "Hello world" in user

    def test_jinja2_blackboard_context(self):
        config = _make_config(
            "{% if bb.document_metadata.language %}Lang: {{ bb.document_metadata.language }}{% endif %}",
            "Extract.",
        )
        bb_ctx = {"document_metadata": {"language": "fr"}}
        system, _ = build_prompt(config, blackboard_context=bb_ctx)
        assert "Lang: fr" in system

    def test_conditional_blackboard_missing(self):
        config = _make_config(
            "{% if bb is defined and bb.document_metadata is defined %}Has metadata{% endif %}",
            "Extract.",
        )
        system, _ = build_prompt(config)
        assert "Has metadata" not in system

    def test_no_context_renders_cleanly(self):
        config = _make_config("Static system", "Static user")
        system, user = build_prompt(config)
        assert system == "Static system"
        assert user == "Static user"
