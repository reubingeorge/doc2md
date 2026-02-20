"""Jinja2-based prompt builder for agent configs."""

from __future__ import annotations

from jinja2.sandbox import SandboxedEnvironment

from doc2md.types import AgentConfig

from jinja2 import ChainableUndefined

_jinja_env = SandboxedEnvironment(
    autoescape=False,
    keep_trailing_newline=True,
    undefined=ChainableUndefined,
)


def build_prompt(
    agent_config: AgentConfig,
    image_b64: str | None = None,
    previous_output: str | None = None,
    blackboard_context: dict | None = None,
) -> tuple[str, str]:
    """Render system and user prompts from an agent config.

    Returns (system_prompt, user_prompt).
    """
    context = _build_context(previous_output, blackboard_context)
    system = _render_template(agent_config.prompt.system, context)
    user = _render_template(agent_config.prompt.user, context)
    return system, user


def _build_context(
    previous_output: str | None,
    blackboard_context: dict | None,
) -> dict:
    ctx: dict = {}
    if previous_output is not None:
        ctx["previous_output"] = previous_output
    if blackboard_context:
        ctx["bb"] = blackboard_context
    return ctx


def _render_template(template_str: str, context: dict) -> str:
    template = _jinja_env.from_string(template_str)
    return template.render(**context)
