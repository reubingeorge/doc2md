"""Merge multi-step or multi-page outputs into final document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from doc2md.config.schema import MergeConfig
from doc2md.types import StepResult

if TYPE_CHECKING:
    from doc2md.agents.engine import AgentEngine
    from doc2md.blackboard.board import Blackboard
    from doc2md.types import AgentConfig


def merge_outputs(
    outputs: dict[str, str],
    merge_config: MergeConfig | None = None,
) -> str:
    """Merge step outputs using the configured strategy."""
    if merge_config is None or merge_config.strategy == "concatenate":
        return _concatenate(outputs)
    return _concatenate(outputs)


async def merge_with_agent(
    outputs: dict[str, str],
    merge_config: MergeConfig,
    agent_engine: AgentEngine,
    agent_configs: dict[str, AgentConfig],
    blackboard: Blackboard,
) -> StepResult:
    """Use a VLM agent to intelligently merge outputs."""
    agent_name = merge_config.agent
    if not agent_name or agent_name not in agent_configs:
        # Fall back to concatenation
        return StepResult(
            step_name="merge",
            agent_name="concatenate",
            markdown=_concatenate(outputs),
        )

    combined_input = "\n\n---\n\n".join(
        f"## Section: {name}\n\n{content}" for name, content in outputs.items()
    )

    result = await agent_engine.execute(
        agent_config=agent_configs[agent_name],
        previous_output=combined_input,
        blackboard=blackboard,
        step_name="merge",
    )
    return result


def _concatenate(outputs: dict[str, str]) -> str:
    """Simple ordered concatenation."""
    return "\n\n".join(outputs.values())
