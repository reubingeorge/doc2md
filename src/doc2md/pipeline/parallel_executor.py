"""Execute parallel pipeline steps concurrently with blackboard merge."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from doc2md.blackboard.board import Blackboard
from doc2md.blackboard.merge import merge_parallel
from doc2md.config.schema import StepConfig
from doc2md.pipeline.data_flow import StepInput
from doc2md.types import AgentConfig, StepResult, TokenUsage

if TYPE_CHECKING:
    from doc2md.agents.engine import AgentEngine
    from doc2md.cache.manager import CacheManager


async def execute_parallel(
    step_config: StepConfig,
    step_input: StepInput,
    blackboard: Blackboard,
    agent_engine: AgentEngine,
    agent_configs: dict[str, AgentConfig],
    cache_manager: CacheManager | None = None,
    pipeline_name: str = "",
) -> StepResult:
    """Execute sub-steps concurrently, then merge blackboards and outputs."""
    from doc2md.pipeline.step_executor import execute_step

    sub_steps = step_config.steps or []
    if not sub_steps:
        raise ValueError(f"Parallel step '{step_config.name}' has no sub-steps")

    # Create blackboard copies for each branch
    branches: list[tuple[StepConfig, Blackboard]] = [
        (sub, blackboard.copy()) for sub in sub_steps
    ]

    # Execute all branches concurrently
    tasks = [
        execute_step(
            sub, step_input, bb_copy, agent_engine, agent_configs,
            cache_manager=cache_manager, pipeline_name=pipeline_name,
        )
        for sub, bb_copy in branches
    ]
    sub_results: list[StepResult] = await asyncio.gather(*tasks)

    # Merge blackboard writes from all branches
    branch_boards = [bb_copy for _, bb_copy in branches]
    merge_parallel(blackboard, branch_boards)

    # Merge outputs
    merged = _merge_parallel_results(step_config.name, sub_results)
    blackboard.write("step_outputs", step_config.name, merged.markdown, writer=step_config.name)
    return merged


def _merge_parallel_results(
    step_name: str,
    results: list[StepResult],
) -> StepResult:
    """Merge results from parallel sub-steps into one."""
    combined_md = "\n\n".join(r.markdown for r in results)
    total_usage = TokenUsage(
        prompt_tokens=sum(r.token_usage.prompt_tokens for r in results),
        completion_tokens=sum(r.token_usage.completion_tokens for r in results),
        total_tokens=sum(r.token_usage.total_tokens for r in results),
    )

    return StepResult(
        step_name=step_name,
        agent_name="parallel",
        markdown=combined_md,
        token_usage=total_usage,
        model_used=results[0].model_used if results else "",
    )
