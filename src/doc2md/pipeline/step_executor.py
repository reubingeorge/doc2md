"""Execute individual pipeline steps by dispatching to the correct handler."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from doc2md.blackboard.board import Blackboard
from doc2md.config.schema import StepConfig, StepType
from doc2md.pipeline.data_flow import StepInput
from doc2md.types import AgentConfig, StepResult, TokenUsage

if TYPE_CHECKING:
    from doc2md.agents.engine import AgentEngine
    from doc2md.cache.manager import CacheManager

# Max concurrent VLM calls per step (pages processed in parallel)
_MAX_PAGE_CONCURRENCY = 4

# Registry of code step functions
_CODE_STEP_REGISTRY: dict[str, Callable[..., str]] = {}


def register_code_step(name: str) -> Callable:
    """Decorator to register a code step function."""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        _CODE_STEP_REGISTRY[name] = fn
        return fn

    return decorator


def get_code_step(name: str) -> Callable[..., str] | None:
    return _CODE_STEP_REGISTRY.get(name)


async def execute_step(
    step_config: StepConfig,
    step_input: StepInput,
    blackboard: Blackboard,
    agent_engine: AgentEngine,
    agent_configs: dict[str, AgentConfig],
    cache_manager: CacheManager | None = None,
    pipeline_name: str = "",
) -> StepResult:
    """Execute a single pipeline step, dispatching by type."""
    if step_config.type == StepType.AGENT:
        return await _execute_agent_step(
            step_config,
            step_input,
            blackboard,
            agent_engine,
            agent_configs,
            cache_manager=cache_manager,
            pipeline_name=pipeline_name,
        )
    elif step_config.type == StepType.CODE:
        return _execute_code_step(step_config, step_input)
    elif step_config.type == StepType.PARALLEL:
        from doc2md.pipeline.parallel_executor import execute_parallel

        return await execute_parallel(
            step_config,
            step_input,
            blackboard,
            agent_engine,
            agent_configs,
            cache_manager=cache_manager,
            pipeline_name=pipeline_name,
        )
    elif step_config.type == StepType.PAGE_ROUTE:
        from doc2md.pipeline.page_router import execute_page_route

        return await execute_page_route(
            step_config,
            step_input,
            blackboard,
            agent_engine,
            agent_configs,
            cache_manager=cache_manager,
            pipeline_name=pipeline_name,
        )
    else:
        raise ValueError(f"Unknown step type: {step_config.type}")


async def _execute_agent_step(
    step_config: StepConfig,
    step_input: StepInput,
    blackboard: Blackboard,
    agent_engine: AgentEngine,
    agent_configs: dict[str, AgentConfig],
    cache_manager: CacheManager | None = None,
    pipeline_name: str = "",
) -> StepResult:
    """Execute a single agent step, processing pages and merging results."""
    agent_name = step_config.agent
    if not agent_name or agent_name not in agent_configs:
        raise ValueError(f"Agent '{agent_name}' not found for step '{step_config.name}'")

    agent_config = agent_configs[agent_name]
    images = step_input.images

    if not images:
        # Text-only step (e.g. summarize)
        result = await agent_engine.execute(
            agent_config=agent_config,
            previous_output=step_input.previous_output,
            blackboard=blackboard,
            step_name=step_config.name,
            cache_manager=cache_manager,
            pipeline_name=pipeline_name,
        )
        blackboard.write("step_outputs", step_config.name, result.markdown, writer=step_config.name)
        return result

    # Process pages concurrently (bounded by semaphore)
    semaphore = asyncio.Semaphore(_MAX_PAGE_CONCURRENCY)

    async def _process_page(i: int, img: bytes) -> StepResult:
        async with semaphore:
            return await agent_engine.execute(
                agent_config=agent_config,
                image_bytes=img,
                previous_output=step_input.previous_output,
                blackboard=blackboard,
                step_name=step_config.name,
                page_num=i + 1,
                cache_manager=cache_manager,
                pipeline_name=pipeline_name,
            )

    page_results = list(
        await asyncio.gather(*[_process_page(i, img) for i, img in enumerate(images)])
    )

    merged = _merge_page_results(step_config.name, agent_name, page_results)
    blackboard.write("step_outputs", step_config.name, merged.markdown, writer=step_config.name)
    return merged


def _execute_code_step(
    step_config: StepConfig,
    step_input: StepInput,
) -> StepResult:
    """Execute a deterministic code step (no VLM call)."""
    fn_name = step_config.function
    if not fn_name:
        raise ValueError(f"Code step '{step_config.name}' missing 'function' field")

    fn = get_code_step(fn_name)
    if fn is None:
        raise ValueError(f"Code step function '{fn_name}' not registered")

    input_text = step_input.previous_output or ""
    output = fn(input_text, **step_config.params)

    return StepResult(
        step_name=step_config.name,
        agent_name=f"code:{fn_name}",
        markdown=output,
    )


def _merge_page_results(
    step_name: str,
    agent_name: str,
    results: list[StepResult],
) -> StepResult:
    """Merge per-page results into a single step result."""
    page_markdowns = [r.markdown for r in results]

    if len(results) == 1:
        results[0].page_markdowns = page_markdowns
        return results[0]

    combined_md = "\n\n".join(page_markdowns)
    total_usage = TokenUsage(
        prompt_tokens=sum(r.token_usage.prompt_tokens for r in results),
        completion_tokens=sum(r.token_usage.completion_tokens for r in results),
        total_tokens=sum(r.token_usage.total_tokens for r in results),
    )

    return StepResult(
        step_name=step_name,
        agent_name=agent_name,
        markdown=combined_md,
        page_markdowns=page_markdowns,
        token_usage=total_usage,
        model_used=results[0].model_used if results else "",
    )
