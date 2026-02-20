"""Page routing — classify pages and route to different agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from doc2md.blackboard.board import Blackboard
from doc2md.config.schema import PageSelector, RouterConfig, RouterStrategy, StepConfig
from doc2md.pipeline.data_flow import StepInput
from doc2md.types import AgentConfig, StepResult, TokenUsage

if TYPE_CHECKING:
    from doc2md.agents.engine import AgentEngine
    from doc2md.cache.manager import CacheManager


async def execute_page_route(
    step_config: StepConfig,
    step_input: StepInput,
    blackboard: Blackboard,
    agent_engine: AgentEngine,
    agent_configs: dict[str, AgentConfig],
    cache_manager: CacheManager | None = None,
    pipeline_name: str = "",
) -> StepResult:
    """Route individual pages to different agents based on classification."""
    router = step_config.router
    if not router:
        raise ValueError(f"Page route step '{step_config.name}' missing router config")

    images = step_input.images
    if not images:
        raise ValueError(f"Page route step '{step_config.name}' received no images")

    total_pages = len(images)

    # Classify pages → agent assignments
    assignments = classify_pages(total_pages, router)

    # Group consecutive pages if cross-page aware
    if step_config.cross_page_aware:
        assignments = _apply_cross_page_grouping(assignments, blackboard)

    # Execute each page with its assigned agent
    page_results: list[StepResult] = []
    for page_num, agent_name in sorted(assignments.items()):
        if agent_name not in agent_configs:
            agent_name = router.default_agent
        if agent_name not in agent_configs:
            raise ValueError(f"Agent '{agent_name}' not found for page {page_num}")

        img_idx = page_num - 1
        if img_idx >= len(images):
            continue

        result = await agent_engine.execute(
            agent_config=agent_configs[agent_name],
            image_bytes=images[img_idx],
            previous_output=step_input.previous_output,
            blackboard=blackboard,
            step_name=f"{step_config.name}_page_{page_num}",
            page_num=page_num,
            cache_manager=cache_manager,
            pipeline_name=pipeline_name,
        )
        page_results.append(result)

    merged = _merge_routed_results(step_config.name, page_results, assignments)
    blackboard.write("step_outputs", step_config.name, merged.markdown, writer=step_config.name)
    return merged


def classify_pages(
    total_pages: int,
    router: RouterConfig,
) -> dict[int, str]:
    """Assign each page to an agent based on router strategy.

    For now, supports rules-based classification.
    VLM classification will be added in Phase 4 (classifier).
    """
    assignments: dict[int, str] = {}

    if router.strategy in (RouterStrategy.RULES, RouterStrategy.HYBRID):
        for rule in router.rules:
            selector = PageSelector(raw=rule.pages)
            pages = selector.resolve(total_pages)
            for page in pages:
                assignments[page] = rule.agent

    # Fill unassigned pages with default agent
    for page in range(1, total_pages + 1):
        if page not in assignments:
            assignments[page] = router.default_agent

    return assignments


def _apply_cross_page_grouping(
    assignments: dict[int, str],
    blackboard: Blackboard,
) -> dict[int, str]:
    """Check blackboard for continuation markers and group pages accordingly."""
    for page_num in sorted(assignments.keys()):
        obs = blackboard.page_observations.get(page_num)
        if obs and obs.continues_on_next_page:
            next_page = page_num + 1
            if next_page in assignments:
                # Route the next page to the same agent
                assignments[next_page] = assignments[page_num]
    return assignments


def _merge_routed_results(
    step_name: str,
    results: list[StepResult],
    assignments: dict[int, str],
) -> StepResult:
    """Merge results from page-routed execution."""
    if not results:
        return StepResult(step_name=step_name, agent_name="page_route", markdown="")

    combined_md = "\n\n".join(r.markdown for r in results)
    total_usage = TokenUsage(
        prompt_tokens=sum(r.token_usage.prompt_tokens for r in results),
        completion_tokens=sum(r.token_usage.completion_tokens for r in results),
        total_tokens=sum(r.token_usage.total_tokens for r in results),
    )

    return StepResult(
        step_name=step_name,
        agent_name="page_route",
        markdown=combined_md,
        token_usage=total_usage,
        model_used=results[0].model_used if results else "",
    )
