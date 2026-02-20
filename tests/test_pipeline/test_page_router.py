"""Tests for page routing: rules, cross-page grouping."""

from unittest.mock import AsyncMock

import pytest

from doc2md.blackboard.board import Blackboard
from doc2md.config.schema import (
    RouterConfig,
    RouterRule,
    RouterStrategy,
    StepConfig,
    StepType,
)
from doc2md.pipeline.data_flow import StepInput
from doc2md.pipeline.page_router import classify_pages, execute_page_route
from doc2md.types import (
    AgentConfig,
    PromptConfig,
    StepResult,
    TokenUsage,
)


def _make_agent(name: str) -> AgentConfig:
    return AgentConfig(name=name, prompt=PromptConfig(system="S", user="U"))


class TestClassifyPages:
    def test_rules_only(self):
        router = RouterConfig(
            strategy=RouterStrategy.RULES,
            rules=[
                RouterRule(pages=[1, 2], agent="cover_page"),
                RouterRule(pages=[-1], agent="signature_page"),
            ],
            default_agent="generic",
        )
        assignments = classify_pages(5, router)
        assert assignments[1] == "cover_page"
        assert assignments[2] == "cover_page"
        assert assignments[3] == "generic"
        assert assignments[5] == "signature_page"

    def test_all_default(self):
        router = RouterConfig(strategy=RouterStrategy.RULES, default_agent="text_extract")
        assignments = classify_pages(3, router)
        assert all(a == "text_extract" for a in assignments.values())
        assert len(assignments) == 3


class TestExecutePageRoute:
    async def test_routes_pages_to_agents(self):
        call_log: list[str] = []

        async def _mock_execute(agent_config, **kwargs):
            call_log.append(agent_config.name)
            return StepResult(
                step_name=kwargs.get("step_name", ""),
                agent_name=agent_config.name,
                markdown=f"Page by {agent_config.name}",
                model_used="gpt-4.1-mini",
            )

        engine = AsyncMock()
        engine.execute = _mock_execute

        step = StepConfig(
            name="route",
            type=StepType.PAGE_ROUTE,
            router=RouterConfig(
                strategy=RouterStrategy.RULES,
                rules=[RouterRule(pages=[1], agent="cover")],
                default_agent="text",
            ),
        )

        configs = {
            "cover": _make_agent("cover"),
            "text": _make_agent("text"),
        }

        step_input = StepInput(images=[b"p1", b"p2", b"p3"])
        bb = Blackboard()

        result = await execute_page_route(step, step_input, bb, engine, configs)

        assert call_log[0] == "cover"  # page 1
        assert call_log[1] == "text"   # page 2
        assert call_log[2] == "text"   # page 3
        assert "Page by cover" in result.markdown

    async def test_cross_page_grouping(self):
        call_log: list[str] = []

        async def _mock_execute(agent_config, **kwargs):
            call_log.append(agent_config.name)
            return StepResult(
                step_name=kwargs.get("step_name", ""),
                agent_name=agent_config.name,
                markdown="Content",
                model_used="gpt-4.1-mini",
            )

        engine = AsyncMock()
        engine.execute = _mock_execute

        step = StepConfig(
            name="route",
            type=StepType.PAGE_ROUTE,
            cross_page_aware=True,
            router=RouterConfig(
                strategy=RouterStrategy.RULES,
                rules=[RouterRule(pages=[1], agent="table_extract")],
                default_agent="text",
            ),
        )

        configs = {
            "table_extract": _make_agent("table_extract"),
            "text": _make_agent("text"),
        }

        bb = Blackboard()
        # Mark page 1 as continuing
        bb.write("page_observations", "1.continues_on_next_page", True, writer="test")

        step_input = StepInput(images=[b"p1", b"p2"])
        await execute_page_route(step, step_input, bb, engine, configs)

        # Page 2 should be routed to table_extract (same as page 1)
        assert call_log[1] == "table_extract"

    async def test_no_router_raises(self):
        step = StepConfig(name="bad", type=StepType.PAGE_ROUTE)
        with pytest.raises(ValueError, match="missing router"):
            await execute_page_route(step, StepInput(images=[b"x"]), Blackboard(), AsyncMock(), {})
