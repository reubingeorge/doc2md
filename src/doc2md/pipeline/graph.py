"""Pipeline step graph — DAG parsing, topological sort, cycle detection."""

from __future__ import annotations

from doc2md.config.schema import PipelineConfig, StepConfig


class CycleError(Exception):
    """Raised when the pipeline step graph contains a cycle."""


class StepGraph:
    """Directed acyclic graph of pipeline steps."""

    def __init__(self, steps: list[StepConfig]) -> None:
        self._steps = {s.name: s for s in steps}
        self._edges: dict[str, list[str]] = {s.name: [] for s in steps}
        self._build_edges(steps)

    @property
    def nodes(self) -> list[str]:
        return list(self._steps.keys())

    @property
    def edges(self) -> dict[str, list[str]]:
        """Returns {step_name: [steps_it_depends_on]}."""
        return dict(self._edges)

    def get_step(self, name: str) -> StepConfig:
        return self._steps[name]

    def topological_sort(self) -> list[str]:
        """Return steps in execution order. Raises CycleError on cycles."""
        visited: set[str] = set()
        in_stack: set[str] = set()
        order: list[str] = []

        for node in self._steps:
            if node not in visited:
                self._dfs(node, visited, in_stack, order)

        return order

    def dependencies_of(self, step_name: str) -> list[str]:
        """Return direct dependencies of a step."""
        return list(self._edges.get(step_name, []))

    def _build_edges(self, steps: list[StepConfig]) -> None:
        """Build dependency edges. Steps without depends_on implicitly depend on previous."""
        for i, step in enumerate(steps):
            if step.depends_on is not None:
                # Explicit dependencies
                for dep in step.depends_on:
                    if dep not in self._steps:
                        raise ValueError(f"Step '{step.name}' depends on unknown step '{dep}'")
                    self._edges[step.name].append(dep)
            elif i > 0:
                # Implicit: depends on previous step in list
                self._edges[step.name].append(steps[i - 1].name)

    def _dfs(
        self,
        node: str,
        visited: set[str],
        in_stack: set[str],
        order: list[str],
    ) -> None:
        visited.add(node)
        in_stack.add(node)

        for dep in self._edges[node]:
            if dep in in_stack:
                raise CycleError(f"Cycle detected involving step '{dep}'")
            if dep not in visited:
                self._dfs(dep, visited, in_stack, order)

        in_stack.remove(node)
        order.append(node)


def parse_pipeline(config: PipelineConfig) -> StepGraph:
    """Parse a pipeline config into a step graph."""
    if not config.steps:
        raise ValueError("Pipeline must have at least one step")
    return StepGraph(config.steps)
