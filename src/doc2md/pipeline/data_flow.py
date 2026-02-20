"""Resolve step inputs based on input mode and pipeline data flow."""

from __future__ import annotations

from dataclasses import dataclass, field

from doc2md.types import InputMode, StepResult


@dataclass
class StepInput:
    """Resolved input for a pipeline step."""

    images: list[bytes] = field(default_factory=list)
    previous_output: str | None = None
    previous_outputs: dict[str, str] = field(default_factory=dict)


def resolve_step_input(
    input_mode: InputMode,
    images: list[bytes],
    depends_on: list[str] | None,
    step_results: dict[str, StepResult],
) -> StepInput:
    """Resolve what a step receives based on its input mode and dependencies."""
    step_input = StepInput()

    if input_mode in (InputMode.IMAGE, InputMode.IMAGE_AND_PREVIOUS):
        step_input.images = images

    if depends_on:
        dep_outputs = {
            name: step_results[name].markdown for name in depends_on if name in step_results
        }

        if input_mode == InputMode.PREVIOUS_OUTPUTS:
            step_input.previous_outputs = dep_outputs
        elif (
            input_mode
            in (
                InputMode.PREVIOUS_OUTPUT,
                InputMode.IMAGE_AND_PREVIOUS,
                InputMode.PREVIOUS_OUTPUT_ONLY,
            )
            and dep_outputs
        ):
            last_dep = depends_on[-1]
            step_input.previous_output = dep_outputs.get(last_dep)

    return step_input
