"""Tests for pipeline data flow resolution."""

from doc2md.pipeline.data_flow import resolve_step_input
from doc2md.types import InputMode, StepResult


def _make_result(name: str, md: str) -> StepResult:
    return StepResult(step_name=name, agent_name="a", markdown=md)


class TestResolveStepInput:
    def test_image_mode(self):
        inp = resolve_step_input(InputMode.IMAGE, [b"img1", b"img2"], None, {})
        assert inp.images == [b"img1", b"img2"]
        assert inp.previous_output is None

    def test_previous_output_mode(self):
        results = {"extract": _make_result("extract", "# Title")}
        inp = resolve_step_input(InputMode.PREVIOUS_OUTPUT, [], ["extract"], results)
        assert inp.images == []
        assert inp.previous_output == "# Title"

    def test_image_and_previous_mode(self):
        results = {"extract": _make_result("extract", "# Title")}
        inp = resolve_step_input(InputMode.IMAGE_AND_PREVIOUS, [b"img"], ["extract"], results)
        assert inp.images == [b"img"]
        assert inp.previous_output == "# Title"

    def test_previous_outputs_mode(self):
        results = {
            "text": _make_result("text", "Text content"),
            "tables": _make_result("tables", "| A | B |"),
        }
        inp = resolve_step_input(InputMode.PREVIOUS_OUTPUTS, [], ["text", "tables"], results)
        assert inp.previous_outputs == {
            "text": "Text content",
            "tables": "| A | B |",
        }

    def test_previous_output_only_mode(self):
        results = {"extract": _make_result("extract", "Content")}
        inp = resolve_step_input(InputMode.PREVIOUS_OUTPUT_ONLY, [b"img"], ["extract"], results)
        assert inp.images == []  # No images for output_only
        assert inp.previous_output == "Content"

    def test_no_dependencies(self):
        inp = resolve_step_input(InputMode.IMAGE, [b"img"], None, {})
        assert inp.previous_output is None
        assert inp.previous_outputs == {}

    def test_multiple_deps_uses_last(self):
        results = {
            "a": _make_result("a", "First"),
            "b": _make_result("b", "Second"),
        }
        inp = resolve_step_input(InputMode.PREVIOUS_OUTPUT, [], ["a", "b"], results)
        assert inp.previous_output == "Second"
