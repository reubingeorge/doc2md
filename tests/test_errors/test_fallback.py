"""Tests for model fallback chain."""

import pytest

from doc2md.errors.exceptions import TerminalError
from doc2md.errors.fallback import FallbackChain


class TestFallbackChain:
    def test_initial_model(self):
        chain = FallbackChain("gpt-4.1-mini", ["gpt-4o-mini"])
        assert chain.current_model == "gpt-4.1-mini"

    def test_next_model(self):
        chain = FallbackChain("gpt-4.1-mini", ["gpt-4o-mini", "gpt-3.5-turbo"])
        next_m = chain.next_model()
        assert next_m == "gpt-4o-mini"
        assert chain.current_model == "gpt-4o-mini"

    def test_exhausted_raises(self):
        chain = FallbackChain("gpt-4.1-mini")
        with pytest.raises(TerminalError, match="exhausted"):
            chain.next_model()

    def test_full_chain_cycle(self):
        chain = FallbackChain("a", ["b", "c"])
        assert chain.current_model == "a"
        chain.next_model()  # -> b
        assert chain.current_model == "b"
        chain.next_model()  # -> c
        assert chain.current_model == "c"
        with pytest.raises(TerminalError):
            chain.next_model()

    def test_exhausted_property(self):
        chain = FallbackChain("a", ["b"])
        assert chain.exhausted is False
        chain.next_model()
        assert chain.exhausted is False
        chain.mark_tried("b")
        assert chain.exhausted is True

    def test_reset(self):
        chain = FallbackChain("a", ["b"])
        chain.next_model()
        chain.reset()
        assert chain.current_model == "a"
        assert chain.exhausted is False

    def test_no_fallbacks(self):
        chain = FallbackChain("primary")
        assert chain.current_model == "primary"
        with pytest.raises(TerminalError):
            chain.next_model()

    def test_mark_tried(self):
        chain = FallbackChain("a", ["b", "c"])
        chain.mark_tried("b")
        # Skips b, goes to c
        next_m = chain.next_model()
        assert next_m == "c"
