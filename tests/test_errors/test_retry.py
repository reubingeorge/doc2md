"""Tests for retry logic."""

import httpx
import openai
import pytest

from doc2md.errors.exceptions import TerminalError, TransientError
from doc2md.errors.retry import classify_openai_error, compute_wait
from doc2md.types import RetryStrategy


def _mock_response(status_code: int) -> httpx.Response:
    """Create a minimal httpx.Response for constructing OpenAI exceptions."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


class TestClassifyOpenAIError:
    def test_rate_limit(self):
        err = classify_openai_error(
            openai.RateLimitError(
                message="rate limit", response=_mock_response(429), body=None,
            )
        )
        assert isinstance(err, TransientError)
        assert err.error_type == "rate_limit"

    def test_internal_server(self):
        err = classify_openai_error(
            openai.InternalServerError(
                message="server err", response=_mock_response(500), body=None,
            )
        )
        assert isinstance(err, TransientError)
        assert err.error_type == "server_error"

    def test_connection_error(self):
        err = classify_openai_error(
            openai.APIConnectionError(request=None)
        )
        assert isinstance(err, TransientError)
        assert err.error_type == "timeout"

    def test_auth_error(self):
        err = classify_openai_error(
            openai.AuthenticationError(
                message="bad key", response=_mock_response(401), body=None,
            )
        )
        assert isinstance(err, TerminalError)
        assert err.error_type == "auth_failure"
        assert err.recoverable_with_fallback is False

    def test_not_found(self):
        err = classify_openai_error(
            openai.NotFoundError(
                message="model not found", response=_mock_response(404), body=None,
            )
        )
        assert isinstance(err, TerminalError)
        assert err.error_type == "model_not_found"
        assert err.recoverable_with_fallback is True

    def test_bad_request(self):
        err = classify_openai_error(
            openai.BadRequestError(
                message="bad input", response=_mock_response(400), body=None,
            )
        )
        assert isinstance(err, TerminalError)
        assert err.error_type == "bad_input"


class TestComputeWait:
    def test_exponential_backoff(self):
        w0 = compute_wait(0, RetryStrategy.EXPONENTIAL, initial_wait=1.0, jitter=False)
        w1 = compute_wait(1, RetryStrategy.EXPONENTIAL, initial_wait=1.0, jitter=False)
        w2 = compute_wait(2, RetryStrategy.EXPONENTIAL, initial_wait=1.0, jitter=False)
        assert w0 == 1.0
        assert w1 == 2.0
        assert w2 == 4.0

    def test_linear_backoff(self):
        w0 = compute_wait(0, RetryStrategy.LINEAR, initial_wait=2.0, jitter=False)
        w1 = compute_wait(1, RetryStrategy.LINEAR, initial_wait=2.0, jitter=False)
        assert w0 == 2.0
        assert w1 == 4.0

    def test_fixed_backoff(self):
        w0 = compute_wait(0, RetryStrategy.FIXED, initial_wait=5.0, jitter=False)
        w1 = compute_wait(1, RetryStrategy.FIXED, initial_wait=5.0, jitter=False)
        assert w0 == 5.0
        assert w1 == 5.0

    def test_max_wait_capped(self):
        w = compute_wait(10, RetryStrategy.EXPONENTIAL, initial_wait=1.0, jitter=False)
        assert w == 60.0  # _MAX_WAIT

    def test_jitter_adds_randomness(self):
        # With jitter, result should be >= base wait
        w = compute_wait(0, RetryStrategy.EXPONENTIAL, initial_wait=1.0, jitter=True)
        assert w >= 1.0
