"""Retry engine — configurable retry logic wrapping VLM calls."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import openai

from doc2md.errors.exceptions import (
    Doc2MdError,
    TerminalError,
    TransientError,
)
from doc2md.errors.fallback import FallbackChain
from doc2md.types import RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_WAIT = 60.0  # seconds


def classify_openai_error(exc: Exception) -> Doc2MdError:
    """Convert an openai exception to our exception hierarchy."""
    if isinstance(exc, openai.RateLimitError):
        retry_after = None
        if hasattr(exc, "response") and exc.response:
            retry_after_str = exc.response.headers.get("retry-after")
            if retry_after_str:
                with contextlib.suppress(ValueError):
                    retry_after = float(retry_after_str)
        return TransientError(
            str(exc),
            error_type="rate_limit",
            http_status=429,
            retry_after=retry_after,
            original=exc,
        )
    if isinstance(exc, (openai.InternalServerError,)):
        status = getattr(exc, "status_code", 500)
        return TransientError(
            str(exc),
            error_type="server_error",
            http_status=status,
            original=exc,
        )
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return TransientError(
            str(exc),
            error_type="timeout",
            original=exc,
        )
    if isinstance(exc, openai.AuthenticationError):
        return TerminalError(
            str(exc),
            error_type="auth_failure",
            http_status=401,
            recoverable_with_fallback=False,
        )
    if isinstance(exc, openai.NotFoundError):
        return TerminalError(
            str(exc),
            error_type="model_not_found",
            http_status=404,
            recoverable_with_fallback=True,
        )
    if isinstance(exc, openai.BadRequestError):
        return TerminalError(
            str(exc),
            error_type="bad_input",
            http_status=400,
            recoverable_with_fallback=False,
        )
    return TerminalError(str(exc), error_type="unknown")


def compute_wait(
    attempt: int,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    initial_wait: float = 1.0,
    jitter: bool = True,
) -> float:
    """Compute wait time for a retry attempt."""
    if strategy == RetryStrategy.EXPONENTIAL:
        wait = initial_wait * (2**attempt)
    elif strategy == RetryStrategy.LINEAR:
        wait = initial_wait * (attempt + 1)
    else:  # FIXED
        wait = initial_wait

    if jitter:
        wait += random.uniform(0, wait * 0.25)

    return min(wait, _MAX_WAIT)


async def retry_with_fallback(
    fn: Callable[..., Awaitable[T]],
    retry_config: RetryConfig,
    fallback_chain: FallbackChain | None = None,
    **kwargs: Any,
) -> T:
    """Execute an async function with retry + model fallback.

    On transient errors: retry with backoff up to max_attempts.
    On terminal errors with fallback: try next model.
    On terminal errors without fallback: raise immediately.
    """
    last_error: Exception | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return await fn(**kwargs)
        except Exception as exc:
            classified = classify_openai_error(exc) if isinstance(exc, openai.OpenAIError) else exc

            if isinstance(classified, TransientError):
                wait = classified.retry_after or compute_wait(
                    attempt,
                    retry_config.strategy,
                )
                logger.warning(
                    "Transient error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    retry_config.max_attempts,
                    classified.error_type,
                    wait,
                )
                last_error = classified
                await asyncio.sleep(wait)
                continue

            if (
                isinstance(classified, TerminalError)
                and classified.recoverable_with_fallback
                and fallback_chain
                and not fallback_chain.exhausted
            ):
                try:
                    next_model = fallback_chain.next_model()
                    kwargs["model"] = next_model
                    logger.info("Trying fallback model: %s", next_model)
                    continue
                except TerminalError:
                    raise classified from exc

            # Terminal or unrecoverable — raise
            raise classified from exc if isinstance(classified, Doc2MdError) else exc

    # Exhausted attempts
    if last_error:
        raise last_error
    raise TransientError("Max retry attempts exhausted")
