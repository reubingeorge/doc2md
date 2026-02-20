"""Custom exception hierarchy for doc2md."""

from __future__ import annotations

from typing import Any


class Doc2MdError(Exception):
    """Base exception for all doc2md errors."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        super().__init__(message)
        self.message = message


class TransientError(Doc2MdError):
    """Transient error — safe to retry with backoff.

    Examples: 429 rate limit, 500/502/503 server error, timeout, connection error.
    """

    def __init__(
        self,
        message: str = "",
        error_type: str = "server_error",
        http_status: int | None = None,
        retry_after: float | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.http_status = http_status
        self.retry_after = retry_after
        self.original = original


class RecoverableError(Doc2MdError):
    """Recoverable error — retry with modification.

    Examples: validation failure, low confidence, content filter, token limit.
    """

    def __init__(
        self,
        message: str = "",
        error_type: str = "validation_failure",
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.suggestion = suggestion


class TerminalError(Doc2MdError):
    """Terminal error — fail fast (try fallback model first if applicable).

    Examples: 401 auth failure, 404 model not found, bad input, invalid config.
    """

    def __init__(
        self,
        message: str = "",
        error_type: str = "auth_failure",
        http_status: int | None = None,
        recoverable_with_fallback: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.http_status = http_status
        self.recoverable_with_fallback = recoverable_with_fallback


class PageLevelError(Doc2MdError):
    """Error isolated to a single page — other pages continue."""

    def __init__(
        self,
        message: str = "",
        page_num: int = 0,
        inner: Doc2MdError | None = None,
    ) -> None:
        super().__init__(message)
        self.page_num = page_num
        self.inner = inner
