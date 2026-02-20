"""Error handling — exceptions, retry logic, and model fallback."""

from doc2md.errors.exceptions import (
    Doc2MdError,
    PageLevelError,
    RecoverableError,
    TerminalError,
    TransientError,
)

__all__ = [
    "Doc2MdError",
    "TransientError",
    "RecoverableError",
    "TerminalError",
    "PageLevelError",
]
