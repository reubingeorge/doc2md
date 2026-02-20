"""Tests for custom exception hierarchy."""

import pytest

from doc2md.errors.exceptions import (
    Doc2MdError,
    PageLevelError,
    RecoverableError,
    TerminalError,
    TransientError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(TransientError, Doc2MdError)
        assert issubclass(RecoverableError, Doc2MdError)
        assert issubclass(TerminalError, Doc2MdError)
        assert issubclass(PageLevelError, Doc2MdError)

    def test_all_inherit_from_exception(self):
        assert issubclass(Doc2MdError, Exception)


class TestTransientError:
    def test_attributes(self):
        err = TransientError(
            "Rate limited",
            error_type="rate_limit",
            http_status=429,
            retry_after=5.0,
        )
        assert err.error_type == "rate_limit"
        assert err.http_status == 429
        assert err.retry_after == 5.0
        assert "Rate limited" in str(err)

    def test_defaults(self):
        err = TransientError("test")
        assert err.error_type == "server_error"
        assert err.http_status is None
        assert err.retry_after is None


class TestRecoverableError:
    def test_attributes(self):
        err = RecoverableError(
            "Low quality",
            error_type="low_confidence",
            suggestion="upscale image",
        )
        assert err.error_type == "low_confidence"
        assert err.suggestion == "upscale image"


class TestTerminalError:
    def test_attributes(self):
        err = TerminalError(
            "Not found",
            error_type="model_not_found",
            http_status=404,
            recoverable_with_fallback=True,
        )
        assert err.error_type == "model_not_found"
        assert err.http_status == 404
        assert err.recoverable_with_fallback is True

    def test_defaults_not_recoverable(self):
        err = TerminalError("auth fail")
        assert err.recoverable_with_fallback is False


class TestPageLevelError:
    def test_attributes(self):
        inner = TransientError("timeout")
        err = PageLevelError("Page 3 failed", page_num=3, inner=inner)
        assert err.page_num == 3
        assert err.inner is inner

    def test_catchable_as_base(self):
        with pytest.raises(Doc2MdError):
            raise PageLevelError("fail", page_num=1)
