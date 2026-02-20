"""Package-level default configuration values."""

from __future__ import annotations

from typing import Any

# Default model settings
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_CLASSIFIER_MODEL = "gpt-4.1-nano"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0

# Default cache settings
DEFAULT_CACHE_MEMORY_MB = 500.0
DEFAULT_CACHE_DISK_MB = 5000.0
DEFAULT_CACHE_DISABLED = False

# Default concurrency settings
DEFAULT_MAX_WORKERS = 5
DEFAULT_RPM_LIMIT = 3500
DEFAULT_TPM_LIMIT = 100_000

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_STRATEGY = "exponential"

# Default confidence thresholds
DEFAULT_CONFIDENCE_HIGH = 0.8
DEFAULT_CONFIDENCE_MEDIUM = 0.6
DEFAULT_CONFIDENCE_LOW = 0.3

# Log level
DEFAULT_LOG_LEVEL = "WARNING"


def get_defaults() -> dict[str, Any]:
    """Return all defaults as a flat dictionary for merging."""
    return {
        "model": DEFAULT_MODEL,
        "classifier_model": DEFAULT_CLASSIFIER_MODEL,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "cache_memory_mb": DEFAULT_CACHE_MEMORY_MB,
        "cache_disk_mb": DEFAULT_CACHE_DISK_MB,
        "cache_disabled": DEFAULT_CACHE_DISABLED,
        "max_workers": DEFAULT_MAX_WORKERS,
        "rpm_limit": DEFAULT_RPM_LIMIT,
        "tpm_limit": DEFAULT_TPM_LIMIT,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_strategy": DEFAULT_RETRY_STRATEGY,
        "log_level": DEFAULT_LOG_LEVEL,
    }
