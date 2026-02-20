"""Configuration hierarchy — merges sources in priority order.

Precedence (later overrides earlier):
  1. Package defaults
  2. Global config   (~/.doc2md/config.yaml)
  3. Project config   (./doc2md.yaml)
  4. Environment variables (OPENAI_API_KEY, DOC2MD_*)
  5. Runtime arguments
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from doc2md.config.defaults import get_defaults

logger = logging.getLogger(__name__)

_GLOBAL_CONFIG_PATH = Path.home() / ".doc2md" / "config.yaml"
_PROJECT_CONFIG_NAME = "doc2md.yaml"

# Map of environment variables to config keys
_ENV_MAP: dict[str, str] = {
    "OPENAI_API_KEY": "api_key",
    "OPENAI_BASE_URL": "base_url",
    "DOC2MD_MODEL": "model",
    "DOC2MD_CACHE_DISABLED": "cache_disabled",
    "DOC2MD_CACHE_DB_PATH": "cache_db_path",
    "DOC2MD_CACHE_MEMORY_MB": "cache_memory_mb",
    "DOC2MD_CACHE_DISK_MB": "cache_disk_mb",
    "DOC2MD_CONFIG_DIR": "config_dir",
    "DOC2MD_PIPELINE_DIR": "pipeline_dir",
    "DOC2MD_LOG_LEVEL": "log_level",
    "DOC2MD_MAX_WORKERS": "max_workers",
    "DOC2MD_RPM_LIMIT": "rpm_limit",
    "DOC2MD_TPM_LIMIT": "tpm_limit",
}

# Keys that should be parsed as specific types
_TYPE_MAP: dict[str, type] = {
    "cache_memory_mb": float,
    "cache_disk_mb": float,
    "max_workers": int,
    "rpm_limit": int,
    "tpm_limit": int,
    "max_tokens": int,
    "temperature": float,
    "max_retries": int,
}

# Boolean env var values
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def load_config_hierarchy(**runtime_overrides: Any) -> dict[str, Any]:
    """Load and merge configuration from all sources.

    Returns a merged dict with the final resolved values.
    """
    config = get_defaults()

    # Layer 2: Global config
    global_cfg = _load_yaml_config(_GLOBAL_CONFIG_PATH)
    if global_cfg:
        config.update(global_cfg)

    # Layer 3: Project config (search from cwd upward)
    project_path = _find_project_config()
    if project_path:
        project_cfg = _load_yaml_config(project_path)
        if project_cfg:
            config.update(project_cfg)

    # Layer 4: Environment variables
    env_cfg = _load_env_vars()
    config.update(env_cfg)

    # Layer 5: Runtime arguments (highest priority)
    # Filter out None values — only override when explicitly set
    for key, value in runtime_overrides.items():
        if value is not None:
            config[key] = value

    return config


def _load_yaml_config(path: Path) -> dict[str, Any] | None:
    """Load a YAML config file if it exists."""
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        logger.warning("Config file %s is not a mapping, ignoring", path)
    except Exception as e:
        logger.warning("Failed to load config %s: %s", path, e)
    return None


def _find_project_config() -> Path | None:
    """Search for doc2md.yaml from cwd upward."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / _PROJECT_CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def _load_env_vars() -> dict[str, Any]:
    """Read DOC2MD_* and OPENAI_API_KEY environment variables."""
    result: dict[str, Any] = {}
    for env_key, config_key in _ENV_MAP.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        result[config_key] = _coerce_env_value(config_key, value)
    return result


def _coerce_env_value(key: str, value: str) -> Any:
    """Coerce an environment variable string to the appropriate type."""
    if key.endswith("_disabled") or key.startswith("no_"):
        return value.lower() in _TRUTHY

    target_type = _TYPE_MAP.get(key)
    if target_type:
        try:
            return target_type(value)
        except (ValueError, TypeError):
            logger.warning(
                "Cannot convert env var for '%s' to %s: %s", key, target_type.__name__, value
            )
            return value

    return value
