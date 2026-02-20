"""Cache key generation — content-addressed, blackboard-aware."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def generate_cache_key(
    image_hash: str,
    pipeline_name: str,
    step_name: str,
    agent_name: str,
    agent_version: str,
    model_id: str,
    prompt_hash: str,
    blackboard_snapshot: dict[str, Any] | None = None,
) -> str:
    """Generate a SHA256 cache key from all deterministic inputs.

    The blackboard_snapshot should contain ONLY the regions this agent
    subscribed to (from agent's blackboard.reads). Unsubscribed regions
    do not affect the cache key.
    """
    components = [
        image_hash,
        pipeline_name,
        step_name,
        agent_name,
        agent_version,
        model_id,
        prompt_hash,
        _hash_dict(blackboard_snapshot) if blackboard_snapshot else "",
    ]
    combined = "|".join(components)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def hash_image(image_bytes: bytes) -> str:
    """Hash image bytes for cache key use."""
    return hashlib.sha256(image_bytes).hexdigest()


def hash_prompt(system_prompt: str, user_prompt: str) -> str:
    """Hash prompt content for cache key use."""
    combined = system_prompt + "||" + user_prompt
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _hash_dict(d: dict[str, Any]) -> str:
    """Deterministic hash of a dict via sorted JSON."""
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
