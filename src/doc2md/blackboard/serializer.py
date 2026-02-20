"""Serialize blackboard regions for prompt injection."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from doc2md.blackboard.board import Blackboard

# Max characters for a single region in the prompt context
_MAX_REGION_CHARS = 8000


def serialize_for_prompt(
    blackboard: Blackboard,
    subscriptions: list[str],
) -> dict[str, Any]:
    """Serialize only the subscribed regions/keys into a prompt-safe dict.

    Subscriptions can be:
      - "document_metadata"                → entire region
      - "document_metadata.language"       → single field
      - "page_observations.*.quality_score" → specific field across pages
    """
    result: dict[str, Any] = {}

    for sub in subscriptions:
        parts = sub.split(".")
        region = parts[0]
        _add_region_data(blackboard, region, parts[1:], result)

    return result


def _add_region_data(
    blackboard: Blackboard,
    region: str,
    subpath: list[str],
    result: dict[str, Any],
) -> None:
    """Add data from a single subscription path to the result dict."""
    store = getattr(blackboard, region, None)
    if store is None:
        return

    if region not in result:
        result[region] = {}

    if not subpath:
        # Whole region requested
        result[region] = _serialize_value(store)
        _truncate_if_needed(result, region)
        return

    if region == "document_metadata":
        field = subpath[0]
        val = getattr(store, field, None)
        if val is not None:
            result[region][field] = val

    elif region == "page_observations" and subpath[0] == "*":
        # Wildcard: e.g. page_observations.*.quality_score
        field = subpath[1] if len(subpath) > 1 else None
        for page_num, obs in store.items():
            if field:
                val = getattr(obs, field, None)
                if val is not None:
                    if page_num not in result[region]:
                        result[region][page_num] = {}
                    result[region][page_num][field] = val
            else:
                result[region][page_num] = _serialize_value(obs)

    elif region in ("step_outputs", "agent_notes", "confidence_signals"):
        key = subpath[0]
        if key in store:
            result[region][key] = copy.deepcopy(store[key])


def _serialize_value(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    return copy.deepcopy(obj)


def _truncate_if_needed(result: dict, region: str) -> None:
    serialized = str(result[region])
    if len(serialized) > _MAX_REGION_CHARS:
        result[region] = {"_truncated": True, "_preview": serialized[:_MAX_REGION_CHARS]}
