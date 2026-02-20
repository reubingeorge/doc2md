"""Blackboard — typed, region-based shared memory for pipeline execution."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

from doc2md.blackboard.events import BlackboardEvent, EventLog, EventType
from doc2md.blackboard.regions import (
    VALID_REGIONS,
    DocumentMetadata,
    PageObservation,
)

logger = logging.getLogger(__name__)


class BlackboardView:
    """Read-only frozen view of subscribed blackboard regions."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError as err:
            raise AttributeError(f"BlackboardView has no region '{name}'") from err

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


class Blackboard:
    """Typed, region-based shared memory for pipeline execution."""

    def __init__(self) -> None:
        self.document_metadata = DocumentMetadata()
        self.page_observations: dict[int, PageObservation] = {}
        self.step_outputs: dict[str, str] = {}
        self.agent_notes: dict[str, dict[str, Any]] = {}
        self.confidence_signals: dict[str, dict[str, Any]] = {}
        self._event_log = EventLog()

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    # ── Core operations ──

    def read(self, region: str, key: str, reader: str = "") -> Any:
        """Read a value from a region. Logs a READ event."""
        self._validate_region(region)
        value = self._get_value(region, key)
        self._event_log.append(
            BlackboardEvent(
                event_type=EventType.READ,
                region=region,
                key=key,
                agent_name=reader,
            )
        )
        return value

    def write(self, region: str, key: str, value: Any, writer: str = "") -> None:
        """Write a value to a region. Validates and logs a WRITE event."""
        self._validate_region(region)
        self._set_value(region, key, value)
        self._event_log.append(
            BlackboardEvent(
                event_type=EventType.WRITE,
                region=region,
                key=key,
                value=value,
                agent_name=writer,
            )
        )

    def query(self, region: str, filter_fn: Callable[[Any], bool]) -> list[Any]:
        """Query a region with a filter function."""
        self._validate_region(region)
        store = self._get_region_store(region)
        if isinstance(store, dict):
            return [v for v in store.values() if filter_fn(v)]
        # For DocumentMetadata, filter against fields
        return [getattr(store, f) for f in store.model_fields if filter_fn(getattr(store, f))]

    def subscribe(self, regions: list[str]) -> BlackboardView:
        """Return a read-only view of specific regions for prompt injection."""
        data: dict[str, Any] = {}
        for region in regions:
            base_region = region.split(".")[0]
            self._validate_region(base_region)
            data[base_region] = self._serialize_region(base_region)
        return BlackboardView(data)

    def snapshot(self) -> dict[str, Any]:
        """Frozen snapshot for cache key computation."""
        return {
            "document_metadata": self.document_metadata.model_dump(),
            "page_observations": {k: v.model_dump() for k, v in self.page_observations.items()},
            "step_outputs": dict(self.step_outputs),
            "agent_notes": copy.deepcopy(self.agent_notes),
            "confidence_signals": copy.deepcopy(self.confidence_signals),
        }

    def to_jinja_context(self, subscriptions: list[str]) -> dict[str, Any]:
        """Serialize subscribed regions into a dict for Jinja2 prompt rendering."""
        view = self.subscribe(subscriptions)
        return view.to_dict()

    def copy(self) -> Blackboard:
        """Create a deep copy for parallel step execution."""
        new = Blackboard()
        new.document_metadata = self.document_metadata.model_copy(deep=True)
        new.page_observations = {
            k: v.model_copy(deep=True) for k, v in self.page_observations.items()
        }
        new.step_outputs = dict(self.step_outputs)
        new.agent_notes = copy.deepcopy(self.agent_notes)
        new.confidence_signals = copy.deepcopy(self.confidence_signals)
        # Event log is NOT copied — each branch gets its own
        return new

    # ── Internal helpers ──

    def _validate_region(self, region: str) -> None:
        if region not in VALID_REGIONS:
            raise ValueError(f"Invalid blackboard region: '{region}'. Valid: {VALID_REGIONS}")

    def _get_region_store(self, region: str) -> Any:
        return getattr(self, region)

    def _get_value(self, region: str, key: str) -> Any:
        store = self._get_region_store(region)
        if isinstance(store, DocumentMetadata):
            return getattr(store, key, None)
        if isinstance(store, dict):
            # Support dotted keys like "3.quality_score" for page_observations
            parts = key.split(".", 1)
            typed_key: Any = parts[0]
            if region == "page_observations":
                typed_key = int(typed_key)
            val = store.get(typed_key)
            if val is not None and len(parts) > 1:
                if isinstance(val, PageObservation):
                    return getattr(val, parts[1], None)
                if isinstance(val, dict):
                    return val.get(parts[1])
            return val
        return None

    def _set_value(self, region: str, key: str, value: Any) -> None:
        store = self._get_region_store(region)

        if region == "document_metadata":
            if hasattr(store, key):
                old = getattr(store, key)
                if old is not None and old != value:
                    logger.warning(
                        "Blackboard conflict: document_metadata.%s changing from %r to %r",
                        key,
                        old,
                        value,
                    )
            setattr(self.document_metadata, key, value)

        elif region == "page_observations":
            parts = key.split(".", 1)
            page_num = int(parts[0])
            if page_num not in self.page_observations:
                self.page_observations[page_num] = PageObservation()
            if len(parts) > 1:
                setattr(self.page_observations[page_num], parts[1], value)
            elif isinstance(value, PageObservation):
                self.page_observations[page_num] = value
            elif isinstance(value, dict):
                obs = self.page_observations[page_num]
                for k, v in value.items():
                    setattr(obs, k, v)

        elif region == "step_outputs":
            self.step_outputs[key] = value

        elif region == "agent_notes":
            parts = key.split(".", 1)
            agent_name = parts[0]
            if agent_name not in self.agent_notes:
                self.agent_notes[agent_name] = {}
            if len(parts) > 1:
                self.agent_notes[agent_name][parts[1]] = value
            elif isinstance(value, dict):
                self.agent_notes[agent_name].update(value)
            else:
                self.agent_notes[agent_name] = value

        elif region == "confidence_signals":
            self.confidence_signals[key] = value

    def _serialize_region(self, region: str) -> Any:
        store = self._get_region_store(region)
        if isinstance(store, DocumentMetadata):
            return store.model_dump(exclude_none=True)
        if region == "page_observations":
            return {k: v.model_dump(exclude_none=True) for k, v in store.items()}
        return copy.deepcopy(store)
