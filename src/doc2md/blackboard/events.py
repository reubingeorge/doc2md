"""Blackboard event log — append-only audit trail of all reads and writes."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

import time


class EventType(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


class BlackboardEvent(BaseModel):
    """A single blackboard read or write event."""

    timestamp: float = Field(default_factory=time.monotonic)
    event_type: EventType
    region: str
    key: str
    value: Any = None  # Only populated for WRITE events
    agent_name: str = ""


class EventLog:
    """Append-only, queryable event log."""

    def __init__(self) -> None:
        self._events: list[BlackboardEvent] = []

    def append(self, event: BlackboardEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> list[BlackboardEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def query_by_agent(self, agent_name: str) -> list[BlackboardEvent]:
        return [e for e in self._events if e.agent_name == agent_name]

    def query_by_region(self, region: str) -> list[BlackboardEvent]:
        return [e for e in self._events if e.region == region]

    def query_by_type(self, event_type: EventType) -> list[BlackboardEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def query_writes(self) -> list[BlackboardEvent]:
        return self.query_by_type(EventType.WRITE)

    def query_reads(self) -> list[BlackboardEvent]:
        return self.query_by_type(EventType.READ)
