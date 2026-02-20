"""Blackboard — typed, region-based shared memory for pipeline execution."""

from doc2md.blackboard.board import Blackboard, BlackboardView
from doc2md.blackboard.events import BlackboardEvent, EventLog, EventType
from doc2md.blackboard.regions import DocumentMetadata, PageObservation, UncertainRegion

__all__ = [
    "Blackboard",
    "BlackboardView",
    "BlackboardEvent",
    "EventLog",
    "EventType",
    "DocumentMetadata",
    "PageObservation",
    "UncertainRegion",
]
