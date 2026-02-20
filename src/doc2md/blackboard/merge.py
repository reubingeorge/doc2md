"""Merge logic for parallel step blackboard copies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from doc2md.blackboard.regions import DocumentMetadata, PageObservation

if TYPE_CHECKING:
    from doc2md.blackboard.board import Blackboard

logger = logging.getLogger(__name__)


def merge_parallel(target: Blackboard, sources: list[Blackboard]) -> None:
    """Merge blackboard writes from parallel branches into the target.

    Merge rules per region:
      document_metadata  — last-write-wins with warning on conflict
      page_observations  — deep merge per-page, per-field
      step_outputs       — keyed by step name (no conflict possible)
      agent_notes        — keyed by agent name (no conflict possible)
      confidence_signals — keyed by step+page (no conflict possible)
    """
    for source in sources:
        _merge_document_metadata(target, source)
        _merge_page_observations(target, source)
        _merge_step_outputs(target, source)
        _merge_agent_notes(target, source)
        _merge_confidence_signals(target, source)


def _merge_document_metadata(target: Blackboard, source: Blackboard) -> None:
    for field_name in DocumentMetadata.model_fields:
        source_val = getattr(source.document_metadata, field_name)
        target_val = getattr(target.document_metadata, field_name)
        if source_val is None:
            continue
        if field_name == "extra":
            if source_val:
                target.document_metadata.extra.update(source_val)
            continue
        if target_val is not None and target_val != source_val:
            logger.warning(
                "Blackboard merge conflict: document_metadata.%s: %r vs %r (using latter)",
                field_name,
                target_val,
                source_val,
            )
        setattr(target.document_metadata, field_name, source_val)


def _merge_page_observations(target: Blackboard, source: Blackboard) -> None:
    for page_num, source_obs in source.page_observations.items():
        if page_num not in target.page_observations:
            target.page_observations[page_num] = source_obs.model_copy(deep=True)
        else:
            _deep_merge_observation(target.page_observations[page_num], source_obs)


def _deep_merge_observation(target_obs: PageObservation, source_obs: PageObservation) -> None:
    """Merge non-default fields from source into target."""
    for field_name, field_info in PageObservation.model_fields.items():
        source_val = getattr(source_obs, field_name)
        default = field_info.default
        # Skip fields still at their default
        if source_val == default:
            continue
        if field_name == "uncertain_regions":
            existing = {ur.model_dump_json() for ur in target_obs.uncertain_regions}
            for ur in source_obs.uncertain_regions:
                if ur.model_dump_json() not in existing:
                    target_obs.uncertain_regions.append(ur)
                    existing.add(ur.model_dump_json())
        elif field_name == "content_types":
            merged = list(dict.fromkeys(target_obs.content_types + source_obs.content_types))
            target_obs.content_types = merged
        elif field_name == "extra":
            target_obs.extra.update(source_obs.extra)
        else:
            setattr(target_obs, field_name, source_val)


def _merge_step_outputs(target: Blackboard, source: Blackboard) -> None:
    target.step_outputs.update(source.step_outputs)


def _merge_agent_notes(target: Blackboard, source: Blackboard) -> None:
    for agent, notes in source.agent_notes.items():
        if agent not in target.agent_notes:
            target.agent_notes[agent] = {}
        target.agent_notes[agent].update(notes)


def _merge_confidence_signals(target: Blackboard, source: Blackboard) -> None:
    target.confidence_signals.update(source.confidence_signals)
