"""Tests for blackboard event log."""

from doc2md.blackboard.events import BlackboardEvent, EventLog, EventType


class TestBlackboardEvent:
    def test_write_event(self):
        event = BlackboardEvent(
            event_type=EventType.WRITE,
            region="document_metadata",
            key="language",
            value="fr",
            agent_name="metadata_extract",
        )
        assert event.event_type == EventType.WRITE
        assert event.value == "fr"

    def test_read_event_no_value(self):
        event = BlackboardEvent(
            event_type=EventType.READ,
            region="document_metadata",
            key="language",
            agent_name="text_extract",
        )
        assert event.value is None


class TestEventLog:
    def test_append_and_len(self):
        log = EventLog()
        assert len(log) == 0
        log.append(BlackboardEvent(
            event_type=EventType.WRITE, region="r", key="k", agent_name="a"
        ))
        assert len(log) == 1

    def test_events_returns_copy(self):
        log = EventLog()
        log.append(BlackboardEvent(
            event_type=EventType.WRITE, region="r", key="k", agent_name="a"
        ))
        events = log.events
        events.clear()
        assert len(log) == 1  # Original unaffected

    def test_query_by_agent(self):
        log = EventLog()
        log.append(BlackboardEvent(
            event_type=EventType.WRITE, region="r", key="k", agent_name="agent_a"
        ))
        log.append(BlackboardEvent(
            event_type=EventType.READ, region="r", key="k", agent_name="agent_b"
        ))
        assert len(log.query_by_agent("agent_a")) == 1
        assert len(log.query_by_agent("agent_b")) == 1
        assert len(log.query_by_agent("agent_c")) == 0

    def test_query_by_region(self):
        log = EventLog()
        log.append(BlackboardEvent(
            event_type=EventType.WRITE, region="step_outputs", key="k", agent_name="a"
        ))
        log.append(BlackboardEvent(
            event_type=EventType.WRITE, region="agent_notes", key="k", agent_name="a"
        ))
        assert len(log.query_by_region("step_outputs")) == 1

    def test_query_writes_and_reads(self):
        log = EventLog()
        log.append(BlackboardEvent(
            event_type=EventType.WRITE, region="r", key="k", agent_name="a"
        ))
        log.append(BlackboardEvent(
            event_type=EventType.READ, region="r", key="k", agent_name="b"
        ))
        assert len(log.query_writes()) == 1
        assert len(log.query_reads()) == 1
