"""Integration test: two agents sharing data via blackboard."""

from unittest.mock import AsyncMock

from doc2md.agents.engine import AgentEngine
from doc2md.blackboard.board import Blackboard
from doc2md.types import (
    AgentConfig,
    BlackboardConfig,
    InputMode,
    PromptConfig,
    TokenUsage,
    VLMResponse,
    WritesVia,
)


def _make_vlm_response(content: str) -> VLMResponse:
    return VLMResponse(
        content=content,
        model="gpt-4.1-mini",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


class TestBlackboardIntegration:
    async def test_metadata_agent_writes_language_text_agent_reads_it(self):
        """metadata_extract writes language → text_extract reads it in its prompt."""
        bb = Blackboard()

        # Agent 1: metadata_extract — writes language to blackboard via VLM
        metadata_agent = AgentConfig(
            name="metadata_extract",
            input=InputMode.IMAGE,
            prompt=PromptConfig(system="Extract metadata.", user="Metadata."),
            blackboard=BlackboardConfig(
                writes=["document_metadata.language"],
                writes_via=WritesVia.PROMPT_ELICITED,
            ),
        )

        mock_vlm = AsyncMock()
        mock_vlm.send_request = AsyncMock(
            return_value=_make_vlm_response(
                "---\ntitle: Test\n---\n\n"
                "<blackboard>\n"
                "document_metadata:\n"
                "  language: fr\n"
                "</blackboard>"
            )
        )

        engine = AgentEngine(mock_vlm)
        result1 = await engine.execute(
            agent_config=metadata_agent,
            image_bytes=b"fake",
            blackboard=bb,
        )

        # Verify language was written to blackboard
        assert bb.document_metadata.language == "fr"
        assert "<blackboard>" not in result1.markdown

        # Agent 2: text_extract — reads language from blackboard
        text_agent = AgentConfig(
            name="text_extract",
            input=InputMode.IMAGE,
            prompt=PromptConfig(
                system="{% if bb.document_metadata.language %}Language: {{ bb.document_metadata.language }}{% endif %}",
                user="Extract text.",
            ),
            blackboard=BlackboardConfig(
                reads=["document_metadata.language"],
            ),
        )

        mock_vlm.send_request = AsyncMock(
            return_value=_make_vlm_response("# Chapitre 1\n\nLe texte en français.")
        )

        result2 = await engine.execute(
            agent_config=text_agent,
            image_bytes=b"fake",
            blackboard=bb,
        )

        # Verify the VLM was called with language in the system prompt
        call_kwargs = mock_vlm.send_request.call_args.kwargs
        assert "Language: fr" in call_kwargs["system_prompt"]
        assert "Chapitre" in result2.markdown

    async def test_event_log_tracks_cross_agent_flow(self):
        """Event log records the full audit trail across agents."""
        bb = Blackboard()

        mock_vlm = AsyncMock()
        mock_vlm.send_request = AsyncMock(
            return_value=_make_vlm_response(
                "Content\n<blackboard>\ndocument_metadata:\n  layout: two_column\n</blackboard>"
            )
        )
        engine = AgentEngine(mock_vlm)

        agent = AgentConfig(
            name="agent_a",
            prompt=PromptConfig(system="S", user="U"),
            blackboard=BlackboardConfig(
                reads=["document_metadata"],
                writes=["document_metadata.layout"],
                writes_via=WritesVia.PROMPT_ELICITED,
            ),
        )

        await engine.execute(agent_config=agent, image_bytes=b"fake", blackboard=bb)

        # Should have both reads (for prompt build) and writes (from response)
        assert len(bb.event_log) > 0
        writes = bb.event_log.query_writes()
        assert any(e.key == "layout" for e in writes)
