"""Tests for document auto-classification."""

from unittest.mock import AsyncMock

import pytest

from doc2md.agents.classifier import classify_document
from doc2md.agents.registry import PipelineRegistry
from doc2md.blackboard.board import Blackboard
from doc2md.types import TokenUsage, VLMResponse


def _mock_vlm_response(content: str) -> VLMResponse:
    return VLMResponse(
        content=content,
        model="gpt-4.1-nano",
        token_usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    )


class TestClassifyDocument:
    @pytest.fixture
    def registry(self):
        return PipelineRegistry()

    @pytest.fixture
    def mock_vlm(self):
        return AsyncMock()

    async def test_successful_classification(self, registry, mock_vlm):
        mock_vlm.send_request = AsyncMock(
            return_value=_mock_vlm_response(
                '{"pipeline_name": "receipt", "confidence": 0.92, '
                '"reasoning": "Looks like a receipt", '
                '"content_types_detected": ["tables", "prose"]}'
            )
        )

        result = await classify_document(b"img", registry, mock_vlm)

        assert result.pipeline_name == "receipt"
        assert result.confidence == 0.92
        assert "tables" in result.content_types_detected

    async def test_low_confidence_falls_back_to_generic(self, registry, mock_vlm):
        mock_vlm.send_request = AsyncMock(
            return_value=_mock_vlm_response(
                '{"pipeline_name": "legal_contract", "confidence": 0.3, '
                '"reasoning": "Not sure", "content_types_detected": []}'
            )
        )

        result = await classify_document(b"img", registry, mock_vlm)

        assert result.pipeline_name == "generic"

    async def test_unknown_pipeline_falls_back(self, registry, mock_vlm):
        mock_vlm.send_request = AsyncMock(
            return_value=_mock_vlm_response(
                '{"pipeline_name": "nonexistent_pipeline", "confidence": 0.95, '
                '"reasoning": "Test", "content_types_detected": []}'
            )
        )

        result = await classify_document(b"img", registry, mock_vlm)

        assert result.pipeline_name == "generic"

    async def test_invalid_json_falls_back(self, registry, mock_vlm):
        mock_vlm.send_request = AsyncMock(
            return_value=_mock_vlm_response("This is not valid JSON at all")
        )

        result = await classify_document(b"img", registry, mock_vlm)

        assert result.pipeline_name == "generic"
        assert result.confidence == 0.0

    async def test_vlm_error_falls_back(self, registry, mock_vlm):
        mock_vlm.send_request = AsyncMock(side_effect=Exception("API Error"))

        result = await classify_document(b"img", registry, mock_vlm)

        assert result.pipeline_name == "generic"
        assert result.confidence == 0.0

    async def test_writes_content_types_to_blackboard(self, registry, mock_vlm):
        mock_vlm.send_request = AsyncMock(
            return_value=_mock_vlm_response(
                '{"pipeline_name": "receipt", "confidence": 0.9, '
                '"reasoning": "Receipt", '
                '"content_types_detected": ["tables", "prose"]}'
            )
        )
        bb = Blackboard()

        await classify_document(b"img", registry, mock_vlm, blackboard=bb)

        assert bb.document_metadata.content_types == ["tables", "prose"]

    async def test_strips_markdown_fences_from_json(self, registry, mock_vlm):
        mock_vlm.send_request = AsyncMock(
            return_value=_mock_vlm_response(
                '```json\n{"pipeline_name": "academic", "confidence": 0.85, '
                '"reasoning": "Paper", "content_types_detected": ["prose"]}\n```'
            )
        )

        result = await classify_document(b"img", registry, mock_vlm)

        assert result.pipeline_name == "academic"
        assert result.confidence == 0.85
