"""Integration test: convert() end-to-end with mocked VLM."""

from unittest.mock import AsyncMock, patch

import pytest

from doc2md.core import Doc2Md
from doc2md.types import TokenUsage, VLMResponse


def _mock_vlm_response() -> VLMResponse:
    return VLMResponse(
        content="# Receipt\n\nItem: Coffee\nTotal: $4.50",
        model="gpt-4.1-mini",
        token_usage=TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280),
    )


class TestConvertIntegration:
    async def test_convert_single_image(self, tmp_path, sample_image_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        converter = Doc2Md(api_key="test-key", no_cache=True)

        with patch.object(converter, "_get_vlm_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=_mock_vlm_response())
            mock_get_client.return_value = mock_client

            result = await converter.convert_async(img_path)

        assert "# Receipt" in result.markdown
        assert result.pages_processed == 1
        assert result.token_usage.total_tokens == 280
        assert result.classified_as == "generic"

    async def test_convert_uses_specified_agent(self, tmp_path, sample_image_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        agent_yaml = tmp_path / "custom.yaml"
        agent_yaml.write_text("""
agent:
  name: custom
  version: "1.0"
  description: "Custom agent"
  model:
    preferred: gpt-4.1
    max_tokens: 2048
    temperature: 0.0
  prompt:
    system: "Custom system prompt"
    user: "Custom user prompt"
""")

        converter = Doc2Md(api_key="test-key", custom_dir=str(tmp_path), no_cache=True)

        with patch.object(converter, "_get_vlm_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=_mock_vlm_response())
            mock_get_client.return_value = mock_client

            result = await converter.convert_async(img_path, agent="custom")

        assert result.classified_as == "custom"
        call_kwargs = mock_client.send_request.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1"

    async def test_convert_model_override(self, tmp_path, sample_image_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        converter = Doc2Md(api_key="test-key", no_cache=True)

        with patch.object(converter, "_get_vlm_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=_mock_vlm_response())
            mock_get_client.return_value = mock_client

            await converter.convert_async(img_path, model="gpt-4.1")

        call_kwargs = mock_client.send_request.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1"

    async def test_convert_missing_agent_raises(self, tmp_path, sample_image_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        converter = Doc2Md(api_key="test-key", no_cache=True)
        with pytest.raises(KeyError, match="nonexistent"):
            await converter.convert_async(img_path, agent="nonexistent")
