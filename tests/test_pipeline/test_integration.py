"""Integration test: pipeline with blackboard propagation across steps."""

from unittest.mock import AsyncMock, patch

from doc2md.core import Doc2Md
from doc2md.types import TokenUsage, VLMResponse


class TestPipelineIntegration:
    async def test_convert_with_pipeline(self, tmp_path, sample_image_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        # Create agent YAMLs
        for name in ("extract", "validate"):
            (tmp_path / f"{name}.yaml").write_text(f"""
agent:
  name: {name}
  version: "1.0"
  description: "{name} agent"
  model:
    preferred: gpt-4.1-mini
    max_tokens: 1024
    temperature: 0.0
  prompt:
    system: "System for {name}"
    user: "User for {name}"
""")

        # Create pipeline YAML
        (tmp_path / "simple.yaml").write_text("""
pipeline:
  name: simple
  version: "1.0"
  description: "Simple test pipeline"
  steps:
    - name: extract_step
      agent: extract
      input: image
    - name: validate_step
      agent: validate
      input: image_and_previous
      depends_on: [extract_step]
""")

        converter = Doc2Md(
            api_key="test-key",
            custom_dir=str(tmp_path),
            no_cache=True,
        )

        mock_response = VLMResponse(
            content="# Extracted Content",
            model="gpt-4.1-mini",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

        with patch.object(converter, "_get_vlm_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client

            result = await converter.convert_async(img_path, pipeline="simple")

        assert result.classified_as == "simple"
        assert result.pages_processed == 1
        assert "# Extracted Content" in result.markdown

    async def test_convert_single_agent_wraps_in_pipeline(self, tmp_path, sample_image_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        converter = Doc2Md(api_key="test-key", no_cache=True)

        mock_response = VLMResponse(
            content="# Output",
            model="gpt-4.1-mini",
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
        )

        with patch.object(converter, "_get_vlm_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.send_request = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client

            result = await converter.convert_async(img_path)

        assert result.classified_as == "generic"
        assert "# Output" in result.markdown
