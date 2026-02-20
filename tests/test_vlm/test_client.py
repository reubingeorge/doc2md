"""Tests for VLM client message building."""

from doc2md.vlm.client import AsyncVLMClient


class TestBuildMessages:
    def test_text_only_message(self):
        msgs = AsyncVLMClient._build_messages("System", "User prompt", None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "User prompt"

    def test_image_message(self):
        msgs = AsyncVLMClient._build_messages("System", "Describe", "abc123")
        assert len(msgs) == 2
        user_content = msgs[1]["content"]
        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "text"
        assert user_content[0]["text"] == "Describe"
        assert user_content[1]["type"] == "image_url"
        assert "abc123" in user_content[1]["image_url"]["url"]
