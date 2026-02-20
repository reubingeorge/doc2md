"""Tests for cache key generation."""

from doc2md.cache.keys import generate_cache_key, hash_image, hash_prompt


class TestHashImage:
    def test_deterministic(self):
        data = b"fake image data"
        assert hash_image(data) == hash_image(data)

    def test_different_data_different_hash(self):
        assert hash_image(b"aaa") != hash_image(b"bbb")

    def test_returns_hex_string(self):
        h = hash_image(b"test")
        assert len(h) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in h)


class TestHashPrompt:
    def test_deterministic(self):
        h1 = hash_prompt("system", "user")
        h2 = hash_prompt("system", "user")
        assert h1 == h2

    def test_different_prompts_different_hash(self):
        assert hash_prompt("sys1", "user") != hash_prompt("sys2", "user")
        assert hash_prompt("sys", "user1") != hash_prompt("sys", "user2")


class TestGenerateCacheKey:
    def test_deterministic(self):
        kwargs = dict(
            image_hash="abc123",
            pipeline_name="generic",
            step_name="extract",
            agent_name="text_extract",
            agent_version="1.0",
            model_id="gpt-4.1-mini",
            prompt_hash="prompt123",
        )
        assert generate_cache_key(**kwargs) == generate_cache_key(**kwargs)

    def test_different_inputs_different_keys(self):
        base = dict(
            image_hash="abc123",
            pipeline_name="generic",
            step_name="extract",
            agent_name="text_extract",
            agent_version="1.0",
            model_id="gpt-4.1-mini",
            prompt_hash="prompt123",
        )
        k1 = generate_cache_key(**base)
        k2 = generate_cache_key(**{**base, "agent_version": "2.0"})
        assert k1 != k2

    def test_blackboard_snapshot_affects_key(self):
        base = dict(
            image_hash="abc",
            pipeline_name="p",
            step_name="s",
            agent_name="a",
            agent_version="1.0",
            model_id="m",
            prompt_hash="ph",
        )
        k1 = generate_cache_key(**base)
        k2 = generate_cache_key(**base, blackboard_snapshot={"lang": "en"})
        assert k1 != k2

    def test_blackboard_snapshot_order_independent(self):
        base = dict(
            image_hash="abc",
            pipeline_name="p",
            step_name="s",
            agent_name="a",
            agent_version="1.0",
            model_id="m",
            prompt_hash="ph",
        )
        snap1 = {"a": 1, "b": 2}
        snap2 = {"b": 2, "a": 1}
        assert generate_cache_key(**base, blackboard_snapshot=snap1) == generate_cache_key(
            **base, blackboard_snapshot=snap2
        )

    def test_returns_hex_string(self):
        k = generate_cache_key(
            image_hash="x",
            pipeline_name="p",
            step_name="s",
            agent_name="a",
            agent_version="1",
            model_id="m",
            prompt_hash="ph",
        )
        assert len(k) == 64
