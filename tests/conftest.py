import pytest


@pytest.fixture
def sample_image_bytes():
    """Minimal valid PNG for testing (1x1 white pixel)."""
    import base64
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def sample_agent_yaml(tmp_path):
    """Write a minimal agent YAML and return its path."""
    content = """
agent:
  name: test_agent
  version: "1.0"
  description: "Test agent"
  model:
    preferred: gpt-4.1-mini
    max_tokens: 1024
    temperature: 0.0
  prompt:
    system: "Extract text from this document image."
    user: "Extract the content."
"""
    path = tmp_path / "test_agent.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def sample_pipeline_yaml(tmp_path):
    """Write a minimal pipeline YAML and return its path."""
    content = """
pipeline:
  name: test_pipeline
  version: "1.0"
  description: "Test pipeline"
  steps:
    - name: extract
      agent: test_agent
      input: image
    - name: validate
      agent: test_agent
      input: image_and_previous
      depends_on: [extract]
"""
    path = tmp_path / "test_pipeline.yaml"
    path.write_text(content)
    return path
