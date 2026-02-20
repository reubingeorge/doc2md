"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner

from doc2md.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIGroup:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "doc2md" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestConvertCommand:
    def test_help(self, runner):
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--pipeline" in result.output
        assert "--agent" in result.output
        assert "--output" in result.output

    def test_missing_input(self, runner):
        result = runner.invoke(cli, ["convert"])
        assert result.exit_code != 0

    def test_nonexistent_file(self, runner):
        result = runner.invoke(cli, ["convert", "nonexistent_file.pdf"])
        assert result.exit_code != 0


class TestPipelinesCommand:
    def test_lists_pipelines(self, runner):
        result = runner.invoke(cli, ["pipelines"])
        assert result.exit_code == 0
        assert "generic" in result.output

    def test_shows_table(self, runner):
        result = runner.invoke(cli, ["pipelines"])
        assert "Available Pipelines" in result.output
        assert "Name" in result.output


class TestCacheCommands:
    def test_cache_help(self, runner):
        result = runner.invoke(cli, ["cache", "--help"])
        assert result.exit_code == 0
        assert "stats" in result.output
        assert "clear" in result.output

    def test_cache_stats(self, runner):
        result = runner.invoke(cli, ["cache", "stats"])
        assert result.exit_code == 0
        assert "Cache Statistics" in result.output

    def test_cache_clear_needs_confirmation(self, runner):
        result = runner.invoke(cli, ["cache", "clear"], input="n\n")
        assert result.exit_code != 0  # Aborted

    def test_cache_clear_with_yes(self, runner):
        result = runner.invoke(cli, ["cache", "clear", "--yes"])
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()


class TestValidatePipelineCommand:
    def test_valid_pipeline(self, runner, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "pipeline:\n"
            "  name: test_pipeline\n"
            "  steps:\n"
            "    - name: extract\n"
            "      type: agent\n"
            "      agent: generic\n"
        )
        result = runner.invoke(cli, ["validate-pipeline", str(yaml_file)])
        assert result.exit_code == 0
        assert "Valid pipeline" in result.output

    def test_invalid_pipeline(self, runner, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("not_pipeline:\n  foo: bar\n")
        result = runner.invoke(cli, ["validate-pipeline", str(yaml_file)])
        assert result.exit_code == 1
        assert "Invalid pipeline" in result.output

    def test_nonexistent_yaml(self, runner):
        result = runner.invoke(cli, ["validate-pipeline", "nonexistent.yaml"])
        assert result.exit_code != 0
