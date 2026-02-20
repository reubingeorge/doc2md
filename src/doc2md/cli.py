"""Click CLI for doc2md — convert documents to markdown."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from doc2md.config.hierarchy import load_config_hierarchy

console = Console()
error_console = Console(stderr=True)


def _setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=error_console, show_time=False, show_path=False)],
    )


@click.group()
@click.version_option(package_name="doc2md")
def cli() -> None:
    """doc2md — Agentic document-to-markdown converter."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file path.")
@click.option("--output-dir", type=click.Path(), help="Output directory for batch conversion.")
@click.option("--pipeline", type=str, default=None, help="Pipeline name to use.")
@click.option("--agent", type=str, default=None, help="Single agent name.")
@click.option("--model", type=str, default=None, help="Override model for all agents.")
@click.option("--workers", type=int, default=None, help="Concurrent workers for batch.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable caching.")
@click.option("--per-page", is_flag=True, default=False, help="Save each page as a separate file.")
@click.option(
    "--custom-dir", type=click.Path(exists=True), help="Directory with custom agent/pipeline YAMLs."
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v info, -vv debug).")
def convert(
    input_path: str,
    output: str | None,
    output_dir: str | None,
    pipeline: str | None,
    agent: str | None,
    model: str | None,
    workers: int | None,
    no_cache: bool,
    per_page: bool,
    custom_dir: str | None,
    verbose: int,
) -> None:
    """Convert document(s) to markdown."""
    _setup_logging(verbose)

    config = load_config_hierarchy(
        model=model,
        max_workers=workers,
        cache_disabled=no_cache or None,
    )

    input_path_obj = Path(input_path)

    if input_path_obj.is_dir():
        _convert_batch(
            input_path_obj, output_dir, pipeline, agent, model, config, no_cache, custom_dir
        )
    else:
        _convert_single(
            input_path_obj,
            output,
            pipeline,
            agent,
            model,
            config,
            no_cache,
            per_page,
            custom_dir,
            verbose,
        )


def _convert_single(
    input_path: Path,
    output: str | None,
    pipeline: str | None,
    agent: str | None,
    model: str | None,
    config: dict,
    no_cache: bool,
    per_page: bool,
    custom_dir: str | None,
    verbose: int,
) -> None:
    """Convert a single file."""
    from doc2md.core import Doc2Md

    api_key = config.get("api_key")
    base_url = config.get("base_url")

    converter = Doc2Md(
        api_key=api_key,
        base_url=base_url,
        no_cache=no_cache,
        custom_dir=custom_dir,
    )

    try:
        result = asyncio.run(
            converter.convert_async(input_path, agent=agent, pipeline=pipeline, model=model)
        )
    except FileNotFoundError as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        asyncio.run(converter.close())

    # Output
    if output:
        written = result.save(output, per_page=per_page)
        for p in written:
            console.print(f"[green]Written to {p}[/green]")
    else:
        console.print(result.markdown)

    # Verbose summary
    if verbose >= 1:
        _print_summary(result, verbose)


def _convert_batch(
    input_dir: Path,
    output_dir: str | None,
    pipeline: str | None,
    agent: str | None,
    model: str | None,
    config: dict,
    no_cache: bool,
    custom_dir: str | None = None,
) -> None:
    """Convert all supported files in a directory."""
    from doc2md.core import Doc2Md

    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in supported]

    if not files:
        error_console.print("[yellow]No supported files found in directory.[/yellow]")
        return

    out_dir = Path(output_dir) if output_dir else input_dir / "markdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = config.get("api_key")
    base_url = config.get("base_url")
    max_workers = config.get("max_workers", 5)

    converter = Doc2Md(api_key=api_key, base_url=base_url, no_cache=no_cache, custom_dir=custom_dir)

    from doc2md.concurrency.pool import ConcurrencyPool

    async def _run() -> None:
        pool = ConcurrencyPool(max_file_workers=max_workers)
        results = await pool.process_batch(
            converter.convert_async,
            file_paths=[str(f) for f in files],
            agent=agent,
            pipeline=pipeline,
            model=model,
        )
        for file, result in zip(files, results, strict=False):
            out_path = out_dir / f"{file.stem}.md"
            out_path.write_text(result.markdown)

        console.print(f"[green]Converted {len(results)} files to {out_dir}[/green]")

    try:
        asyncio.run(_run())
    except Exception as e:
        error_console.print(f"[red]Error during batch conversion:[/red] {e}")
        sys.exit(1)
    finally:
        asyncio.run(converter.close())


def _print_summary(result: object, verbose: int) -> None:
    """Print a conversion summary."""
    from doc2md.types import ConversionResult

    if not isinstance(result, ConversionResult):
        return

    error_console.print()
    table = Table(title="Conversion Summary", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    if result.classified_as:
        table.add_row("Pipeline", result.classified_as)
    table.add_row("Pages processed", str(result.pages_processed))
    table.add_row("Steps", str(len(result.steps)))

    if result.confidence is not None:
        level = result.confidence_level.value if result.confidence_level else "N/A"
        table.add_row("Confidence", f"{result.confidence:.2f} ({level})")

    if result.needs_human_review:
        table.add_row("Human review", "[yellow]Needed[/yellow]")

    usage = result.token_usage
    table.add_row(
        "Tokens",
        f"{usage.total_tokens:,} (prompt: {usage.prompt_tokens:,}, completion: {usage.completion_tokens:,})",
    )

    error_console.print(table)

    # Show step details at -vv
    if verbose >= 2 and result.steps:
        step_table = Table(title="Step Details", show_header=True)
        step_table.add_column("Step")
        step_table.add_column("Agent")
        step_table.add_column("Tokens")
        step_table.add_column("Cached")
        step_table.add_column("Confidence")

        for name, step in result.steps.items():
            conf = f"{step.confidence:.2f}" if step.confidence else "-"
            step_table.add_row(
                name,
                step.agent_name,
                str(step.token_usage.total_tokens),
                "yes" if step.cached else "no",
                conf,
            )
        error_console.print(step_table)


@cli.command("pipelines")
def list_pipelines() -> None:
    """List available pipelines."""
    from doc2md.agents.registry import PipelineRegistry

    registry = PipelineRegistry()

    table = Table(title="Available Pipelines", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("Steps")

    for info in sorted(registry.list_pipelines(), key=lambda p: p.name):
        table.add_row(
            info.name,
            info.version,
            info.description or "-",
            str(info.step_count),
        )

    console.print(table)


@cli.group()
def cache() -> None:
    """Cache management commands."""


@cache.command("stats")
def cache_stats() -> None:
    """Show cache statistics."""
    from doc2md.cache.manager import CacheManager

    mgr = CacheManager()

    table = Table(title="Cache Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    stats = mgr.stats()
    table.add_row("Entries", str(stats.entries))
    table.add_row("Size (MB)", f"{stats.size_mb:.1f}")
    table.add_row("Hits", str(stats.hits))
    table.add_row("Misses", str(stats.misses))
    table.add_row("Hit rate", f"{stats.hit_rate:.1%}")
    table.add_row("Tokens saved", f"{stats.tokens_saved:,}")

    console.print(table)
    mgr.close()


@cache.command("clear")
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def cache_clear() -> None:
    """Clear all cached data."""
    from doc2md.cache.manager import CacheManager

    mgr = CacheManager()
    mgr.clear()
    mgr.close()
    console.print("[green]Cache cleared.[/green]")


@cli.command("validate-pipeline")
@click.argument("pipeline_yaml", type=click.Path(exists=True))
def validate_pipeline(pipeline_yaml: str) -> None:
    """Validate a pipeline YAML file."""
    from doc2md.config.loader import load_pipeline_yaml

    try:
        config = load_pipeline_yaml(pipeline_yaml)
        console.print(f"[green]Valid pipeline:[/green] {config.name} v{config.version}")
        console.print(f"  Steps: {len(config.steps)}")
        for step in config.steps:
            console.print(f"    - {step.name} ({step.type.value})")
    except Exception as e:
        error_console.print(f"[red]Invalid pipeline:[/red] {e}")
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()
