"""Top-level entry points: convert(), convert_batch(), Doc2Md."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from doc2md.agents.classifier import ClassificationResult, classify_document
from doc2md.agents.engine import AgentEngine
from doc2md.agents.registry import AgentRegistry, PipelineRegistry
from doc2md.blackboard.board import Blackboard
from doc2md.cache.manager import CacheManager
from doc2md.config.schema import PipelineConfig, StepConfig, StepType
from doc2md.pipeline.engine import PipelineEngine
from doc2md.types import AgentConfig, ConversionResult
from doc2md.utils.image import is_pdf, load_image, pdf_to_images
from doc2md.vlm.client import AsyncVLMClient

logger = logging.getLogger(__name__)


class Doc2Md:
    """Main converter class with full lifecycle control."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        config_dir: str | None = None,
        pipeline_dir: str | None = None,
        no_cache: bool = False,
        cache_memory_mb: float = 500,
        cache_disk_mb: float = 5000,
        cache_db_path: Path | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._vlm_client: AsyncVLMClient | None = None

        # Build registries from builtin + user directories
        user_agent_dirs = [Path(config_dir)] if config_dir else []
        user_pipeline_dirs = [Path(pipeline_dir)] if pipeline_dir else []
        self._agent_registry = AgentRegistry(user_dirs=user_agent_dirs)
        self._pipeline_registry = PipelineRegistry(user_dirs=user_pipeline_dirs)

        # Cache
        self._cache_manager = CacheManager(
            memory_max_mb=cache_memory_mb,
            disk_max_mb=cache_disk_mb,
            disk_path=cache_db_path,
            enabled=not no_cache,
        )

    @property
    def agent_registry(self) -> AgentRegistry:
        return self._agent_registry

    @property
    def pipeline_registry(self) -> PipelineRegistry:
        return self._pipeline_registry

    @property
    def cache_manager(self) -> CacheManager:
        return self._cache_manager

    async def convert_async(
        self,
        input_path: str | Path,
        agent: str | None = None,
        pipeline: str | None = None,
        model: str | None = None,
        auto_classify: bool = True,
    ) -> ConversionResult:
        """Convert a document to markdown asynchronously."""
        input_path = Path(input_path)
        page_images = self._load_pages(input_path)
        vlm_client = self._get_vlm_client()
        agent_engine = AgentEngine(vlm_client)
        blackboard = Blackboard()

        # Resolve what to run
        pipeline_config, agent_configs, classification = await self._resolve_execution(
            agent=agent,
            pipeline=pipeline,
            model=model,
            page_images=page_images,
            vlm_client=vlm_client,
            blackboard=blackboard,
            auto_classify=auto_classify,
        )

        # Execute pipeline
        pipeline_engine = PipelineEngine(
            agent_engine,
            agent_configs,
            cache_manager=self._cache_manager,
        )
        result = await pipeline_engine.execute(pipeline_config, page_images, blackboard)

        # Build conversion result with confidence data
        conf_report = result.confidence_report
        return ConversionResult(
            markdown=result.markdown,
            classified_as=result.pipeline_name,
            steps=result.steps,
            token_usage=result.token_usage,
            pages_processed=len(page_images),
            confidence=conf_report.overall if conf_report else None,
            confidence_level=conf_report.level if conf_report else None,
            needs_human_review=conf_report.needs_human_review if conf_report else False,
        )

    async def _resolve_execution(
        self,
        agent: str | None,
        pipeline: str | None,
        model: str | None,
        page_images: list[bytes],
        vlm_client: AsyncVLMClient,
        blackboard: Blackboard,
        auto_classify: bool,
    ) -> tuple[PipelineConfig, dict[str, AgentConfig], ClassificationResult | None]:
        """Resolve to pipeline config + agent configs.

        Priority:
          1. pipeline= specified → use that pipeline
          2. agent= specified → wrap in implicit single-step pipeline
          3. auto_classify=True → classify page 1 and select pipeline
          4. fallback → generic agent in implicit pipeline
        """
        classification = None

        if pipeline:
            pc, ac = self._resolve_pipeline(pipeline, model)
            return pc, ac, None

        if agent:
            pc, ac = self._resolve_agent(agent, model)
            return pc, ac, None

        # Auto-classification
        if auto_classify and page_images:
            try:
                classification = await classify_document(
                    page1_image=page_images[0],
                    pipeline_registry=self._pipeline_registry,
                    vlm_client=vlm_client,
                    blackboard=blackboard,
                )
                logger.info(
                    "Classified as '%s' (confidence=%.2f)",
                    classification.pipeline_name,
                    classification.confidence,
                )
                pc, ac = self._resolve_pipeline(classification.pipeline_name, model)
                return pc, ac, classification
            except Exception as e:
                logger.warning("Auto-classification failed: %s. Using generic.", e)

        # Fallback
        pc, ac = self._resolve_agent("generic", model)
        return pc, ac, None

    def _resolve_pipeline(
        self,
        pipeline_name: str,
        model: str | None,
    ) -> tuple[PipelineConfig, dict[str, AgentConfig]]:
        """Load a pipeline and all its referenced agents from registries."""
        pipeline_config = self._pipeline_registry.get(pipeline_name)
        agent_names = _collect_agent_names(pipeline_config.steps)
        agent_configs: dict[str, AgentConfig] = {}
        for name in agent_names:
            config = self._agent_registry.get(name)
            if model:
                config = config.model_copy(deep=True)
                config.model.preferred = model
            agent_configs[name] = config
        return pipeline_config, agent_configs

    def _resolve_agent(
        self,
        agent_name: str,
        model: str | None,
    ) -> tuple[PipelineConfig, dict[str, AgentConfig]]:
        """Wrap a single agent in an implicit pipeline."""
        config = self._agent_registry.get(agent_name)
        if model:
            config = config.model_copy(deep=True)
            config.model.preferred = model

        implicit_pipeline = PipelineConfig(
            name=agent_name,
            steps=[StepConfig(name="extract", type=StepType.AGENT, agent=agent_name)],
        )
        return implicit_pipeline, {agent_name: config}

    def _get_vlm_client(self) -> AsyncVLMClient:
        if self._vlm_client is None:
            self._vlm_client = AsyncVLMClient(api_key=self._api_key, base_url=self._base_url)
        return self._vlm_client

    @staticmethod
    def _load_pages(input_path: Path) -> list[bytes]:
        if is_pdf(input_path):
            return pdf_to_images(input_path)
        return [load_image(input_path)]

    async def close(self) -> None:
        if self._vlm_client:
            await self._vlm_client.close()
        self._cache_manager.close()


def _collect_agent_names(steps: list[StepConfig]) -> set[str]:
    """Recursively collect all agent names from step configs."""
    names: set[str] = set()
    for step in steps:
        if step.agent:
            names.add(step.agent)
        if step.steps:
            names.update(_collect_agent_names(step.steps))
        if step.router:
            for rule in step.router.rules:
                names.add(rule.agent)
            names.add(step.router.default_agent)
            if step.router.vlm_fallback:
                for cat in step.router.vlm_fallback.categories.values():
                    if "agent" in cat:
                        names.add(cat["agent"])
        if step.merge and step.merge.agent:
            names.add(step.merge.agent)
    return names


# ── Module-level convenience functions ──


def convert(
    input_path: str | Path,
    agent: str | None = None,
    pipeline: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    no_cache: bool = False,
) -> ConversionResult:
    """Convert a document to markdown (sync wrapper)."""
    converter = Doc2Md(api_key=api_key, no_cache=no_cache)
    try:
        return asyncio.run(
            converter.convert_async(input_path, agent=agent, pipeline=pipeline, model=model)
        )
    finally:
        asyncio.run(converter.close())


def convert_batch(
    input_paths: list[str | Path],
    agent: str | None = None,
    pipeline: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_workers: int = 5,
    no_cache: bool = False,
) -> list[ConversionResult]:
    """Convert multiple documents concurrently (sync wrapper)."""
    from doc2md.concurrency.pool import ConcurrencyPool

    converter = Doc2Md(api_key=api_key, no_cache=no_cache)

    async def _run() -> list[ConversionResult]:
        pool = ConcurrencyPool(max_file_workers=max_workers)
        results = await pool.process_batch(
            converter.convert_async,
            file_paths=input_paths,
            agent=agent,
            pipeline=pipeline,
            model=model,
        )
        return results

    try:
        return asyncio.run(_run())
    finally:
        asyncio.run(converter.close())
