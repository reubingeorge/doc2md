"""Agent execution engine — orchestrates prompt building, VLM call, and response parsing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from doc2md.blackboard.writers import get_writer
from doc2md.pipeline.postprocessor import run_postprocessing
from doc2md.pipeline.preprocessor import run_preprocessing
from doc2md.types import AgentConfig, InputMode, StepResult
from doc2md.utils.image import image_to_base64
from doc2md.vlm.client import AsyncVLMClient
from doc2md.vlm.prompt_builder import build_prompt
from doc2md.vlm.response_parser import parse_response

if TYPE_CHECKING:
    from doc2md.blackboard.board import Blackboard
    from doc2md.cache.manager import CacheManager

logger = logging.getLogger(__name__)


class AgentEngine:
    """Executes a single agent: build prompt → call VLM → parse response."""

    def __init__(self, vlm_client: AsyncVLMClient) -> None:
        self._vlm = vlm_client

    async def execute(
        self,
        agent_config: AgentConfig,
        image_bytes: bytes | None = None,
        previous_output: str | None = None,
        blackboard: Blackboard | None = None,
        step_name: str | None = None,
        page_num: int | None = None,
        cache_manager: CacheManager | None = None,
        pipeline_name: str = "",
    ) -> StepResult:
        """Execute an agent and return the step result."""
        resolved_step = step_name or agent_config.name

        logger.info(
            "Agent '%s' starting (model=%s)", agent_config.name, agent_config.model.preferred
        )

        # Run image preprocessing if configured
        if image_bytes and agent_config.preprocessing:
            image_bytes, quality = run_preprocessing(image_bytes, agent_config.preprocessing)
            if blackboard and page_num is not None:
                blackboard.write(
                    "page_observations",
                    f"page_{page_num}.quality_score",
                    quality.overall,
                    writer=resolved_step,
                )

        # Read subscribed blackboard regions for prompt context
        bb_context = self._read_blackboard(agent_config, blackboard, resolved_step)

        image_b64 = self._resolve_image(agent_config.input, image_bytes)
        system_prompt, user_prompt = build_prompt(
            agent_config,
            image_b64=image_b64,
            previous_output=previous_output,
            blackboard_context=bb_context,
        )

        # Check cache before VLM call
        cached_result = self._check_cache(
            cache_manager,
            image_bytes,
            pipeline_name,
            resolved_step,
            agent_config,
            system_prompt,
            user_prompt,
            bb_context,
        )
        if cached_result is not None:
            logger.info("Cache hit for step '%s' agent '%s'", resolved_step, agent_config.name)
            # Re-apply blackboard writes from cached result
            if blackboard and cached_result.blackboard_writes:
                self._apply_blackboard_writes(
                    blackboard, cached_result.blackboard_writes, resolved_step
                )
            return cached_result

        logger.info("Calling VLM '%s' for step '%s'", agent_config.model.preferred, resolved_step)
        vlm_response = await self._vlm.send_request(
            model=agent_config.model.preferred,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_b64=image_b64,
            max_tokens=agent_config.model.max_tokens,
            temperature=agent_config.model.temperature,
        )

        markdown, metadata = parse_response(vlm_response.content)

        # Run markdown postprocessing if configured
        if agent_config.postprocessing:
            markdown = run_postprocessing(markdown, agent_config.postprocessing)

        # Write VLM-elicited blackboard data
        bb_writes = metadata.get("blackboard_writes", {})
        if blackboard and bb_writes:
            self._apply_blackboard_writes(blackboard, bb_writes, resolved_step)

        # Run code-computed writers
        if blackboard:
            self._run_code_writers(agent_config, blackboard, markdown, resolved_step, page_num)

        logger.info(
            "Agent '%s' done — %d chars, %d tokens (prompt=%d, completion=%d)",
            agent_config.name,
            len(markdown),
            vlm_response.token_usage.total_tokens,
            vlm_response.token_usage.prompt_tokens,
            vlm_response.token_usage.completion_tokens,
        )

        result = StepResult(
            step_name=resolved_step,
            agent_name=agent_config.name,
            markdown=markdown,
            token_usage=vlm_response.token_usage,
            confidence_level=metadata.get("confidence_level"),
            blackboard_writes=bb_writes,
            model_used=vlm_response.model,
        )

        # Store in cache
        self._store_in_cache(
            cache_manager,
            result,
            image_bytes,
            pipeline_name,
            resolved_step,
            agent_config,
            system_prompt,
            user_prompt,
            bb_context,
        )

        return result

    @staticmethod
    def _check_cache(
        cache_manager: CacheManager | None,
        image_bytes: bytes | None,
        pipeline_name: str,
        step_name: str,
        agent_config: AgentConfig,
        system_prompt: str,
        user_prompt: str,
        bb_context: dict[str, Any] | None,
    ) -> StepResult | None:
        """Check cache for a previous result. Returns StepResult on hit, None on miss."""
        if not cache_manager or not cache_manager.enabled:
            return None

        from doc2md.cache.keys import generate_cache_key, hash_image, hash_prompt

        image_hash = hash_image(image_bytes) if image_bytes else ""
        prompt_h = hash_prompt(system_prompt, user_prompt)

        key = generate_cache_key(
            image_hash=image_hash,
            pipeline_name=pipeline_name,
            step_name=step_name,
            agent_name=agent_config.name,
            agent_version=agent_config.version,
            model_id=agent_config.model.preferred,
            prompt_hash=prompt_h,
            blackboard_snapshot=bb_context,
        )

        entry = cache_manager.lookup(key)
        if entry is None:
            return None

        return StepResult(
            step_name=step_name,
            agent_name=agent_config.name,
            markdown=entry.markdown,
            token_usage=entry.token_usage,
            blackboard_writes=entry.blackboard_writes,
            model_used=entry.model_used,
            confidence=entry.confidence,
            cached=True,
        )

    @staticmethod
    def _store_in_cache(
        cache_manager: CacheManager | None,
        result: StepResult,
        image_bytes: bytes | None,
        pipeline_name: str,
        step_name: str,
        agent_config: AgentConfig,
        system_prompt: str,
        user_prompt: str,
        bb_context: dict[str, Any] | None,
    ) -> None:
        """Store a VLM result in the cache."""
        if not cache_manager or not cache_manager.enabled:
            return

        from doc2md.cache.keys import generate_cache_key, hash_image, hash_prompt
        from doc2md.cache.stats import CacheEntry

        image_hash = hash_image(image_bytes) if image_bytes else ""
        prompt_h = hash_prompt(system_prompt, user_prompt)

        key = generate_cache_key(
            image_hash=image_hash,
            pipeline_name=pipeline_name,
            step_name=step_name,
            agent_name=agent_config.name,
            agent_version=agent_config.version,
            model_id=agent_config.model.preferred,
            prompt_hash=prompt_h,
            blackboard_snapshot=bb_context,
        )

        entry = CacheEntry(
            key=key,
            pipeline_name=pipeline_name,
            step_name=step_name,
            agent_name=agent_config.name,
            agent_version=agent_config.version,
            markdown=result.markdown,
            blackboard_writes=result.blackboard_writes,
            confidence=result.confidence,
            token_usage=result.token_usage,
            model_used=result.model_used,
        )
        cache_manager.store(key, entry)

    @staticmethod
    def _resolve_image(input_mode: InputMode, image_bytes: bytes | None) -> str | None:
        """Return base64 image if the input mode requires an image."""
        needs_image = input_mode in (
            InputMode.IMAGE,
            InputMode.IMAGE_AND_PREVIOUS,
        )
        if needs_image and image_bytes:
            return image_to_base64(image_bytes)
        return None

    @staticmethod
    def _read_blackboard(
        agent_config: AgentConfig,
        blackboard: Blackboard | None,
        step_name: str,
    ) -> dict[str, Any] | None:
        """Build Jinja2 context from subscribed blackboard regions."""
        if not blackboard or not agent_config.blackboard.reads:
            return None
        return blackboard.to_jinja_context(agent_config.blackboard.reads)

    @staticmethod
    def _apply_blackboard_writes(
        blackboard: Blackboard,
        writes: dict[str, Any],
        writer: str,
    ) -> None:
        """Write VLM-elicited data to the blackboard."""
        for region, data in writes.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    blackboard.write(region, str(key), value, writer=writer)
            else:
                blackboard.write(region, region, data, writer=writer)

    @staticmethod
    def _run_code_writers(
        agent_config: AgentConfig,
        blackboard: Blackboard,
        markdown: str,
        step_name: str,
        page_num: int | None,
    ) -> None:
        """Execute code-computed blackboard writers declared in agent config."""
        for cw in agent_config.blackboard.code_writers:
            writer_entry = get_writer(cw.function)
            if not writer_entry:
                continue
            fn, output_key_template = writer_entry
            # Build kwargs based on declared input type
            kwargs: dict[str, Any] = {}
            if cw.input in ("markdown", "both"):
                kwargs["markdown"] = markdown
            if page_num is not None:
                kwargs["page_num"] = page_num

            value = fn(**kwargs)

            # Resolve output key (replace {page_num} placeholder)
            output_key = cw.output_key
            if page_num is not None:
                output_key = output_key.replace("{page_num}", str(page_num))
                output_key = output_key.replace("*", str(page_num))

            # Parse "region.key" from output_key
            parts = output_key.split(".", 1)
            if len(parts) == 2:
                blackboard.write(parts[0], parts[1], value, writer=step_name)
