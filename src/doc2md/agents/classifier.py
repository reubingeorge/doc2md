"""Auto-classification — classify a document and select the best pipeline."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from doc2md.utils.image import image_to_base64

if TYPE_CHECKING:
    from doc2md.agents.registry import PipelineRegistry
    from doc2md.blackboard.board import Blackboard
    from doc2md.vlm.client import AsyncVLMClient

logger = logging.getLogger(__name__)

_CLASSIFIER_MODEL = "gpt-4.1-nano"
_CLASSIFIER_MAX_TOKENS = 256
_CLASSIFIER_TEMPERATURE = 0.0
_FALLBACK_PIPELINE = "generic"
_CONFIDENCE_THRESHOLD = 0.7


class ClassificationResult(BaseModel):
    pipeline_name: str
    confidence: float
    reasoning: str = ""
    content_types_detected: list[str] = Field(default_factory=list)


async def classify_document(
    page1_image: bytes,
    pipeline_registry: PipelineRegistry,
    vlm_client: AsyncVLMClient,
    blackboard: Blackboard | None = None,
) -> ClassificationResult:
    """Classify a document using page 1 and select the best pipeline.

    Uses a lightweight VLM call with a dynamic prompt built from the
    pipeline registry descriptions.
    """
    system_prompt = _build_classification_prompt(pipeline_registry)
    image_b64 = image_to_base64(page1_image)

    try:
        response = await vlm_client.send_request(
            model=_CLASSIFIER_MODEL,
            system_prompt=system_prompt,
            user_prompt="Classify this document. Respond with JSON only.",
            image_b64=image_b64,
            max_tokens=_CLASSIFIER_MAX_TOKENS,
            temperature=_CLASSIFIER_TEMPERATURE,
        )
        result = _parse_classification(response.content, pipeline_registry)
    except Exception as e:
        logger.warning("Classification failed: %s. Falling back to '%s'.", e, _FALLBACK_PIPELINE)
        result = ClassificationResult(
            pipeline_name=_FALLBACK_PIPELINE,
            confidence=0.0,
            reasoning=f"Classification failed: {e}",
        )

    # Fallback on low confidence
    if result.confidence < _CONFIDENCE_THRESHOLD:
        logger.info(
            "Low classification confidence (%.2f). Using '%s'.",
            result.confidence, _FALLBACK_PIPELINE,
        )
        result.pipeline_name = _FALLBACK_PIPELINE

    # Write content types to blackboard
    if blackboard and result.content_types_detected:
        blackboard.write(
            "document_metadata", "content_types",
            result.content_types_detected, writer="_classifier",
        )

    return result


def _build_classification_prompt(pipeline_registry: PipelineRegistry) -> str:
    """Build the system prompt dynamically from available pipelines."""
    pipelines_desc = []
    for info in pipeline_registry.list_pipelines():
        pipelines_desc.append(f'- "{info.name}": {info.description}')

    pipeline_list = "\n".join(pipelines_desc) if pipelines_desc else '- "generic": General-purpose extraction'

    return f"""You are a document classifier. Given the first page of a document, classify it into the best matching pipeline type.

Available pipelines:
{pipeline_list}

Respond with a JSON object only (no markdown fences):
{{
  "pipeline_name": "<name>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>",
  "content_types_detected": ["<type1>", "<type2>"]
}}

Content types include: prose, tables, handwriting, forms, signatures, headers, footers, images, equations.
Be conservative with confidence — only use > 0.8 when very certain."""


def _parse_classification(
    raw_text: str,
    pipeline_registry: PipelineRegistry,
) -> ClassificationResult:
    """Parse the VLM classification response."""
    text = raw_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse classification JSON: %s", text[:200])
        return ClassificationResult(
            pipeline_name=_FALLBACK_PIPELINE,
            confidence=0.0,
            reasoning="Failed to parse classification response",
        )

    pipeline_name = data.get("pipeline_name", _FALLBACK_PIPELINE)

    # Validate pipeline exists
    if not pipeline_registry.has(pipeline_name):
        logger.warning("Classified as unknown pipeline '%s', falling back", pipeline_name)
        pipeline_name = _FALLBACK_PIPELINE

    return ClassificationResult(
        pipeline_name=pipeline_name,
        confidence=float(data.get("confidence", 0.0)),
        reasoning=data.get("reasoning", ""),
        content_types_detected=data.get("content_types_detected", []),
    )
