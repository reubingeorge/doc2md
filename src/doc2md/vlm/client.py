"""Async VLM client wrapping OpenAI's API with retry and fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from doc2md.errors.exceptions import TerminalError
from doc2md.errors.fallback import FallbackChain
from doc2md.types import RetryConfig, TokenUsage, VLMResponse

if TYPE_CHECKING:
    from doc2md.concurrency.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Transient HTTP status codes that warrant retry
_TRANSIENT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


class AsyncVLMClient:
    """Sends requests to an OpenAI-compatible vision-language model."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._rate_limiter = rate_limiter

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def send_request(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_b64: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> VLMResponse:
        """Send a single VLM request and return parsed response."""
        # Rate limit if available
        if self._rate_limiter:
            await self._rate_limiter.acquire(estimated_tokens=max_tokens)

        messages = self._build_messages(system_prompt, user_prompt, image_b64)

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5

        response = await self._client.chat.completions.create(**kwargs)
        result = self._parse_response(response)

        # Record actual usage
        if self._rate_limiter:
            self._rate_limiter.record_usage(
                result.token_usage.prompt_tokens,
                result.token_usage.completion_tokens,
            )

        return result

    async def send_request_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_chain: FallbackChain,
        image_b64: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> VLMResponse:
        """Send request with automatic model fallback on terminal errors."""
        while True:
            model = fallback_chain.current_model
            try:
                return await self.send_request(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_b64=image_b64,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                )
            except openai.NotFoundError:
                logger.warning("Model '%s' not found, trying fallback", model)
                try:
                    fallback_chain.next_model()
                except TerminalError:
                    raise
            except openai.AuthenticationError:
                raise

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _build_messages(
        system_prompt: str, user_prompt: str, image_b64: str | None
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        if image_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        return messages

    @staticmethod
    def _parse_response(response: openai.types.chat.ChatCompletion) -> VLMResponse:
        choice = response.choices[0]
        usage = response.usage

        token_logprobs = None
        if choice.logprobs and choice.logprobs.content:
            token_logprobs = [
                {"token": lp.token, "logprob": lp.logprob}
                for lp in choice.logprobs.content
            ]

        return VLMResponse(
            content=choice.message.content or "",
            model=response.model,
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            logprobs=token_logprobs,
            finish_reason=choice.finish_reason,
        )
