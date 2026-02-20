"""Cache entry and statistics models."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from doc2md.types import TokenUsage


class CacheEntry(BaseModel):
    """A cached step result."""

    key: str
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float = 7 * 24 * 3600  # 7 days default
    pipeline_name: str = ""
    step_name: str = ""
    agent_name: str = ""
    agent_version: str = ""
    markdown: str = ""
    blackboard_writes: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model_used: str = ""

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_seconds

    @property
    def size_bytes(self) -> int:
        return len(self.markdown.encode("utf-8")) + len(str(self.blackboard_writes).encode("utf-8"))


class CacheStats(BaseModel):
    """Aggregate cache statistics."""

    entries: int = 0
    size_mb: float = 0.0
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
