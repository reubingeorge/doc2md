"""Cache subsystem — two-tier (memory + disk) with content-addressed keys."""

from doc2md.cache.keys import generate_cache_key, hash_image, hash_prompt
from doc2md.cache.manager import CacheManager
from doc2md.cache.stats import CacheEntry, CacheStats

__all__ = [
    "CacheManager",
    "CacheEntry",
    "CacheStats",
    "generate_cache_key",
    "hash_image",
    "hash_prompt",
]
