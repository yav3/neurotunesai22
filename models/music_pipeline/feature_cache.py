"""Feature caching for music processing pipelines.

Content-addressed storage, LRU eviction, batch retrieval, memoization.
Pure numpy.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    key: str
    value: np.ndarray
    content_hash: str
    access_count: int = 0
    byte_size: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_bytes: int = 0
    n_entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class FeatureCache:
    """LRU feature cache with content-addressed deduplication."""

    def __init__(self, max_entries: int = 1024, max_bytes: int = 256 * 1024 * 1024):
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._stats = CacheStats()
        self._total_bytes = 0

    @staticmethod
    def content_hash(data: np.ndarray) -> str:
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]

    def get(self, key: str) -> np.ndarray | None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key].access_count += 1
            self._stats.hits += 1
            return self._store[key].value
        self._stats.misses += 1
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        byte_size = value.nbytes
        if key in self._store:
            old = self._store.pop(key)
            self._total_bytes -= old.byte_size

        while (len(self._store) >= self._max_entries
               or self._total_bytes + byte_size > self._max_bytes) and self._store:
            _, evicted = self._store.popitem(last=False)
            self._total_bytes -= evicted.byte_size
            self._stats.evictions += 1

        entry = CacheEntry(
            key=key,
            value=value,
            content_hash=self.content_hash(value),
            byte_size=byte_size,
        )
        self._store[key] = entry
        self._total_bytes += byte_size
        self._stats.n_entries = len(self._store)
        self._stats.total_bytes = self._total_bytes

    def batch_get(self, keys: list[str]) -> dict[str, np.ndarray | None]:
        return {k: self.get(k) for k in keys}

    def invalidate(self, key: str) -> bool:
        if key in self._store:
            entry = self._store.pop(key)
            self._total_bytes -= entry.byte_size
            self._stats.n_entries = len(self._store)
            self._stats.total_bytes = self._total_bytes
            return True
        return False

    def invalidate_by_hash(self, content_hash: str) -> int:
        to_remove = [k for k, v in self._store.items() if v.content_hash == content_hash]
        for k in to_remove:
            self.invalidate(k)
        return len(to_remove)

    def clear(self) -> None:
        self._store.clear()
        self._total_bytes = 0
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats


def make_cache_key(track_id: str, feature_name: str, params: dict | None = None) -> str:
    """Deterministic cache key from track ID, feature name, and params."""
    parts = [track_id, feature_name]
    if params:
        for k in sorted(params):
            parts.append(f"{k}={params[k]}")
    return "|".join(parts)


def memoize_feature(cache: FeatureCache, key: str):
    """Decorator-style memoization using a FeatureCache instance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cached = cache.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                cache.put(key, result)
            return result
        return wrapper
    return decorator
