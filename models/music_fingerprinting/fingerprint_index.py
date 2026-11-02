"""Fingerprint index: LSH, MinHash, inverted index, top-k retrieval."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

import numpy as np


@dataclass
class IndexEntry:
    track_id: str
    offset_ms: float
    hash_value: int


@dataclass
class SearchResult:
    track_id: str
    score: float
    offset_ms: float
    match_count: int


@dataclass
class LSHConfig:
    n_tables: int = 20
    n_bits: int = 8
    seed: int = 42


@dataclass
class MinHashConfig:
    n_permutations: int = 128
    seed: int = 42


class InvertedIndex:
    def __init__(self) -> None:
        self._index: Dict[int, List[IndexEntry]] = {}
        self._track_count: int = 0

    def insert(self, hash_val: int, track_id: str, offset_ms: float) -> None:
        entry = IndexEntry(track_id, offset_ms, hash_val)
        self._index.setdefault(hash_val, []).append(entry)

    def insert_batch(self, hashes: np.ndarray, track_id: str,
                     offsets_ms: np.ndarray) -> None:
        for h, o in zip(hashes.flat, offsets_ms.flat):
            self.insert(int(h), track_id, float(o))
        self._track_count += 1

    def query(self, hash_val: int) -> List[IndexEntry]:
        return self._index.get(hash_val, [])

    def query_batch(self, hashes: np.ndarray) -> Dict[str, List[float]]:
        hits: Dict[str, List[float]] = {}
        for h in hashes.flat:
            for entry in self.query(int(h)):
                hits.setdefault(entry.track_id, []).append(entry.offset_ms)
        return hits

    @property
    def size(self) -> int:
        return sum(len(v) for v in self._index.values())

    @property
    def n_buckets(self) -> int:
        return len(self._index)


def _random_hyperplanes(n_dims: int, n_bits: int,
                        rng: np.random.RandomState) -> np.ndarray:
    return rng.randn(n_bits, n_dims)


def lsh_hash(vector: np.ndarray, hyperplanes: np.ndarray) -> int:
    projections = hyperplanes @ vector
    bits = (projections > 0).astype(int)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


class LSHIndex:
    def __init__(self, n_dims: int, config: LSHConfig | None = None) -> None:
        cfg = config or LSHConfig()
        self.n_dims = n_dims
        self.config = cfg
        rng = np.random.RandomState(cfg.seed)
        self._tables: List[Dict[int, List[Tuple[str, np.ndarray]]]] = []
        self._hyperplanes: List[np.ndarray] = []
        for _ in range(cfg.n_tables):
            hp = _random_hyperplanes(n_dims, cfg.n_bits, rng)
            self._hyperplanes.append(hp)
            self._tables.append({})

    def insert(self, key: str, vector: np.ndarray) -> None:
        for hp, table in zip(self._hyperplanes, self._tables):
            h = lsh_hash(vector, hp)
            table.setdefault(h, []).append((key, vector))

    def query(self, vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        candidates: Dict[str, np.ndarray] = {}
        for hp, table in zip(self._hyperplanes, self._tables):
            h = lsh_hash(vector, hp)
            for key, vec in table.get(h, []):
                if key not in candidates:
                    candidates[key] = vec
        scored: List[Tuple[str, float]] = []
        for key, vec in candidates.items():
            dist = float(np.linalg.norm(vector - vec))
            scored.append((key, dist))
        scored.sort(key=lambda x: x[1])
        return scored[:k]


def minhash_signature(item_set: Set[int], n_perms: int = 128,
                      seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    large_prime = 2**31 - 1
    a = rng.randint(1, large_prime, size=n_perms)
    b = rng.randint(0, large_prime, size=n_perms)
    sig = np.full(n_perms, np.iinfo(np.int64).max, dtype=np.int64)
    for item in item_set:
        hashes = (a * item + b) % large_prime
        sig = np.minimum(sig, hashes)
    return sig


def minhash_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def multi_probe_lsh(vector: np.ndarray, hyperplanes: np.ndarray,
                    n_probes: int = 3) -> List[int]:
    projections = hyperplanes @ vector
    base_bits = (projections > 0).astype(int)
    base_hash = 0
    for b in base_bits:
        base_hash = (base_hash << 1) | int(b)
    probes = [base_hash]
    margins = np.abs(projections)
    flip_order = np.argsort(margins)
    for i in range(min(n_probes - 1, len(flip_order))):
        flipped = base_bits.copy()
        flipped[flip_order[i]] = 1 - flipped[flip_order[i]]
        h = 0
        for b in flipped:
            h = (h << 1) | int(b)
        probes.append(h)
    return probes


def top_k_by_offset_histogram(hits: Dict[str, List[float]],
                              k: int = 5,
                              tolerance_ms: float = 100.0) -> List[SearchResult]:
    results: List[SearchResult] = []
    for track_id, offsets in hits.items():
        if not offsets:
            continue
        arr = np.sort(offsets)
        best_count = 0
        best_offset = arr[0]
        for ref in arr:
            count = int(np.sum(np.abs(arr - ref) < tolerance_ms))
            if count > best_count:
                best_count = count
                best_offset = ref
        results.append(SearchResult(
            track_id=track_id,
            score=float(best_count),
            offset_ms=float(best_offset),
            match_count=best_count,
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bit_sample_hash(spectrum: np.ndarray, n_bits: int = 32) -> int:
    if len(spectrum) < n_bits:
        spectrum = np.pad(spectrum, (0, n_bits - len(spectrum)))
    indices = np.linspace(0, len(spectrum) - 1, n_bits).astype(int)
    median_val = np.median(spectrum)
    bits = (spectrum[indices] > median_val).astype(int)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h
