"""Locality-sensitive hashing for approximate nearest-neighbor fingerprint search."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class MinHashFingerprint:
    """MinHash for estimating Jaccard similarity between feature sets.

    Represents a set of integer features as a compact signature via
    universal hash functions h_i(x) = (a_i * x + b_i) mod p.
    """

    def __init__(self, n_hashes: int = 128, seed: int = 42) -> None:
        self.n_hashes = n_hashes
        rng = np.random.RandomState(seed)
        self._p = 2**31 - 1
        self._a = rng.randint(1, self._p, size=n_hashes).astype(np.int64)
        self._b = rng.randint(0, self._p, size=n_hashes).astype(np.int64)

    def signature(self, feature_set: set[int] | list[int]) -> np.ndarray:
        if not feature_set:
            return np.full(self.n_hashes, self._p, dtype=np.int64)
        sig = np.full(self.n_hashes, self._p, dtype=np.int64)
        for x in feature_set:
            hvals = (self._a * int(x) + self._b) % self._p
            sig = np.minimum(sig, hvals)
        return sig

    def jaccard_estimate(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))

    @staticmethod
    def jaccard_exact(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


class LSHIndex:
    """Locality-sensitive hashing index using random hyperplane projection.

    For cosine similarity: project vectors onto random hyperplanes,
    hash by sign pattern, use banding for approximate threshold search.
    """

    def __init__(
        self,
        dim: int,
        n_tables: int = 10,
        n_bits: int = 8,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits
        rng = np.random.RandomState(seed)
        self._planes = [rng.randn(n_bits, dim) for _ in range(n_tables)]
        self._tables: list[dict[int, list[str]]] = [{} for _ in range(n_tables)]
        self._vectors: dict[str, np.ndarray] = {}

    def _hash_vector(self, vec: np.ndarray, table_idx: int) -> int:
        projections = self._planes[table_idx] @ vec
        bits = (projections > 0).astype(int)
        h = 0
        for i, b in enumerate(bits):
            h |= (int(b) << i)
        return h

    def insert(self, item_id: str, vector: np.ndarray) -> None:
        vec = vector.ravel()
        if len(vec) != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {len(vec)}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._vectors[item_id] = vec
        for i in range(self.n_tables):
            h = self._hash_vector(vec, i)
            self._tables[i].setdefault(h, []).append(item_id)

    def query(
        self,
        vector: np.ndarray,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        vec = vector.ravel()
        if len(vec) != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {len(vec)}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        candidates: set[str] = set()
        for i in range(self.n_tables):
            h = self._hash_vector(vec, i)
            if h in self._tables[i]:
                candidates.update(self._tables[i][h])

        results = []
        for cid in candidates:
            sim = float(np.dot(vec, self._vectors[cid]))
            if sim >= threshold:
                results.append((cid, sim))

        return sorted(results, key=lambda x: -x[1])

    @property
    def n_items(self) -> int:
        return len(self._vectors)
