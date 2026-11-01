"""Track embedding space for similarity search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackEmbedding:
    track_id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingSpace:
    def __init__(self, dim: int | None = None):
        self._embeddings: dict[str, TrackEmbedding] = {}
        self._dim = dim
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []
        self._dirty = True

    def add(self, emb: TrackEmbedding) -> None:
        if self._dim is None:
            self._dim = len(emb.vector)
        if len(emb.vector) != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {len(emb.vector)}")
        self._embeddings[emb.track_id] = emb
        self._dirty = True

    def add_many(self, embeddings: list[TrackEmbedding]) -> int:
        for e in embeddings:
            self.add(e)
        return len(embeddings)

    def get(self, track_id: str) -> TrackEmbedding | None:
        return self._embeddings.get(track_id)

    def __len__(self) -> int:
        return len(self._embeddings)

    @property
    def dim(self) -> int:
        return self._dim or 0

    def _rebuild_matrix(self) -> None:
        if not self._dirty:
            return
        self._ids = list(self._embeddings.keys())
        if self._ids:
            self._matrix = np.vstack([self._embeddings[i].vector for i in self._ids])
        else:
            self._matrix = np.empty((0, self._dim or 0))
        self._dirty = False

    def nearest(
        self,
        query: np.ndarray,
        n: int = 5,
        exclude: set[str] | None = None,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        self._rebuild_matrix()
        if self._matrix is None or len(self._matrix) == 0:
            return []

        if metric == "cosine":
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            q_norm = np.linalg.norm(query)
            q_norm = q_norm if q_norm > 0 else 1.0
            sims = (self._matrix / norms) @ (query / q_norm)
            dists = 1.0 - sims
        elif metric == "euclidean":
            dists = np.linalg.norm(self._matrix - query, axis=1)
        else:
            dists = np.sum(np.abs(self._matrix - query), axis=1)

        exclude = exclude or set()
        results = []
        order = np.argsort(dists)
        for idx in order:
            tid = self._ids[idx]
            if tid in exclude:
                continue
            results.append((tid, float(dists[idx])))
            if len(results) >= n:
                break
        return results

    def reduce_dim(self, target_dim: int = 2, seed: int | None = None) -> np.ndarray:
        self._rebuild_matrix()
        if self._matrix is None or len(self._matrix) == 0:
            return np.empty((0, target_dim))
        mat = self._matrix - self._matrix.mean(axis=0)
        rng = np.random.default_rng(seed)
        if mat.shape[0] < target_dim or mat.shape[1] < target_dim:
            return mat[:, :target_dim] if mat.shape[1] >= target_dim else np.zeros((mat.shape[0], target_dim))
        try:
            _, s, vt = np.linalg.svd(mat, full_matrices=False)
            return (mat @ vt[:target_dim].T)
        except np.linalg.LinAlgError:
            return np.zeros((mat.shape[0], target_dim))

    def all_ids(self) -> list[str]:
        return list(self._embeddings.keys())

    def to_matrix(self) -> tuple[np.ndarray, list[str]]:
        self._rebuild_matrix()
        return self._matrix.copy() if self._matrix is not None else np.empty((0, 0)), list(self._ids)
