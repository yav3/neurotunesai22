"""Track embedding space for nearest-neighbor music search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.music_recommendation.similarity import cosine_similarity, euclidean_distance


EMBEDDING_FEATURES = [
    "bpm", "valence", "arousal", "dominance", "energy",
    "danceability", "acousticness", "instrumentalness",
    "spectral_centroid", "spectral_rolloff", "brightness",
    "bass_energy", "mid_energy", "treble_energy",
    "rms_energy", "dynamic_range",
]


@dataclass
class TrackEmbedding:
    track_id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def build_embedding(
    track_features: dict[str, float],
    feature_keys: list[str] | None = None,
) -> np.ndarray:
    keys = feature_keys or EMBEDDING_FEATURES
    vec = np.array([track_features.get(k, 0.0) for k in keys], dtype=float)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec


def pca_reduce(embeddings: np.ndarray, n_dims: int) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return embeddings
    n_dims = min(n_dims, embeddings.shape[1], embeddings.shape[0])
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        return centered[:, :n_dims]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_dims]
    return centered @ eigvecs[:, idx]


class EmbeddingSpace:
    def __init__(self, metric: str = "cosine"):
        self._embeddings: dict[str, TrackEmbedding] = {}
        self._metric = metric

    def add(self, emb: TrackEmbedding) -> None:
        self._embeddings[emb.track_id] = emb

    def remove(self, track_id: str) -> bool:
        return self._embeddings.pop(track_id, None) is not None

    def __len__(self) -> int:
        return len(self._embeddings)

    def get(self, track_id: str) -> TrackEmbedding | None:
        return self._embeddings.get(track_id)

    def all_ids(self) -> list[str]:
        return list(self._embeddings.keys())

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self._metric == "cosine":
            return 1.0 - cosine_similarity(a, b)
        return euclidean_distance(a, b)

    def query(
        self,
        vector: np.ndarray,
        n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        exclude = exclude or set()
        scored = []
        for tid, emb in self._embeddings.items():
            if tid in exclude:
                continue
            dist = self._distance(vector, emb.vector)
            scored.append((tid, dist))
        scored.sort(key=lambda x: x[1])
        return scored[:n]

    def nearest(
        self,
        track_id: str,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        emb = self._embeddings.get(track_id)
        if emb is None:
            return []
        return self.query(emb.vector, n=n, exclude={track_id})

    def to_matrix(self) -> tuple[np.ndarray, list[str]]:
        ids = list(self._embeddings.keys())
        if not ids:
            return np.empty((0, 0)), []
        vecs = np.array([self._embeddings[tid].vector for tid in ids])
        return vecs, ids
