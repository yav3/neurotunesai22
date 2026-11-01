"""Session-based music recommendation.

Sequence-aware approaches that model the current listening session
rather than long-term user profiles.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SessionState:
    """Running state of a listening session."""

    track_ids: list[int] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def add(self, track_id: int, feature_vec: np.ndarray, timestamp: float = 0.0) -> None:
        self.track_ids.append(track_id)
        self.features.append(np.asarray(feature_vec, dtype=float))
        self.timestamps.append(timestamp)

    def __len__(self) -> int:
        return len(self.track_ids)

    @property
    def feature_matrix(self) -> np.ndarray:
        if not self.features:
            return np.empty((0, 0))
        return np.vstack(self.features)


def session_centroid(session: SessionState) -> np.ndarray:
    """Mean feature vector over all tracks in the session."""
    if len(session) == 0:
        return np.array([])
    return session.feature_matrix.mean(axis=0)


def exponential_decay_centroid(
    session: SessionState, half_life: float = 3.0
) -> np.ndarray:
    """Recency-weighted centroid: more recent tracks weighted higher.

    half_life is measured in number of tracks (not time).
    """
    n = len(session)
    if n == 0:
        return np.array([])
    positions = np.arange(n, dtype=float)
    weights = np.exp(-np.log(2) * (n - 1 - positions) / max(half_life, 1e-6))
    weights /= weights.sum()
    return (session.feature_matrix.T @ weights).ravel()


def markov_transition_matrix(
    session: SessionState, n_items: int
) -> np.ndarray:
    """First-order Markov transition matrix from session sequences.

    T[i,j] = P(next = j | current = i), estimated from observed transitions.
    """
    T = np.zeros((n_items, n_items))
    ids = session.track_ids
    for a, b in zip(ids[:-1], ids[1:]):
        if a < n_items and b < n_items:
            T[a, b] += 1.0
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    return T / row_sums


def markov_predict(
    transition: np.ndarray, current_item: int, n: int = 10
) -> list[tuple[int, float]]:
    """Predict next items from a Markov transition matrix."""
    if current_item >= transition.shape[0]:
        return []
    probs = transition[current_item]
    top_idx = np.argsort(probs)[::-1][:n]
    return [(int(i), float(probs[i])) for i in top_idx if probs[i] > 0]


@dataclass
class SequentialPattern:
    """A frequently occurring item subsequence."""

    items: tuple[int, ...]
    support: int
    confidence: float


def find_sequential_patterns(
    sessions: list[list[int]], min_support: int = 2, max_len: int = 3
) -> list[SequentialPattern]:
    """Mine frequent sequential patterns from session histories.

    Counts contiguous subsequences of length 2..max_len.
    """
    from collections import Counter

    patterns: list[SequentialPattern] = []
    for length in range(2, max_len + 1):
        counter: Counter[tuple[int, ...]] = Counter()
        for session in sessions:
            for i in range(len(session) - length + 1):
                subseq = tuple(session[i : i + length])
                counter[subseq] += 1
        for subseq, count in counter.items():
            if count >= min_support:
                prefix_count = sum(
                    1 for s in sessions
                    for i in range(len(s) - length + 2)
                    if tuple(s[i : i + length - 1]) == subseq[:-1]
                )
                conf = count / max(prefix_count, 1)
                patterns.append(SequentialPattern(subseq, count, float(conf)))
    patterns.sort(key=lambda p: (-p.support, -p.confidence))
    return patterns


def session_knn_recommend(
    current_session: SessionState,
    historical_sessions: list[SessionState],
    candidate_features: np.ndarray,
    k: int = 5,
    n: int = 10,
) -> list[tuple[int, float]]:
    """k-NN over session centroids → aggregate neighbors' next tracks."""
    if len(current_session) == 0 or not historical_sessions:
        return []
    query = session_centroid(current_session)
    centroids = np.vstack([session_centroid(s) for s in historical_sessions if len(s) > 0])
    if len(centroids) == 0:
        return []

    norms_c = np.linalg.norm(centroids, axis=1)
    norms_c[norms_c < 1e-12] = 1.0
    norm_q = max(np.linalg.norm(query), 1e-12)
    sims = centroids @ query / (norms_c * norm_q)

    top_k = np.argsort(sims)[::-1][:k]
    known = set(current_session.track_ids)

    item_scores: dict[int, float] = {}
    for idx in top_k:
        w = max(sims[idx], 0.0)
        s = historical_sessions[idx]
        for tid in s.track_ids:
            if tid not in known:
                item_scores[tid] = item_scores.get(tid, 0.0) + w

    ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n]


@dataclass
class SessionMomentumTracker:
    """Track feature-space momentum to predict session direction."""

    window: int = 3

    def momentum(self, session: SessionState) -> np.ndarray:
        if len(session) < 2:
            return np.zeros(session.features[0].shape if session.features else (0,))
        mat = session.feature_matrix
        w = min(self.window, len(session))
        recent = mat[-w:]
        deltas = np.diff(recent, axis=0)
        return deltas.mean(axis=0)

    def predict_next_feature(self, session: SessionState) -> np.ndarray:
        if len(session) == 0:
            return np.array([])
        current = session.features[-1]
        m = self.momentum(session)
        return current + m

    def recommend(
        self,
        session: SessionState,
        candidate_features: np.ndarray,
        candidate_ids: np.ndarray | None = None,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        target = self.predict_next_feature(session)
        if len(target) == 0 or len(candidate_features) == 0:
            return []
        dists = np.linalg.norm(candidate_features - target, axis=1)
        top_idx = np.argsort(dists)[:n]
        ids = candidate_ids if candidate_ids is not None else np.arange(len(candidate_features))
        return [(int(ids[i]), float(1.0 / (1.0 + dists[i]))) for i in top_idx]


def session_diversity_score(session: SessionState) -> float:
    """Measure feature-space diversity of a session (mean pairwise distance)."""
    if len(session) < 2:
        return 0.0
    mat = session.feature_matrix
    n = len(mat)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(mat[i] - mat[j]))
            count += 1
    return total / max(count, 1)


def session_coherence_score(session: SessionState) -> float:
    """Measure feature-space coherence (mean cosine similarity of consecutive pairs)."""
    if len(session) < 2:
        return 1.0
    mat = session.feature_matrix
    sims = []
    for i in range(len(mat) - 1):
        a, b = mat[i], mat[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(a, b) / (na * nb)))
    return float(np.mean(sims))
