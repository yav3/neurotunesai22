"""Semantic embedding space for music concepts.

Maps musical items into a continuous semantic space defined by perceptual
dimensions (energy, valence, complexity, etc.).  Supports nearest-neighbour
retrieval, analogy completion (a:b :: c:?), spherical interpolation,
trajectory generation via cubic spline, latent semantic analysis on
tag-document matrices, and cross-modal Procrustes alignment.

Pure numpy / scipy — no ML framework dependencies.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class SemanticDimension(Enum):
    ENERGY = "energy"
    VALENCE = "valence"
    COMPLEXITY = "complexity"
    NOVELTY = "novelty"
    FAMILIARITY = "familiarity"
    GROOVE = "groove"
    TENSION = "tension"
    BRIGHTNESS = "brightness"


_DIM_ORDER: List[SemanticDimension] = list(SemanticDimension)
_DIM_INDEX: Dict[SemanticDimension, int] = {d: i for i, d in enumerate(_DIM_ORDER)}
N_DIMS = len(_DIM_ORDER)


@dataclass
class SemanticVector:
    label: str
    dimension_scores: Dict[SemanticDimension, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        return np.array([self.dimension_scores.get(d, 0.0) for d in _DIM_ORDER])


def semantic_distance(
    a: SemanticVector,
    b: SemanticVector,
    weights: Optional[Dict[SemanticDimension, float]] = None,
) -> float:
    va, vb = a.to_array(), b.to_array()
    if weights is None:
        return float(np.linalg.norm(va - vb))
    w = np.array([weights.get(d, 1.0) for d in _DIM_ORDER])
    return float(np.sqrt(np.sum(w * (va - vb) ** 2)))


class SemanticSpace:
    def __init__(self) -> None:
        self._vectors: List[SemanticVector] = []
        self._matrix: Optional[np.ndarray] = None

    def register(self, vec: SemanticVector) -> int:
        idx = len(self._vectors)
        self._vectors.append(vec)
        self._matrix = None
        return idx

    def register_many(self, vecs: Sequence[SemanticVector]) -> List[int]:
        return [self.register(v) for v in vecs]

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None or self._matrix.shape[0] != len(self._vectors):
            self._matrix = np.array([v.to_array() for v in self._vectors])
        return self._matrix

    def __len__(self) -> int:
        return len(self._vectors)

    def nearest(
        self, query: SemanticVector, k: int = 5, exclude_self: bool = True,
    ) -> List[Tuple[int, float]]:
        qv = query.to_array()
        dists = np.linalg.norm(self.matrix - qv, axis=1)
        order = np.argsort(dists)
        results: List[Tuple[int, float]] = []
        for idx in order:
            if exclude_self and self._vectors[idx] is query:
                continue
            results.append((int(idx), float(dists[idx])))
            if len(results) >= k:
                break
        return results

    def centroid(self, indices: Optional[List[int]] = None) -> np.ndarray:
        if indices is None:
            return self.matrix.mean(axis=0)
        return self.matrix[indices].mean(axis=0)

    def cluster(self, k: int, max_iter: int = 100, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        n = len(self._vectors)
        if n == 0 or k <= 0:
            return np.array([], dtype=int)
        k = min(k, n)
        centres = self.matrix[rng.choice(n, k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(self.matrix[:, None, :] - centres[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = self.matrix[labels == c]
                if len(members) > 0:
                    centres[c] = members.mean(axis=0)
        return labels


def tag_cooccurrence_matrix(
    vectors: Sequence[SemanticVector],
) -> Tuple[np.ndarray, List[str]]:
    all_tags = sorted({t for v in vectors for t in v.tags})
    tag_idx = {t: i for i, t in enumerate(all_tags)}
    n = len(all_tags)
    cooc = np.zeros((n, n))
    for v in vectors:
        idxs = [tag_idx[t] for t in v.tags]
        for i in idxs:
            for j in idxs:
                cooc[i, j] += 1.0
    return cooc, all_tags


def latent_semantic_analysis(
    vectors: Sequence[SemanticVector], n_components: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    all_tags = sorted({t for v in vectors for t in v.tags})
    tag_idx = {t: i for i, t in enumerate(all_tags)}
    m = len(vectors)
    n = len(all_tags)
    td = np.zeros((n, m))
    for j, v in enumerate(vectors):
        for t in v.tags:
            td[tag_idx[t], j] += 1.0
    k = min(n_components, min(n, m))
    U, s, Vt = np.linalg.svd(td, full_matrices=False)
    U_k = U[:, :k] * s[:k]
    V_k = Vt[:k, :].T * s[:k]
    return U_k, V_k, all_tags


def concept_graph(
    vectors: Sequence[SemanticVector], threshold: float = 0.5,
) -> np.ndarray:
    n = len(vectors)
    mat = np.zeros((n, n))
    arrs = np.array([v.to_array() for v in vectors])
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(arrs[i] - arrs[j]))
            sim = np.exp(-d * d / 2.0)
            if sim >= threshold:
                mat[i, j] = mat[j, i] = sim
    return mat


def analogy_completion(
    a: SemanticVector, b: SemanticVector, c: SemanticVector,
    space: SemanticSpace,
) -> Tuple[int, float]:
    va, vb, vc = a.to_array(), b.to_array(), c.to_array()
    target = vc + (vb - va)
    dists = np.linalg.norm(space.matrix - target, axis=1)
    best = int(np.argmin(dists))
    return best, float(dists[best])


def interpolate_semantic(
    a: SemanticVector, b: SemanticVector, t: float, spherical: bool = False,
) -> np.ndarray:
    va, vb = a.to_array(), b.to_array()
    if not spherical:
        return (1.0 - t) * va + t * vb
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return (1.0 - t) * va + t * vb
    ua, ub = va / na, vb / nb
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    omega = np.arccos(dot)
    if abs(omega) < 1e-8:
        return (1.0 - t) * va + t * vb
    sa = np.sin((1.0 - t) * omega) / np.sin(omega)
    sb = np.sin(t * omega) / np.sin(omega)
    r = (1.0 - t) * na + t * nb
    return r * (sa * ua + sb * ub)


def semantic_trajectory(
    waypoints: Sequence[SemanticVector], n_points: int = 50,
) -> np.ndarray:
    from scipy.interpolate import CubicSpline

    m = len(waypoints)
    if m < 2:
        if m == 1:
            return np.tile(waypoints[0].to_array(), (n_points, 1))
        return np.zeros((n_points, N_DIMS))
    arrs = np.array([w.to_array() for w in waypoints])
    t_knots = np.linspace(0.0, 1.0, m)
    t_eval = np.linspace(0.0, 1.0, n_points)
    result = np.zeros((n_points, N_DIMS))
    for d in range(N_DIMS):
        cs = CubicSpline(t_knots, arrs[:, d])
        result[:, d] = cs(t_eval)
    return result


def cross_modal_alignment(
    source: np.ndarray, target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    from scipy.linalg import orthogonal_procrustes

    s_mean = source.mean(axis=0)
    t_mean = target.mean(axis=0)
    s_c = source - s_mean
    t_c = target - t_mean
    R, scale = orthogonal_procrustes(s_c, t_c)
    aligned = s_c @ R + t_mean
    residual = float(np.sqrt(np.sum((aligned - target) ** 2) / max(len(target), 1)))
    return R, residual
