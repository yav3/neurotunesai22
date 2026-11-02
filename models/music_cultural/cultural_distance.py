"""Cultural distance metrics between musical traditions.

Quantifies how far apart two musical cultures are based on scale systems,
rhythmic organization, ornamentation vocabulary, tuning preferences, and
formal structures.  Uses information-theoretic divergence (KL, JS) on
distributional features and graph-distance on a tradition affinity network.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Sequence
from enum import Enum


class CulturalDimension(Enum):
    SCALE = "scale"
    RHYTHM = "rhythm"
    TUNING = "tuning"
    ORNAMENTATION = "ornamentation"
    FORM = "form"
    TEXTURE = "texture"
    TIMBRE = "timbre"


@dataclass
class TraditionProfile:
    name: str
    scale_histogram: np.ndarray        # 12-bin pitch-class distribution
    rhythm_histogram: np.ndarray       # 16-bin onset quantisation grid
    tuning_deviations: np.ndarray      # cents deviation per scale degree
    ornament_density: float            # ornaments per beat
    predominant_meter: Tuple[int, int] # e.g. (7, 8)
    form_segments: int                 # typical number of formal sections
    texture_polyphony: float           # 0 (monophonic) – 1 (dense polyphony)
    timbral_brightness: float          # spectral centroid normalised 0–1


@dataclass
class CulturalDistanceResult:
    total: float
    per_dimension: Dict[CulturalDimension, float]
    dominant_dimension: CulturalDimension


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def scale_distance(a: TraditionProfile, b: TraditionProfile) -> float:
    return _js_divergence(a.scale_histogram, b.scale_histogram)


def rhythm_distance(a: TraditionProfile, b: TraditionProfile) -> float:
    return _js_divergence(a.rhythm_histogram, b.rhythm_histogram)


def tuning_distance(a: TraditionProfile, b: TraditionProfile) -> float:
    n = min(len(a.tuning_deviations), len(b.tuning_deviations))
    if n == 0:
        return 0.0
    diff = a.tuning_deviations[:n] - b.tuning_deviations[:n]
    return float(np.sqrt(np.mean(diff ** 2))) / 50.0  # normalise by 50 cents


def ornament_distance(a: TraditionProfile, b: TraditionProfile) -> float:
    max_density = max(a.ornament_density, b.ornament_density, 1e-10)
    return abs(a.ornament_density - b.ornament_density) / max_density


def form_distance(a: TraditionProfile, b: TraditionProfile) -> float:
    max_seg = max(a.form_segments, b.form_segments, 1)
    return abs(a.form_segments - b.form_segments) / max_seg


def texture_distance(a: TraditionProfile, b: TraditionProfile) -> float:
    return abs(a.texture_polyphony - b.texture_polyphony)


def timbre_distance(a: TraditionProfile, b: TraditionProfile) -> float:
    return abs(a.timbral_brightness - b.timbral_brightness)


_DIMENSION_FN = {
    CulturalDimension.SCALE: scale_distance,
    CulturalDimension.RHYTHM: rhythm_distance,
    CulturalDimension.TUNING: tuning_distance,
    CulturalDimension.ORNAMENTATION: ornament_distance,
    CulturalDimension.FORM: form_distance,
    CulturalDimension.TEXTURE: texture_distance,
    CulturalDimension.TIMBRE: timbre_distance,
}


def cultural_distance(
    a: TraditionProfile,
    b: TraditionProfile,
    weights: Optional[Dict[CulturalDimension, float]] = None,
) -> CulturalDistanceResult:
    if weights is None:
        weights = {d: 1.0 for d in CulturalDimension}
    per_dim: Dict[CulturalDimension, float] = {}
    for dim, fn in _DIMENSION_FN.items():
        per_dim[dim] = fn(a, b)
    total_w = sum(weights.get(d, 0.0) for d in CulturalDimension)
    total = sum(weights.get(d, 0.0) * per_dim[d] for d in CulturalDimension) / max(total_w, 1e-10)
    dominant = max(per_dim, key=lambda d: per_dim[d])
    return CulturalDistanceResult(total=total, per_dimension=per_dim, dominant_dimension=dominant)


def pairwise_distance_matrix(
    profiles: List[TraditionProfile],
    weights: Optional[Dict[CulturalDimension, float]] = None,
) -> np.ndarray:
    n = len(profiles)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cultural_distance(profiles[i], profiles[j], weights).total
            mat[i, j] = mat[j, i] = d
    return mat


def tradition_mds_2d(profiles: List[TraditionProfile],
                     weights: Optional[Dict[CulturalDimension, float]] = None) -> np.ndarray:
    D = pairwise_distance_matrix(profiles, weights)
    n = len(D)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    coords = eigvecs[:, idx[:2]] * np.sqrt(np.maximum(eigvals[idx[:2]], 0.0))
    return coords


def tradition_cluster(profiles: List[TraditionProfile], k: int = 3,
                      weights: Optional[Dict[CulturalDimension, float]] = None,
                      max_iter: int = 50, seed: int = 42) -> List[int]:
    coords = tradition_mds_2d(profiles, weights)
    rng = np.random.RandomState(seed)
    centres = coords[rng.choice(len(coords), size=k, replace=False)]
    labels = np.zeros(len(coords), dtype=int)
    for _ in range(max_iter):
        dists = np.array([np.linalg.norm(coords - c, axis=1) for c in centres])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centres[c] = coords[mask].mean(axis=0)
    return labels.tolist()


TRADITION_TEMPLATES: Dict[str, Dict[str, float]] = {
    "north_indian": {"ornament_density": 2.5, "texture_polyphony": 0.15, "timbral_brightness": 0.4},
    "south_indian": {"ornament_density": 3.0, "texture_polyphony": 0.1, "timbral_brightness": 0.45},
    "arabic":       {"ornament_density": 1.8, "texture_polyphony": 0.2, "timbral_brightness": 0.5},
    "javanese":     {"ornament_density": 1.0, "texture_polyphony": 0.7, "timbral_brightness": 0.3},
    "west_african": {"ornament_density": 0.5, "texture_polyphony": 0.8, "timbral_brightness": 0.55},
    "celtic":       {"ornament_density": 2.0, "texture_polyphony": 0.25, "timbral_brightness": 0.5},
    "flamenco":     {"ornament_density": 2.2, "texture_polyphony": 0.3, "timbral_brightness": 0.55},
    "western_cpe":  {"ornament_density": 0.3, "texture_polyphony": 0.9, "timbral_brightness": 0.6},
    "klezmer":      {"ornament_density": 1.5, "texture_polyphony": 0.3, "timbral_brightness": 0.5},
    "chinese":      {"ornament_density": 1.2, "texture_polyphony": 0.35, "timbral_brightness": 0.35},
}


def influence_graph(
    profiles: List[TraditionProfile],
    threshold: float = 0.3,
    weights: Optional[Dict[CulturalDimension, float]] = None,
) -> List[Tuple[int, int, float]]:
    D = pairwise_distance_matrix(profiles, weights)
    edges = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            if D[i, j] < threshold:
                edges.append((i, j, float(D[i, j])))
    return edges


def borrowing_score(source: TraditionProfile, target: TraditionProfile) -> Dict[CulturalDimension, float]:
    scores = {}
    for dim, fn in _DIMENSION_FN.items():
        d = fn(source, target)
        scores[dim] = max(0.0, 1.0 - d)
    return scores


def fusion_potential(profiles: List[TraditionProfile]) -> float:
    if len(profiles) < 2:
        return 0.0
    D = pairwise_distance_matrix(profiles)
    upper = D[np.triu_indices_from(D, k=1)]
    mean_d = float(np.mean(upper))
    std_d = float(np.std(upper))
    return float(np.clip(1.0 - mean_d + 0.5 * std_d, 0, 1))
