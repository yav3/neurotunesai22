"""Perceptual timbre space: multidimensional scaling, MPEG-7 descriptors, instrument clustering.

Models timbre as points in a perceptual space derived from psychoacoustic
experiments (Grey 1977, McAdams 1995, Peeters 2004). Provides MPEG-7 audio
descriptors, Sethares-style roughness, and clustering utilities for
instrument family identification and timbre interpolation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class MPEG7Descriptor(Enum):
    HARMONIC_SPECTRAL_CENTROID = "harmonic_spectral_centroid"
    HARMONIC_SPECTRAL_DEVIATION = "harmonic_spectral_deviation"
    HARMONIC_SPECTRAL_SPREAD = "harmonic_spectral_spread"
    LOG_ATTACK_TIME = "log_attack_time"
    TEMPORAL_CENTROID = "temporal_centroid"
    SPECTRAL_FLATNESS = "spectral_flatness"


@dataclass
class TimbreVector:
    """A point in perceptual timbre space."""
    label: str
    descriptors: np.ndarray
    descriptor_names: List[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.descriptors)


def harmonic_spectral_centroid(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    """MPEG-7 AudioSpectrumCentroid for harmonic partials."""
    total = np.sum(magnitudes)
    if total < 1e-15:
        return 0.0
    return float(np.sum(magnitudes * freqs) / total)


def harmonic_spectral_deviation(magnitudes: np.ndarray) -> float:
    """Deviation of harmonic amplitudes from spectral envelope (3-point smoothed)."""
    n = len(magnitudes)
    if n < 3:
        return 0.0
    smoothed = np.convolve(magnitudes, [1/3, 1/3, 1/3], mode='same')
    smoothed[0] = (magnitudes[0] + magnitudes[1]) / 2
    smoothed[-1] = (magnitudes[-2] + magnitudes[-1]) / 2
    dev = np.abs(magnitudes - smoothed)
    total = np.sum(magnitudes)
    if total < 1e-15:
        return 0.0
    return float(np.sum(dev) / total)


def harmonic_spectral_spread(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    """RMS spread of harmonics around centroid (Hz)."""
    centroid = harmonic_spectral_centroid(magnitudes, freqs)
    total = np.sum(magnitudes)
    if total < 1e-15:
        return 0.0
    return float(np.sqrt(np.sum(magnitudes * (freqs - centroid) ** 2) / total))


def log_attack_time(envelope: np.ndarray, sr: int = 44100,
                    threshold_lo: float = 0.2, threshold_hi: float = 0.9) -> float:
    """Log₁₀ of attack time in seconds (MPEG-7 AudioSpectrumFlatness)."""
    peak = np.max(np.abs(envelope))
    if peak < 1e-15:
        return -3.0
    norm = np.abs(envelope) / peak
    lo_idx = np.searchsorted(norm, threshold_lo)
    hi_idx = np.searchsorted(norm, threshold_hi)
    if hi_idx <= lo_idx:
        return -3.0
    attack_s = (hi_idx - lo_idx) / sr
    return float(np.log10(max(attack_s, 1e-6)))


def temporal_centroid(envelope: np.ndarray, sr: int = 44100) -> float:
    """Temporal centroid: center of gravity of the amplitude envelope (seconds)."""
    total = np.sum(np.abs(envelope))
    if total < 1e-15:
        return 0.0
    times = np.arange(len(envelope)) / sr
    return float(np.sum(np.abs(envelope) * times) / total)


def spectral_flatness_measure(spectrum: np.ndarray) -> float:
    """Wiener entropy: geometric mean / arithmetic mean of power spectrum."""
    power = spectrum ** 2
    power = power[power > 0]
    if len(power) == 0:
        return 0.0
    geo_mean = np.exp(np.mean(np.log(power + 1e-30)))
    arith_mean = np.mean(power)
    if arith_mean < 1e-30:
        return 0.0
    return float(geo_mean / arith_mean)


def compute_mpeg7_descriptors(magnitudes: np.ndarray, freqs: np.ndarray,
                              envelope: np.ndarray, sr: int = 44100) -> Dict[str, float]:
    return {
        "harmonic_spectral_centroid": harmonic_spectral_centroid(magnitudes, freqs),
        "harmonic_spectral_deviation": harmonic_spectral_deviation(magnitudes),
        "harmonic_spectral_spread": harmonic_spectral_spread(magnitudes, freqs),
        "log_attack_time": log_attack_time(envelope, sr),
        "temporal_centroid": temporal_centroid(envelope, sr),
        "spectral_flatness": spectral_flatness_measure(magnitudes),
    }


def sethares_roughness(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Sethares (1993) roughness model: sum of beating between all partial pairs."""
    n = len(freqs)
    roughness = 0.0
    a1, b1 = 3.5, 5.75  # Plomp-Levelt parameters
    a2, b2 = 0.24, 0.0207  # bandwidth scaling
    for i in range(n):
        for j in range(i + 1, n):
            f_min = min(freqs[i], freqs[j])
            f_diff = abs(freqs[i] - freqs[j])
            s = a2 * f_min + b2
            if s < 1e-10:
                continue
            x = f_diff / s
            r = amps[i] * amps[j] * (np.exp(-a1 * x) - np.exp(-b1 * x))
            roughness += max(0.0, r)
    return float(roughness)


def pairwise_timbre_distance(vectors: List[TimbreVector],
                             weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted Euclidean distance matrix between timbre vectors."""
    n = len(vectors)
    d = vectors[0].dim
    mat = np.zeros((n, d))
    for i, v in enumerate(vectors):
        mat[i] = v.descriptors[:d]
    if weights is None:
        weights = np.ones(d)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = (mat[i] - mat[j]) * np.sqrt(weights[:d])
            dist[i, j] = dist[j, i] = float(np.sqrt(np.sum(diff ** 2)))
    return dist


def mds_2d(dist_matrix: np.ndarray) -> np.ndarray:
    """Classical MDS: project distance matrix to 2D coordinates."""
    n = dist_matrix.shape[0]
    D2 = dist_matrix ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = 2
    coords = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0))
    return coords


def timbre_centroid(vectors: List[TimbreVector]) -> np.ndarray:
    """Mean descriptor vector across a set of timbre points."""
    mat = np.array([v.descriptors for v in vectors])
    return np.mean(mat, axis=0)


def timbre_interpolation(v1: TimbreVector, v2: TimbreVector,
                         alpha: float) -> np.ndarray:
    """Linear interpolation between two timbre vectors."""
    return (1 - alpha) * v1.descriptors + alpha * v2.descriptors


def cluster_timbres_kmeans(vectors: List[TimbreVector], k: int,
                           max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering in timbre descriptor space."""
    rng = np.random.RandomState(seed)
    mat = np.array([v.descriptors for v in vectors])
    n, d = mat.shape
    idx = rng.choice(n, k, replace=False)
    centroids = mat[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.array([np.sum((mat - c) ** 2, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = mat[labels == c]
            if len(members) > 0:
                centroids[c] = np.mean(members, axis=0)
    return labels, centroids


def silhouette_score(vectors: List[TimbreVector], labels: np.ndarray) -> float:
    """Mean silhouette coefficient for timbre clustering quality."""
    mat = np.array([v.descriptors for v in vectors])
    n = len(vectors)
    if n < 2:
        return 0.0
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    scores = np.zeros(n)
    for i in range(n):
        same = mat[labels == labels[i]]
        if len(same) <= 1:
            scores[i] = 0.0
            continue
        a_i = np.mean(np.sqrt(np.sum((same - mat[i]) ** 2, axis=1)))
        b_i = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = mat[labels == lbl]
            d_other = np.mean(np.sqrt(np.sum((other - mat[i]) ** 2, axis=1)))
            b_i = min(b_i, d_other)
        scores[i] = (b_i - a_i) / max(a_i, b_i, 1e-10)
    return float(np.mean(scores))


def grey_dimensions(log_attack: float, spectral_centroid: float,
                    spectral_flux: float) -> np.ndarray:
    """Map to Grey (1977) 3D timbre space: attack, brightness, flux."""
    return np.array([log_attack, spectral_centroid, spectral_flux])


def mcadams_dimensions(log_attack: float, spectral_centroid: float,
                       spectral_irregularity: float, spectral_flux: float) -> np.ndarray:
    """McAdams et al. (1995) 4D timbre space."""
    return np.array([log_attack, spectral_centroid, spectral_irregularity, spectral_flux])
