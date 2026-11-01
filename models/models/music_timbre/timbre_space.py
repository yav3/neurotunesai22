"""Timbre space: multidimensional timbre representations and similarity.

Models perceptual timbre space as defined by Grey (1977), McAdams et al. (1995),
and more recent embedding approaches. Maps spectral/temporal descriptors to
low-dimensional perceptual coordinates for instrument comparison and interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TimbrePoint:
    name: str
    coordinates: np.ndarray
    descriptors: dict[str, float] = field(default_factory=dict)

    @property
    def n_dims(self) -> int:
        return len(self.coordinates)


@dataclass(frozen=True)
class TimbreDistance:
    point_a: str
    point_b: str
    euclidean: float
    weighted: float
    per_dimension: np.ndarray


@dataclass
class TimbreSpace:
    """Multidimensional timbre space with distance computation and interpolation."""
    n_dims: int
    dim_names: list[str]
    dim_weights: np.ndarray
    _points: dict[str, TimbrePoint] = field(default_factory=dict)

    def add_point(self, point: TimbrePoint) -> None:
        if point.n_dims != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dims, got {point.n_dims}")
        self._points[point.name] = point

    def get_point(self, name: str) -> TimbrePoint | None:
        return self._points.get(name)

    @property
    def n_points(self) -> int:
        return len(self._points)

    def distance(self, name_a: str, name_b: str) -> TimbreDistance:
        a = self._points[name_a]
        b = self._points[name_b]
        diff = a.coordinates - b.coordinates
        per_dim = np.abs(diff)
        euclidean = float(np.linalg.norm(diff))
        weighted = float(np.sqrt(np.sum(self.dim_weights * diff ** 2)))
        return TimbreDistance(name_a, name_b, euclidean, weighted, per_dim)

    def nearest(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Find k nearest points to a query coordinate."""
        dists = []
        for name, pt in self._points.items():
            diff = query - pt.coordinates
            d = float(np.sqrt(np.sum(self.dim_weights * diff ** 2)))
            dists.append((name, d))
        dists.sort(key=lambda x: x[1])
        return dists[:k]

    def interpolate(self, name_a: str, name_b: str, alpha: float) -> np.ndarray:
        """Linear interpolation between two timbre points."""
        a = self._points[name_a].coordinates
        b = self._points[name_b].coordinates
        return (1.0 - alpha) * a + alpha * b

    def centroid(self) -> np.ndarray:
        if not self._points:
            return np.zeros(self.n_dims)
        coords = np.array([p.coordinates for p in self._points.values()])
        return np.mean(coords, axis=0)

    def pairwise_distance_matrix(self) -> tuple[list[str], np.ndarray]:
        names = sorted(self._points.keys())
        n = len(names)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(names[i], names[j])
                mat[i, j] = d.weighted
                mat[j, i] = d.weighted
        return names, mat


def grey_timbre_axes() -> TimbreSpace:
    """Classic 3D timbre space (Grey 1977): attack time, spectral centroid, spectral flux."""
    return TimbreSpace(
        n_dims=3,
        dim_names=["attack_time", "spectral_centroid", "spectral_flux"],
        dim_weights=np.array([1.0, 1.0, 1.0]),
    )


def mcadams_timbre_axes() -> TimbreSpace:
    """McAdams et al. (1995) timbre dimensions: attack, brightness, spectral irregularity."""
    return TimbreSpace(
        n_dims=3,
        dim_names=["log_attack_time", "brightness", "spectral_irregularity"],
        dim_weights=np.array([0.8, 1.0, 0.6]),
    )


def descriptors_to_coordinates(
    attack_time_ms: float,
    spectral_centroid_hz: float,
    spectral_irregularity: float,
    brightness_val: float,
    roughness: float,
) -> np.ndarray:
    """Map raw descriptors to a normalized 5D timbre coordinate.

    Normalization ranges chosen from typical instrument values.
    """
    lat_norm = np.clip((np.log10(max(attack_time_ms, 0.1)) - 0.0) / 3.0, 0, 1)
    sc_norm = np.clip((spectral_centroid_hz - 200.0) / 5800.0, 0, 1)
    si_norm = np.clip(spectral_irregularity / 2.0, 0, 1)
    br_norm = np.clip(brightness_val, 0, 1)
    ro_norm = np.clip(roughness / 5.0, 0, 1)
    return np.array([lat_norm, sc_norm, si_norm, br_norm, ro_norm])


def timbre_similarity(desc_a: np.ndarray, desc_b: np.ndarray,
                      weights: np.ndarray | None = None) -> float:
    """Weighted cosine similarity between timbre descriptor vectors."""
    if weights is None:
        weights = np.ones(len(desc_a))
    wa = desc_a * weights
    wb = desc_b * weights
    na = np.linalg.norm(wa)
    nb = np.linalg.norm(wb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(wa, wb) / (na * nb))


def mds_reduce(distance_matrix: np.ndarray, n_dims: int = 3) -> np.ndarray:
    """Classical MDS (multidimensional scaling) via double centering + eigen-decomposition.

    Projects a distance matrix to n_dims coordinates preserving pairwise distances.
    """
    n = distance_matrix.shape[0]
    D_sq = distance_matrix ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * centering @ D_sq @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    k = min(n_dims, np.sum(eigenvalues > 1e-10))
    if k == 0:
        return np.zeros((n, n_dims))
    coords = eigenvectors[:, :k] * np.sqrt(np.maximum(eigenvalues[:k], 0.0))
    if k < n_dims:
        padding = np.zeros((n, n_dims - k))
        coords = np.hstack([coords, padding])
    return coords


def stress(distance_matrix: np.ndarray, coordinates: np.ndarray) -> float:
    """Kruskal's stress-1 for evaluating MDS fit quality."""
    n = len(coordinates)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d_orig = distance_matrix[i, j]
            d_mds = float(np.linalg.norm(coordinates[i] - coordinates[j]))
            num += (d_orig - d_mds) ** 2
            den += d_orig ** 2
    if den < 1e-12:
        return 0.0
    return float(np.sqrt(num / den))
