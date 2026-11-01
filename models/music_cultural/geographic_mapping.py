"""Geographic mapping of musical styles: spatial autocorrelation, diffusion
gradients, IDW interpolation, Mantel tests, cultural boundary detection.

Models the geographic dimension of musical culture — how style features
correlate with distance, where cultural boundaries lie, and how innovations
diffuse from a source across space.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class GeographicPoint:
    lat: float
    lon: float
    name: str
    features: np.ndarray


@dataclass
class StyleRegion:
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    characteristic_features: np.ndarray


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def geographic_distance_matrix(points: List[GeographicPoint]) -> np.ndarray:
    """Pairwise geographic distance matrix (km)."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(points[i].lat, points[i].lon,
                                   points[j].lat, points[j].lon)
            D[i, j] = D[j, i] = d
    return D


def feature_distance_matrix(points: List[GeographicPoint]) -> np.ndarray:
    """Pairwise Euclidean feature distance matrix."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i].features - points[j].features))
            D[i, j] = D[j, i] = d
    return D


def _upper_triangle(M: np.ndarray) -> np.ndarray:
    """Extract strict upper triangle as flat array."""
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    return M[idx]


def mantel_test(geo_dist: np.ndarray, feat_dist: np.ndarray,
                n_permutations: int = 999) -> Tuple[float, float]:
    """Mantel test: Pearson correlation between geographic and feature
    distance matrices, with permutation p-value.

    Returns (r, p_value).
    """
    g = _upper_triangle(geo_dist)
    f = _upper_triangle(feat_dist)
    if len(g) < 3:
        return 0.0, 1.0
    r_obs = float(np.corrcoef(g, f)[0, 1])
    n = geo_dist.shape[0]
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        perm_geo = geo_dist[np.ix_(perm, perm)]
        g_perm = _upper_triangle(perm_geo)
        r_perm = float(np.corrcoef(g_perm, f)[0, 1])
        if r_perm >= r_obs:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return r_obs, p_value


def spatial_autocorrelation(points: List[GeographicPoint],
                            feature_index: int) -> float:
    """Moran's I for a single feature across geographic points.

    Uses inverse-distance weights. Returns I ∈ [-1, 1].
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = np.array([p.features[feature_index] for p in points])
    x_bar = np.mean(x)
    dev = x - x_bar
    geo = geographic_distance_matrix(points)
    W = np.zeros_like(geo)
    mask = geo > 0
    W[mask] = 1.0 / geo[mask]
    np.fill_diagonal(W, 0.0)
    W_sum = W.sum()
    if W_sum == 0:
        return 0.0
    num = 0.0
    for i in range(n):
        for j in range(n):
            num += W[i, j] * dev[i] * dev[j]
    denom = np.sum(dev ** 2)
    if denom == 0:
        return 0.0
    return float((n / W_sum) * (num / denom))


def inverse_distance_weighting(known_points: List[GeographicPoint],
                                query_lat: float, query_lon: float,
                                power: float = 2.0) -> np.ndarray:
    """IDW interpolation of feature vectors at a query location."""
    if not known_points:
        return np.array([])
    dim = len(known_points[0].features)
    dists = np.array([haversine_distance(p.lat, p.lon, query_lat, query_lon)
                      for p in known_points])
    exact = np.where(dists < 1e-10)[0]
    if len(exact) > 0:
        return known_points[exact[0]].features.copy()
    weights = 1.0 / (dists ** power)
    weights /= weights.sum()
    result = np.zeros(dim)
    for w, p in zip(weights, known_points):
        result += w * p.features
    return result


def classify_region(point: GeographicPoint,
                    regions: List[StyleRegion]) -> Optional[str]:
    """Assign point to nearest region within radius, or None."""
    best_name = None
    best_dist = float("inf")
    for r in regions:
        d = haversine_distance(point.lat, point.lon, r.center_lat, r.center_lon)
        if d <= r.radius_km and d < best_dist:
            best_dist = d
            best_name = r.name
    return best_name


def diffusion_gradient(points: List[GeographicPoint],
                       source_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Feature similarity decay from a source point.

    Returns (distances_km, similarities) sorted by distance.
    """
    n = len(points)
    src = points[source_index]
    dists = []
    sims = []
    for i in range(n):
        if i == source_index:
            continue
        d = haversine_distance(src.lat, src.lon, points[i].lat, points[i].lon)
        norm_src = np.linalg.norm(src.features)
        norm_i = np.linalg.norm(points[i].features)
        if norm_src > 0 and norm_i > 0:
            sim = float(np.dot(src.features, points[i].features) / (norm_src * norm_i))
        else:
            sim = 0.0
        dists.append(d)
        sims.append(sim)
    order = np.argsort(dists)
    return np.array(dists)[order], np.array(sims)[order]


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100,
            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means returning (labels, centroids)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
    return labels, centroids


def cultural_boundary_detection(points: List[GeographicPoint],
                                 n_clusters: int = 3) -> Dict[str, object]:
    """Detect cultural boundaries via k-means on feature space.

    Returns dict with 'labels', 'centroids', 'boundary_pairs' (indices of
    geographically adjacent points with different cluster labels).
    """
    if len(points) < n_clusters:
        return {"labels": np.zeros(len(points), dtype=int),
                "centroids": np.array([]),
                "boundary_pairs": []}
    X = np.array([p.features for p in points])
    labels, centroids = _kmeans(X, n_clusters)
    geo = geographic_distance_matrix(points)
    median_dist = np.median(geo[geo > 0]) if np.any(geo > 0) else 1.0
    boundary_pairs = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j] and geo[i, j] <= median_dist:
                boundary_pairs.append((i, j))
    return {"labels": labels, "centroids": centroids,
            "boundary_pairs": boundary_pairs}


def isolation_by_distance(points: List[GeographicPoint],
                          feature_index: int) -> Tuple[float, float]:
    """Test isolation-by-distance: regression of feature difference on
    geographic distance.

    Returns (slope, r_squared).
    """
    n = len(points)
    if n < 3:
        return 0.0, 0.0
    geo = geographic_distance_matrix(points)
    g = _upper_triangle(geo)
    feat = np.array([p.features[feature_index] for p in points])
    feat_diff = np.zeros_like(geo)
    for i in range(n):
        for j in range(i + 1, n):
            feat_diff[i, j] = feat_diff[j, i] = abs(feat[i] - feat[j])
    f = _upper_triangle(feat_diff)
    if np.std(g) < 1e-15:
        return 0.0, 0.0
    slope = float(np.cov(g, f)[0, 1] / np.var(g))
    r = float(np.corrcoef(g, f)[0, 1])
    return slope, r ** 2
