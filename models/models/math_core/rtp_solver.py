"""RTP kernel — patent claim execution primitives.

US 12,361,263 B1 claim 1: dominance → evolution → simulate → eliminate → test → identify.
US 12,651,165 B2 claim 1: SOM visualisation → train GRN node models.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def dominance_factor(series: np.ndarray, w_intensity: float = 0.6,
                     w_frequency: float = 0.4) -> dict[str, float]:
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or x.size < 2:
        return {"intensity": 0.0, "frequency": 0.0, "dominance": 0.0}
    d = np.diff(x)
    intensity = float(math.sqrt(float(np.mean(d * d))))
    s = np.sign(d)
    crossings = int(np.sum(s[1:] != s[:-1])) if d.size > 1 else 0
    frequency = crossings / max(d.size - 1, 1)
    return {
        "intensity": intensity,
        "frequency": frequency,
        "dominance": w_intensity * intensity + w_frequency * frequency,
    }


def evolution_of_internal_component(series: np.ndarray, window: int = 8,
                                    half_life: float = 5.0,
                                    w_intensity: float = 0.6,
                                    w_frequency: float = 0.4) -> dict[str, Any]:
    x = np.asarray(series, dtype=float)
    window = min(window, x.size)
    n_win = max(x.size - window + 1, 1)
    dom = [dominance_factor(x[k:k + window], w_intensity, w_frequency)["dominance"]
           for k in range(n_win)]
    alpha = 1.0 - 0.5 ** (1.0 / max(half_life, 0.5))
    e = 0.0
    traj: list[float] = []
    for dk in dom:
        e = (1.0 - alpha) * e + alpha * dk
        traj.append(e)
    return {"dominance_windows": dom, "trajectory": traj, "final": traj[-1] if traj else 0.0}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b)) / den if den > 1e-12 else 0.0


def lagged_correlation(source: np.ndarray, target: np.ndarray,
                       max_lag: int) -> tuple[float, int]:
    best, best_lag = 0.0, 0
    t = source.size
    for lag in range(0, min(max_lag, t - 3) + 1):
        c = abs(_corr(source[:t - lag], target[lag:]))
        if c > best + 1e-12:
            best, best_lag = c, lag
    return best, best_lag


def simulate_relationships(entity_series: np.ndarray, max_lag: int = 4,
                           dominance_exponent: float = 0.0) -> np.ndarray:
    vals = np.asarray(entity_series, dtype=float)
    t, n = vals.shape
    dom_scores = np.array([dominance_factor(vals[:, j])["dominance"] for j in range(n)])
    max_dom = float(dom_scores.max()) if dom_scores.max() > 0 else 1.0
    norm_dom = dom_scores / max_dom
    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rho, _ = lagged_correlation(vals[:, i], vals[:, j], max_lag)
            score_matrix[i, j] = (norm_dom[i] ** dominance_exponent) * rho
    return score_matrix


def eliminate_drivers(score_matrix: np.ndarray, target_idx: int,
                      quantile: float = 0.67) -> list[int]:
    n = score_matrix.shape[0]
    scores = score_matrix[:, target_idx].copy()
    scores[target_idx] = -np.inf
    n_keep = max(1, int(math.floor(quantile * (n - 1))))
    order = np.argsort(scores)[::-1]
    kept = [int(idx) for idx in order[:n_keep] if idx != target_idx]
    return kept


def test_drivers(entity_series: np.ndarray, target_idx: int,
                 kept_indices: list[int], threshold: float = 0.0) -> list[tuple[int, float]]:
    vals = np.asarray(entity_series, dtype=float)
    y = vals[:, target_idx]
    y_centered = y - y.mean()
    ss_total = float(np.sum(y_centered ** 2))
    if ss_total < 1e-12:
        return [(idx, 0.0) for idx in kept_indices]
    passing = []
    residual = y_centered.copy()
    prev_r2 = 0.0
    for idx in kept_indices:
        x = vals[:, idx]
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered ** 2))
        if denom < 1e-12:
            continue
        beta = float(np.sum(residual * x_centered)) / denom
        fitted = beta * x_centered
        residual = residual - fitted
        ss_res = float(np.sum(residual ** 2))
        r2 = 1.0 - ss_res / ss_total
        gain = r2 - prev_r2
        prev_r2 = r2
        if gain >= threshold:
            passing.append((idx, gain))
    return passing


def identify_target(passing_drivers: list[tuple[int, float]],
                    score_matrix: np.ndarray, target_idx: int) -> int | None:
    if not passing_drivers:
        return None
    best_idx = max(passing_drivers,
                   key=lambda p: p[1] * score_matrix[p[0], target_idx])[0]
    return best_idx
