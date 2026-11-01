"""VADE Drift Estimator — Volumetric Angular Drift Estimator (DIV4).

Whitening Σ^{-1/2} with Tikhonov ridge, projection to unit sphere,
drift estimation θ between successive projections, monitoring with alarms.
"""

from __future__ import annotations

import numpy as np

NOMINAL_DRIFT_RAD = 0.12


class DriftEstimator:
    def __init__(self, dim: int = 5, ridge: float = 1e-6,
                 cond_ceiling: float = 1e6, decay: float = 0.95):
        self.dim = dim
        self.ridge = ridge
        self.cond_ceiling = cond_ceiling
        self.decay = decay
        self.cov: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.whitener: np.ndarray | None = None
        self.ridged = False

    def fit_whitening(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dim or X.shape[0] < self.dim + 1:
            raise ValueError(f"X must be (n >= {self.dim + 1}, {self.dim})")
        cov = np.cov(X, rowvar=False)
        if self.cov is not None:
            cov = self.decay * self.cov + (1 - self.decay) * cov
        w, v = np.linalg.eigh(cov)
        cond = float(w.max() / max(w.min(), 1e-300))
        self.ridged = cond > self.cond_ceiling
        if self.ridged:
            w = w + self.ridge
        self.cov = cov
        self.mean = X.mean(0)
        self.whitener = (v * (1.0 / np.sqrt(np.maximum(w, 1e-300)))) @ v.T
        return self.whitener

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.whitener is None:
            raise RuntimeError("call fit_whitening first")
        z = self.whitener @ (np.asarray(x, dtype=float) - self.mean)
        n = np.linalg.norm(z)
        return z / n if n > 0 else z

    def drift(self, x_prev: np.ndarray, x_now: np.ndarray) -> float:
        # atan2 form is exact at p=q and avoids arccos ill-conditioning at 1
        p, q = self.project(x_prev), self.project(x_now)
        return float(2.0 * np.arctan2(np.linalg.norm(p - q), np.linalg.norm(p + q)))

    def monitor(self, stream: np.ndarray, nominal: float | None = None,
                alarm_factor: float = 3.0) -> dict:
        X = np.asarray(stream, dtype=float)
        thetas = [self.drift(X[i - 1], X[i]) for i in range(1, len(X))]
        t = np.array(thetas)
        if nominal is None:
            nominal = float(np.median(t)) if t.size else NOMINAL_DRIFT_RAD
        alarms = [i + 1 for i, th in enumerate(thetas) if th > alarm_factor * nominal]
        return {
            "theta": thetas,
            "mean": float(t.mean()) if t.size else 0.0,
            "std": float(t.std()) if t.size else 0.0,
            "alarms": alarms,
            "nominal": nominal,
            "ridged": self.ridged,
        }
