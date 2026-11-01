"""Feature normalization and scaling for music data pipelines.

Standard scaling, min-max, robust, per-feature and per-track normalization.
Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ScalingMethod(Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    L2 = "l2"
    LOG = "log"
    POWER = "power"


@dataclass
class FittedScaler:
    method: ScalingMethod
    center: np.ndarray
    scale: np.ndarray
    n_features: int


def fit_standard(X: np.ndarray) -> FittedScaler:
    """Fit zero-mean unit-variance scaler."""
    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-10] = 1.0
    return FittedScaler(ScalingMethod.STANDARD, center, scale, X.shape[1] if X.ndim > 1 else 1)


def fit_minmax(X: np.ndarray, feature_range: tuple[float, float] = (0.0, 1.0)) -> FittedScaler:
    """Fit min-max scaler to [a, b]."""
    a, b = feature_range
    xmin = np.min(X, axis=0)
    xmax = np.max(X, axis=0)
    rng = xmax - xmin
    rng[rng < 1e-10] = 1.0
    center = xmin
    scale = rng / (b - a)
    return FittedScaler(ScalingMethod.MINMAX, center, scale, X.shape[1] if X.ndim > 1 else 1)


def fit_robust(X: np.ndarray) -> FittedScaler:
    """Fit robust scaler using median and IQR."""
    center = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    scale = q75 - q25
    scale[np.abs(scale) < 1e-10] = 1.0
    return FittedScaler(ScalingMethod.ROBUST, center, scale, X.shape[1] if X.ndim > 1 else 1)


def transform(X: np.ndarray, scaler: FittedScaler) -> np.ndarray:
    """Apply fitted scaler to data."""
    if scaler.method == ScalingMethod.STANDARD:
        return (X - scaler.center) / scaler.scale
    elif scaler.method == ScalingMethod.MINMAX:
        return (X - scaler.center) / scaler.scale
    elif scaler.method == ScalingMethod.ROBUST:
        return (X - scaler.center) / scaler.scale
    elif scaler.method == ScalingMethod.L2:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return X / norms
    return X


def inverse_transform(X: np.ndarray, scaler: FittedScaler) -> np.ndarray:
    """Reverse scaling."""
    if scaler.method in (ScalingMethod.STANDARD, ScalingMethod.ROBUST):
        return X * scaler.scale + scaler.center
    elif scaler.method == ScalingMethod.MINMAX:
        return X * scaler.scale + scaler.center
    return X


def per_track_normalize(features: np.ndarray) -> np.ndarray:
    """Normalize each row (track) independently to unit L2 norm."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return features / norms


def log_normalize(X: np.ndarray, offset: float = 1e-6) -> np.ndarray:
    """Log-transform non-negative features (e.g. spectral magnitudes)."""
    return np.log(X + offset)


def power_normalize(X: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Power-law compression: sign(x)|x|^alpha."""
    return np.sign(X) * np.abs(X) ** alpha


def quantile_clip(X: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Clip features to quantile range to reduce outlier influence."""
    lo = np.quantile(X, lower, axis=0)
    hi = np.quantile(X, upper, axis=0)
    return np.clip(X, lo, hi)


def feature_importance_weight(
    X: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Variance-ratio feature weighting: between-class / within-class variance."""
    classes = np.unique(y)
    overall_mean = np.mean(X, axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for c in classes:
        mask = y == c
        class_mean = np.mean(X[mask], axis=0)
        n_c = np.sum(mask)
        between += n_c * (class_mean - overall_mean) ** 2
        within += np.sum((X[mask] - class_mean) ** 2, axis=0)
    within[within < 1e-10] = 1e-10
    return between / within
