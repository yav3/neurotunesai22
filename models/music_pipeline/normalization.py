"""Feature normalization and standardization for ML pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureStatistics:
    """Per-feature statistics computed over a corpus."""
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    n_samples: int = 0

    @property
    def n_features(self) -> int:
        return len(self.mean)


class ZScoreNormalizer:
    """Standard z-score normalization: (x - mean) / std."""

    def __init__(self) -> None:
        self._stats: FeatureStatistics | None = None

    @property
    def is_fitted(self) -> bool:
        return self._stats is not None

    def fit(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("Expected 2D array")
        self._stats = FeatureStatistics(
            mean=X.mean(axis=0),
            std=np.maximum(X.std(axis=0), 1e-10),
            min_val=X.min(axis=0),
            max_val=X.max(axis=0),
            n_samples=X.shape[0],
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("Not fitted")
        return (X - self._stats.mean) / self._stats.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("Not fitted")
        return Z * self._stats.std + self._stats.mean

    @property
    def statistics(self) -> FeatureStatistics | None:
        return self._stats


class MinMaxNormalizer:
    """Scale features to [0, 1] range."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self._min: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._min is not None

    def fit(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("Expected 2D array")
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        data_range = np.maximum(x_max - x_min, 1e-10)
        lo, hi = self.feature_range
        self._scale = (hi - lo) / data_range
        self._min = x_min

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._min is None or self._scale is None:
            raise RuntimeError("Not fitted")
        lo, _ = self.feature_range
        return (X - self._min) * self._scale + lo

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self._min is None or self._scale is None:
            raise RuntimeError("Not fitted")
        lo, _ = self.feature_range
        return (X_scaled - lo) / self._scale + self._min


class RobustNormalizer:
    """Median/IQR-based normalization, robust to outliers."""

    def __init__(self) -> None:
        self._median: np.ndarray | None = None
        self._iqr: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._median is not None

    def fit(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("Expected 2D array")
        self._median = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self._iqr = np.maximum(q75 - q25, 1e-10)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._median is None or self._iqr is None:
            raise RuntimeError("Not fitted")
        return (X - self._median) / self._iqr

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class FeatureSelector:
    """Select features by variance threshold or correlation pruning."""

    def __init__(self, variance_threshold: float = 0.01) -> None:
        self.variance_threshold = variance_threshold
        self._mask: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._mask is not None

    @property
    def n_selected(self) -> int:
        if self._mask is None:
            return 0
        return int(np.sum(self._mask))

    def fit(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("Expected 2D array")
        variances = np.var(X, axis=0)
        self._mask = variances >= self.variance_threshold

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("Not fitted")
        return X[:, self._mask]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def correlation_prune(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Remove features whose pairwise |correlation| exceeds threshold."""
        if X.ndim != 2 or X.shape[1] < 2:
            return X
        corr = np.corrcoef(X, rowvar=False)
        n = corr.shape[0]
        drop = set()
        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j in drop:
                    continue
                if abs(corr[i, j]) > threshold:
                    drop.add(j)
        keep = sorted(set(range(n)) - drop)
        return X[:, keep]
