"""Music dataset: feature normalization, train/val/test splits, batching."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureNormalizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    _fitted: bool = False

    def fit(self, X: np.ndarray) -> FeatureNormalizer:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-8] = 1.0
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Call fit() before transform()")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Call fit() before inverse_transform()")
        return X * self.std + self.mean


class MusicDataset:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        metadata: list[dict[str, Any]] | None = None,
        normalize: bool = True,
    ) -> None:
        self.raw_features = features.copy()
        self.labels = labels.copy() if labels is not None else None
        self.metadata = metadata or [{} for _ in range(len(features))]
        self.normalizer = FeatureNormalizer()

        if normalize and len(features) > 1:
            self.features = self.normalizer.fit_transform(features.astype(np.float64))
        else:
            self.features = features.astype(np.float64)

    def __len__(self) -> int:
        return len(self.features)

    def split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> tuple[MusicDataset, MusicDataset, MusicDataset]:
        rng = np.random.RandomState(seed)
        n = len(self)
        indices = rng.permutation(n)

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        def _subset(idx: np.ndarray) -> MusicDataset:
            ds = MusicDataset.__new__(MusicDataset)
            ds.raw_features = self.raw_features[idx]
            ds.features = self.features[idx]
            ds.labels = self.labels[idx] if self.labels is not None else None
            ds.metadata = [self.metadata[i] for i in idx]
            ds.normalizer = self.normalizer
            return ds

        return _subset(train_idx), _subset(val_idx), _subset(test_idx)

    def batches(self, batch_size: int = 32, shuffle: bool = True, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray | None]]:
        rng = np.random.RandomState(seed)
        n = len(self)
        indices = rng.permutation(n) if shuffle else np.arange(n)

        result = []
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            feat = self.features[idx]
            lab = self.labels[idx] if self.labels is not None else None
            result.append((feat, lab))
        return result

    def feature_dim(self) -> int:
        return self.features.shape[1] if self.features.ndim >= 2 else 1

    def class_counts(self) -> dict[int, int]:
        if self.labels is None:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
