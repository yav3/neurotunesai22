"""Dataset construction: stratified splits, feature caching, statistics."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DatasetSplit:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train_ids), len(self.val_ids), len(self.test_ids)


def stratified_split(
    item_ids: list[str],
    labels: list[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    rng = np.random.default_rng(seed)
    label_groups: dict[int, list[str]] = {}
    for item_id, label in zip(item_ids, labels):
        label_groups.setdefault(label, []).append(item_id)

    train, val, test = [], [], []
    for label in sorted(label_groups):
        ids = list(label_groups[label])
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        train.extend(ids[:n_train])
        val.extend(ids[n_train:n_train + n_val])
        test.extend(ids[n_train + n_val:])
    return DatasetSplit(train_ids=train, val_ids=val, test_ids=test)


class FeatureCache:
    def __init__(self, max_entries: int = 1000):
        self._store: dict[str, np.ndarray] = {}
        self._max = max_entries
        self._hits = 0
        self._misses = 0

    @staticmethod
    def content_key(data: np.ndarray) -> str:
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]

    def get(self, key: str) -> np.ndarray | None:
        v = self._store.get(key)
        if v is not None:
            self._hits += 1
        else:
            self._misses += 1
        return v

    def put(self, key: str, features: np.ndarray) -> None:
        if len(self._store) >= self._max and key not in self._store:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = features

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._store.clear()
        self._hits = self._misses = 0


@dataclass(frozen=True)
class DatasetStats:
    n_samples: int
    n_features: int
    mean: np.ndarray
    std: np.ndarray
    n_classes: int
    class_counts: dict[int, int]
    imbalance_ratio: float
    outlier_fraction: float


def compute_dataset_statistics(
    features: np.ndarray,
    labels: np.ndarray | None = None,
) -> DatasetStats:
    n_samples = features.shape[0]
    n_features = features.shape[1] if features.ndim > 1 else 1
    mean = features.mean(axis=0)
    std = features.std(axis=0)

    n_classes = 0
    class_counts: dict[int, int] = {}
    imbalance = 1.0
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique)
        class_counts = {int(u): int(c) for u, c in zip(unique, counts)}
        imbalance = float(counts.max() / max(counts.min(), 1))

    flat = features.flatten()
    q1, q3 = np.percentile(flat, [25, 75])
    iqr = q3 - q1
    outlier_mask = (flat < q1 - 1.5 * iqr) | (flat > q3 + 1.5 * iqr)
    outlier_frac = float(outlier_mask.mean())

    return DatasetStats(
        n_samples=n_samples,
        n_features=n_features,
        mean=mean,
        std=std,
        n_classes=n_classes,
        class_counts=class_counts,
        imbalance_ratio=imbalance,
        outlier_fraction=outlier_frac,
    )


def balance_classes(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "oversample",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    if method == "oversample":
        target = int(counts.max())
        all_feats, all_labels = [], []
        for cls, cnt in zip(unique, counts):
            mask = labels == cls
            cls_feats = features[mask]
            if cnt < target:
                extra_idx = rng.choice(cnt, target - cnt, replace=True)
                cls_feats = np.vstack([cls_feats, cls_feats[extra_idx]])
            all_feats.append(cls_feats)
            all_labels.extend([cls] * len(cls_feats))
        return np.vstack(all_feats), np.array(all_labels)
    elif method == "undersample":
        target = int(counts.min())
        all_feats, all_labels = [], []
        for cls in unique:
            mask = labels == cls
            cls_feats = features[mask]
            idx = rng.choice(len(cls_feats), target, replace=False)
            all_feats.append(cls_feats[idx])
            all_labels.extend([cls] * target)
        return np.vstack(all_feats), np.array(all_labels)
    return features.copy(), labels.copy()


def normalize_features(
    features: np.ndarray,
    method: str = "standard",
    stats: DatasetStats | None = None,
) -> np.ndarray:
    if method == "standard":
        mean = stats.mean if stats else features.mean(axis=0)
        std = stats.std if stats else features.std(axis=0)
        std = np.where(std < 1e-15, 1.0, std)
        return (features - mean) / std
    elif method == "minmax":
        fmin = features.min(axis=0)
        fmax = features.max(axis=0)
        rng = fmax - fmin
        rng = np.where(rng < 1e-15, 1.0, rng)
        return (features - fmin) / rng
    return features.copy()
