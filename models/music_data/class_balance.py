"""Class imbalance handling for music classification tasks.

SMOTE-like oversampling, undersampling, class weighting. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BalanceStrategy(Enum):
    OVERSAMPLE = "oversample"
    UNDERSAMPLE = "undersample"
    SMOTE = "smote"
    CLASS_WEIGHT = "class_weight"


@dataclass
class BalanceResult:
    X: np.ndarray
    y: np.ndarray
    strategy: BalanceStrategy
    original_distribution: dict[int, int]
    balanced_distribution: dict[int, int]


def class_distribution(y: np.ndarray) -> dict[int, int]:
    """Count samples per class."""
    unique, counts = np.unique(y, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency class weights: n_total / (n_classes * n_class)."""
    dist = class_distribution(y)
    n = len(y)
    n_classes = len(dist)
    return {c: n / (n_classes * count) for c, count in dist.items()}


def random_oversample(
    X: np.ndarray, y: np.ndarray, seed: int = 42,
) -> BalanceResult:
    """Random oversampling: duplicate minority samples to match majority count."""
    rng = np.random.RandomState(seed)
    dist = class_distribution(y)
    max_count = max(dist.values())
    orig_dist = dict(dist)

    all_X = [X]
    all_y = [y]
    for c, count in dist.items():
        if count < max_count:
            idx = np.where(y == c)[0]
            extra = max_count - count
            sampled = rng.choice(idx, size=extra, replace=True)
            all_X.append(X[sampled])
            all_y.append(np.full(extra, c))

    X_bal = np.vstack(all_X)
    y_bal = np.concatenate(all_y)
    return BalanceResult(X_bal, y_bal, BalanceStrategy.OVERSAMPLE, orig_dist, class_distribution(y_bal))


def random_undersample(
    X: np.ndarray, y: np.ndarray, seed: int = 42,
) -> BalanceResult:
    """Random undersampling: reduce majority to match minority count."""
    rng = np.random.RandomState(seed)
    dist = class_distribution(y)
    min_count = min(dist.values())
    orig_dist = dict(dist)

    indices: list[np.ndarray] = []
    for c in dist:
        idx = np.where(y == c)[0]
        sampled = rng.choice(idx, size=min_count, replace=False)
        indices.append(sampled)

    sel = np.concatenate(indices)
    X_bal = X[sel]
    y_bal = y[sel]
    return BalanceResult(X_bal, y_bal, BalanceStrategy.UNDERSAMPLE, orig_dist, class_distribution(y_bal))


def smote_oversample(
    X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42,
) -> BalanceResult:
    """SMOTE: generate synthetic minority samples via k-NN interpolation."""
    rng = np.random.RandomState(seed)
    dist = class_distribution(y)
    max_count = max(dist.values())
    orig_dist = dict(dist)

    all_X = [X.copy()]
    all_y = [y.copy()]

    for c, count in dist.items():
        if count >= max_count:
            continue
        idx = np.where(y == c)[0]
        minority_X = X[idx]
        n_minority = len(idx)
        n_synthetic = max_count - count

        k_eff = min(k, n_minority - 1)
        if k_eff < 1:
            sampled = rng.choice(idx, size=n_synthetic, replace=True)
            all_X.append(X[sampled])
            all_y.append(np.full(n_synthetic, c))
            continue

        from scipy.spatial import cKDTree
        tree = cKDTree(minority_X)
        synthetic = np.zeros((n_synthetic, X.shape[1]))
        for i in range(n_synthetic):
            ref = rng.randint(n_minority)
            _, nn_idx = tree.query(minority_X[ref], k=k_eff + 1)
            nn_pick = nn_idx[rng.randint(1, k_eff + 1)]
            lam = rng.random()
            synthetic[i] = minority_X[ref] + lam * (minority_X[nn_pick] - minority_X[ref])

        all_X.append(synthetic)
        all_y.append(np.full(n_synthetic, c))

    X_bal = np.vstack(all_X)
    y_bal = np.concatenate(all_y)
    return BalanceResult(X_bal, y_bal, BalanceStrategy.SMOTE, orig_dist, class_distribution(y_bal))


def imbalance_ratio(y: np.ndarray) -> float:
    """Ratio of majority to minority class size. 1.0 = perfectly balanced."""
    dist = class_distribution(y)
    counts = list(dist.values())
    return max(counts) / max(min(counts), 1)


def effective_number_weights(
    y: np.ndarray, beta: float = 0.999,
) -> dict[int, float]:
    """Class-balanced loss weights via effective number of samples (Cui et al. 2019)."""
    dist = class_distribution(y)
    weights = {}
    for c, n_c in dist.items():
        e_n = 1.0 - beta ** n_c
        weights[c] = (1.0 - beta) / max(e_n, 1e-10)
    total = sum(weights.values())
    return {c: w * len(dist) / total for c, w in weights.items()}
