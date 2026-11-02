"""Stratified dataset splitting for music ML: label-aware, temporal, artist-aware splits."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SplitResult:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    label_distribution: dict[str, dict[str, float]]  # split -> label -> fraction


@dataclass
class FoldResult:
    folds: list[tuple[np.ndarray, np.ndarray]]  # (train_idx, val_idx) per fold
    n_folds: int


def stratified_split(labels: np.ndarray,
                     train_frac: float = 0.7,
                     val_frac: float = 0.15,
                     seed: int = 42) -> SplitResult:
    """Split data preserving label distribution in each partition.

    For multi-label tasks, pass the dominant label per sample.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    unique_labels = np.unique(labels)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for lab in unique_labels:
        mask = np.where(labels == lab)[0]
        rng.shuffle(mask)
        n_lab = len(mask)
        n_train = max(1, int(n_lab * train_frac))
        n_val = max(1, int(n_lab * val_frac))
        train_idx.extend(mask[:n_train].tolist())
        val_idx.extend(mask[n_train:n_train + n_val].tolist())
        test_idx.extend(mask[n_train + n_val:].tolist())

    def _dist(indices: list[int]) -> dict[str, float]:
        if not indices:
            return {}
        sub = labels[indices]
        unique, counts = np.unique(sub, return_counts=True)
        total = counts.sum()
        return {str(u): float(c / total) for u, c in zip(unique, counts)}

    return SplitResult(
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
        {"train": _dist(train_idx), "val": _dist(val_idx), "test": _dist(test_idx)},
    )


def artist_aware_split(labels: np.ndarray,
                       artist_ids: np.ndarray,
                       train_frac: float = 0.7,
                       val_frac: float = 0.15,
                       seed: int = 42) -> SplitResult:
    """Split ensuring no artist appears in both train and test.

    Prevents the model from memorising artist-specific production style
    rather than learning musical features. Artists are assigned to splits
    at the artist level, then samples follow their artist.
    """
    rng = np.random.RandomState(seed)
    unique_artists = np.unique(artist_ids)
    rng.shuffle(unique_artists)

    n_artists = len(unique_artists)
    n_train = int(n_artists * train_frac)
    n_val = int(n_artists * val_frac)

    train_artists = set(unique_artists[:n_train].tolist())
    val_artists = set(unique_artists[n_train:n_train + n_val].tolist())
    test_artists = set(unique_artists[n_train + n_val:].tolist())

    train_idx = [i for i, a in enumerate(artist_ids) if a in train_artists]
    val_idx = [i for i, a in enumerate(artist_ids) if a in val_artists]
    test_idx = [i for i, a in enumerate(artist_ids) if a in test_artists]

    def _dist(indices: list[int]) -> dict[str, float]:
        if not indices:
            return {}
        sub = labels[indices]
        unique, counts = np.unique(sub, return_counts=True)
        total = counts.sum()
        return {str(u): float(c / total) for u, c in zip(unique, counts)}

    return SplitResult(
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
        {"train": _dist(train_idx), "val": _dist(val_idx), "test": _dist(test_idx)},
    )


def temporal_split(timestamps: np.ndarray,
                   train_frac: float = 0.7,
                   val_frac: float = 0.15) -> SplitResult:
    """Split by time: train on earlier data, test on later.

    Prevents future-leakage in music trend / popularity prediction.
    """
    order = np.argsort(timestamps)
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    return SplitResult(train_idx, val_idx, test_idx, {})


def stratified_kfold(labels: np.ndarray,
                     n_folds: int = 5,
                     seed: int = 42) -> FoldResult:
    """Stratified k-fold cross-validation preserving label proportions."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    unique_labels = np.unique(labels)

    # Assign each sample to a fold, per-label
    fold_assignments = np.zeros(n, dtype=int)
    for lab in unique_labels:
        mask = np.where(labels == lab)[0]
        rng.shuffle(mask)
        for i, idx in enumerate(mask):
            fold_assignments[idx] = i % n_folds

    folds = []
    for f in range(n_folds):
        val_mask = fold_assignments == f
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        folds.append((train_idx, val_idx))

    return FoldResult(folds, n_folds)


def group_kfold(groups: np.ndarray, n_folds: int = 5,
                seed: int = 42) -> FoldResult:
    """Group k-fold: no group (e.g. album) spans train and val in any fold."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    group_to_fold: dict = {}
    for i, g in enumerate(unique_groups):
        group_to_fold[g] = i % n_folds

    fold_assignments = np.array([group_to_fold[g] for g in groups])

    folds = []
    for f in range(n_folds):
        val_mask = fold_assignments == f
        folds.append((np.where(~val_mask)[0], np.where(val_mask)[0]))

    return FoldResult(folds, n_folds)


def class_weight_balanced(labels: np.ndarray) -> dict:
    """Compute balanced class weights inversely proportional to class frequency.

    Useful for imbalanced genre/mood datasets.
    """
    unique, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    weights = {}
    for lab, cnt in zip(unique, counts):
        weights[lab] = n / (len(unique) * cnt)
    return weights


def split_quality_check(split: SplitResult,
                        labels: np.ndarray) -> dict[str, float]:
    """Check split quality: label balance, size ratios, overlap."""
    n = len(labels)
    train_set = set(split.train_indices.tolist())
    val_set = set(split.val_indices.tolist())
    test_set = set(split.test_indices.tolist())

    # Check overlap
    train_val_overlap = len(train_set & val_set)
    train_test_overlap = len(train_set & test_set)
    val_test_overlap = len(val_set & test_set)

    # Check coverage
    covered = len(train_set | val_set | test_set)

    # Jensen-Shannon divergence between train and test label distributions
    train_dist = split.label_distribution.get("train", {})
    test_dist = split.label_distribution.get("test", {})
    all_labels = sorted(set(list(train_dist.keys()) + list(test_dist.keys())))
    if all_labels:
        p = np.array([train_dist.get(l, 0) for l in all_labels]) + 1e-10
        q = np.array([test_dist.get(l, 0) for l in all_labels]) + 1e-10
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * float(np.sum(p * np.log2(p / m))) + 0.5 * float(np.sum(q * np.log2(q / m)))
    else:
        jsd = 0.0

    return {
        "coverage": covered / n,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "label_jsd_train_test": jsd,
        "train_frac": len(train_set) / n,
        "val_frac": len(val_set) / n,
        "test_frac": len(test_set) / n,
    }
