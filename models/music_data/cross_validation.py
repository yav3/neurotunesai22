"""Stratified cross-validation for music datasets.

Genre-aware and artist-aware splits that prevent data leakage.
Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class CVFold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    fold_id: int


@dataclass
class CVResult:
    folds: list[CVFold]
    n_folds: int
    strategy: str
    label_distribution: dict[int, list[float]]


def stratified_kfold(
    labels: np.ndarray, n_folds: int = 5, seed: int = 42,
) -> list[CVFold]:
    """Stratified K-fold: each fold preserves label proportions."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    unique_labels = np.unique(labels)
    fold_indices: list[list[int]] = [[] for _ in range(n_folds)]

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            fold_indices[i % n_folds].append(ix)

    folds: list[CVFold] = []
    all_idx = np.arange(n)
    for k in range(n_folds):
        val_idx = np.array(sorted(fold_indices[k]))
        train_idx = np.setdiff1d(all_idx, val_idx)
        folds.append(CVFold(train_idx=train_idx, val_idx=val_idx, fold_id=k))
    return folds


def group_kfold(
    groups: np.ndarray, n_folds: int = 5, seed: int = 42,
) -> list[CVFold]:
    """Group K-fold: no group (e.g. artist) appears in both train and val."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n = len(groups)
    all_idx = np.arange(n)

    group_to_fold = {}
    for i, g in enumerate(unique_groups):
        group_to_fold[g] = i % n_folds

    folds: list[CVFold] = []
    for k in range(n_folds):
        val_mask = np.array([group_to_fold[g] == k for g in groups])
        val_idx = all_idx[val_mask]
        train_idx = all_idx[~val_mask]
        folds.append(CVFold(train_idx=train_idx, val_idx=val_idx, fold_id=k))
    return folds


def temporal_split(
    timestamps: np.ndarray, n_folds: int = 5,
) -> list[CVFold]:
    """Expanding-window temporal split for music release-date ordering."""
    sorted_idx = np.argsort(timestamps)
    n = len(timestamps)
    chunk = n // (n_folds + 1)

    folds: list[CVFold] = []
    for k in range(n_folds):
        train_end = chunk * (k + 1)
        val_start = train_end
        val_end = min(train_end + chunk, n)
        train_idx = sorted_idx[:train_end]
        val_idx = sorted_idx[val_start:val_end]
        folds.append(CVFold(train_idx=train_idx, val_idx=val_idx, fold_id=k))
    return folds


def label_distribution_per_fold(
    labels: np.ndarray, folds: list[CVFold],
) -> dict[int, list[float]]:
    """Return per-fold label proportions {label: [fold0_prop, fold1_prop, ...]}."""
    unique = np.unique(labels)
    dist: dict[int, list[float]] = {int(u): [] for u in unique}
    for fold in folds:
        n_val = len(fold.val_idx)
        val_labels = labels[fold.val_idx]
        for u in unique:
            count = np.sum(val_labels == u)
            dist[int(u)].append(float(count / max(n_val, 1)))
    return dist


def leakage_check(
    groups: np.ndarray, folds: list[CVFold],
) -> list[tuple[int, int, set]]:
    """Detect group leakage across folds. Returns (fold_a, fold_b, shared_groups)."""
    violations: list[tuple[int, int, set]] = []
    for i, fa in enumerate(folds):
        train_groups = set(groups[fa.train_idx])
        val_groups = set(groups[fa.val_idx])
        overlap = train_groups & val_groups
        if overlap:
            violations.append((fa.fold_id, fa.fold_id, overlap))
    return violations
