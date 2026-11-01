"""Feature engineering utilities for music datasets.

Normalization, imputation, feature selection, and derived feature
construction for music feature matrices.  Pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from scipy import stats as sp_stats


class NormMethod(str, Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"
    L2 = "l2"
    LOG = "log"
    QUANTILE = "quantile"


class ImputeStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"
    ITERATIVE = "iterative"
    ZERO = "zero"


@dataclass
class NormParams:
    """Fitted normalization parameters for reproducible transforms."""
    method: NormMethod
    shift: np.ndarray
    scale: np.ndarray


@dataclass
class FeatureImportance:
    """Feature importance scores from variance or MI analysis."""
    names: list[str]
    scores: np.ndarray
    ranking: list[int]
    selected: list[str]


def fit_normalize(X: np.ndarray, method: NormMethod = NormMethod.ZSCORE) -> NormParams:
    """Fit normalization parameters from training data."""
    if method == NormMethod.MINMAX:
        lo = np.nanmin(X, axis=0)
        hi = np.nanmax(X, axis=0)
        scale = hi - lo
        scale[scale < 1e-15] = 1.0
        return NormParams(method=method, shift=lo, scale=scale)
    if method == NormMethod.ZSCORE:
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0, ddof=1)
        sigma[sigma < 1e-15] = 1.0
        return NormParams(method=method, shift=mu, scale=sigma)
    if method == NormMethod.ROBUST:
        med = np.nanmedian(X, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-15] = 1.0
        return NormParams(method=method, shift=med, scale=iqr)
    if method == NormMethod.L2:
        norms = np.sqrt(np.nansum(X ** 2, axis=1, keepdims=True))
        norms[norms < 1e-15] = 1.0
        return NormParams(method=method, shift=np.zeros(X.shape[1]), scale=norms.ravel())
    if method == NormMethod.LOG:
        return NormParams(method=method, shift=np.zeros(X.shape[1]), scale=np.ones(X.shape[1]))
    if method == NormMethod.QUANTILE:
        ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
        n = X.shape[0]
        return NormParams(method=method, shift=np.zeros(X.shape[1]), scale=np.full(X.shape[1], n))
    return NormParams(method=method, shift=np.zeros(X.shape[1]), scale=np.ones(X.shape[1]))


def apply_normalize(X: np.ndarray, params: NormParams) -> np.ndarray:
    """Apply fitted normalization to data."""
    if params.method == NormMethod.MINMAX:
        return (X - params.shift) / params.scale
    if params.method == NormMethod.ZSCORE:
        return (X - params.shift) / params.scale
    if params.method == NormMethod.ROBUST:
        return (X - params.shift) / params.scale
    if params.method == NormMethod.L2:
        norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        norms[norms < 1e-15] = 1.0
        return X / norms
    if params.method == NormMethod.LOG:
        return np.log1p(np.maximum(X, 0.0))
    if params.method == NormMethod.QUANTILE:
        ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
        return ranks / max(X.shape[0] - 1, 1)
    return X.copy()


def impute(
    X: np.ndarray,
    strategy: ImputeStrategy = ImputeStrategy.MEDIAN,
    k: int = 5,
    max_iter: int = 10,
) -> np.ndarray:
    """Impute missing values (NaN) in a feature matrix."""
    result = X.copy()
    mask = np.isnan(result)
    if not mask.any():
        return result

    if strategy == ImputeStrategy.ZERO:
        result[mask] = 0.0
        return result

    if strategy == ImputeStrategy.MEAN:
        col_vals = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            result[mask[:, j], j] = col_vals[j]
        return result

    if strategy == ImputeStrategy.MEDIAN:
        col_vals = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            result[mask[:, j], j] = col_vals[j]
        return result

    if strategy == ImputeStrategy.KNN:
        result = _knn_impute(X, k)
        return result

    if strategy == ImputeStrategy.ITERATIVE:
        result = _iterative_impute(X, max_iter)
        return result

    return result


def _knn_impute(X: np.ndarray, k: int) -> np.ndarray:
    """K-nearest-neighbour imputation using available features."""
    result = X.copy()
    mask = np.isnan(result)
    col_medians = np.nanmedian(X, axis=0)
    X_filled = X.copy()
    for j in range(X.shape[1]):
        X_filled[mask[:, j], j] = col_medians[j]

    for i in range(X.shape[0]):
        missing_cols = np.where(mask[i])[0]
        if len(missing_cols) == 0:
            continue
        present_cols = np.where(~mask[i])[0]
        if len(present_cols) == 0:
            result[i, missing_cols] = col_medians[missing_cols]
            continue
        dists = np.sqrt(np.sum((X_filled[:, present_cols] - X_filled[i, present_cols]) ** 2, axis=1))
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:k]
        for j in missing_cols:
            nn_vals = X[nn_idx, j]
            valid = nn_vals[~np.isnan(nn_vals)]
            result[i, j] = float(np.mean(valid)) if len(valid) > 0 else col_medians[j]
    return result


def _iterative_impute(X: np.ndarray, max_iter: int) -> np.ndarray:
    """Round-robin iterative imputation (simplified MICE)."""
    result = X.copy()
    mask = np.isnan(result)
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        result[mask[:, j], j] = col_medians[j]

    for _ in range(max_iter):
        for j in range(X.shape[1]):
            missing_rows = np.where(mask[:, j])[0]
            if len(missing_rows) == 0:
                continue
            present_rows = np.where(~mask[:, j])[0]
            if len(present_rows) < 2:
                continue
            other_cols = [c for c in range(X.shape[1]) if c != j]
            if not other_cols:
                continue
            A = result[present_rows][:, other_cols]
            b = result[present_rows, j]
            AtA = A.T @ A + 1e-6 * np.eye(len(other_cols))
            try:
                beta = np.linalg.solve(AtA, A.T @ b)
            except np.linalg.LinAlgError:
                continue
            predictions = result[missing_rows][:, other_cols] @ beta
            result[missing_rows, j] = predictions
    return result


def variance_threshold(
    X: np.ndarray,
    threshold: float = 0.01,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Remove features with variance below threshold."""
    variances = np.nanvar(X, axis=0)
    keep = variances >= threshold
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    kept_names = [feature_names[i] for i in range(len(feature_names)) if keep[i]]
    return X[:, keep], kept_names


def mutual_info_classif(
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
    feature_names: list[str] | None = None,
    top_k: int | None = None,
) -> FeatureImportance:
    """Feature importance via mutual information with class labels."""
    n_feat = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_feat)]
    scores = np.zeros(n_feat)
    classes = np.unique(y)
    py = np.array([np.mean(y == c) for c in classes])
    for j in range(n_feat):
        xj = X[:, j]
        finite = np.isfinite(xj)
        if finite.sum() < 10:
            continue
        xf = xj[finite]
        yf = y[finite]
        bins = np.linspace(np.min(xf), np.max(xf), n_bins + 1)
        xd = np.digitize(xf, bins[:-1]) - 1
        xd = np.clip(xd, 0, n_bins - 1)
        n = len(xf)
        mi = 0.0
        for xi in range(n_bins):
            mask_x = xd == xi
            px = mask_x.sum() / n
            if px < 1e-15:
                continue
            for ci, c in enumerate(classes):
                mask_y = yf == c
                pxy = (mask_x & mask_y).sum() / n
                if pxy < 1e-15:
                    continue
                mi += pxy * np.log(pxy / (px * py[ci]))
        scores[j] = max(mi, 0.0)
    ranking = list(np.argsort(scores)[::-1])
    k = top_k if top_k is not None else n_feat
    selected = [feature_names[i] for i in ranking[:k] if scores[i] > 0]
    return FeatureImportance(
        names=feature_names,
        scores=scores,
        ranking=ranking,
        selected=selected,
    )


def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Generate polynomial interaction features up to given degree.

    For degree=2 on [a, b]: returns [a, b, a², ab, b²].
    """
    if degree < 1:
        return X.copy()
    result = [X]
    if degree >= 2:
        n = X.shape[1]
        interactions = []
        for i in range(n):
            for j in range(i, n):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if interactions:
            result.append(np.hstack(interactions))
    return np.hstack(result)


def temporal_features(
    values: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Derive temporal features from a 1D time series.

    Returns: [rolling_mean, rolling_std, delta, delta2, ema].
    """
    n = len(values)
    rm = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = values[i - window + 1:i + 1]
        rm[i] = np.mean(seg)
        rs[i] = np.std(seg, ddof=1) if window > 1 else 0.0
    delta = np.zeros(n)
    delta[1:] = np.diff(values)
    delta2 = np.zeros(n)
    delta2[1:] = np.diff(delta)
    alpha = 2.0 / (window + 1)
    ema = np.zeros(n)
    ema[0] = values[0]
    for i in range(1, n):
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
    return np.column_stack([rm, rs, delta, delta2, ema])
