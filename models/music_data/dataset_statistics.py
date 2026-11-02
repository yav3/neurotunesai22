"""Dataset statistics for music feature collections.

Distribution analysis, class balance metrics, feature correlation,
and summary statistics for music datasets.  Pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
from scipy import stats as sp_stats


class DistributionType(str, Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    BIMODAL = "bimodal"
    HEAVY_TAILED = "heavy_tailed"
    UNKNOWN = "unknown"


@dataclass
class FeatureStats:
    """Summary statistics for a single feature column."""
    name: str
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    n_missing: int
    n_unique: int
    distribution: DistributionType


@dataclass
class ClassBalance:
    """Class distribution metrics for a labelled dataset."""
    labels: list[str]
    counts: list[int]
    proportions: list[float]
    entropy: float
    imbalance_ratio: float
    is_balanced: bool


@dataclass
class CorrelationReport:
    """Pairwise feature correlation analysis."""
    feature_names: list[str]
    pearson: np.ndarray
    spearman: np.ndarray
    highly_correlated_pairs: list[tuple[str, str, float]]
    redundant_features: list[str]


@dataclass
class DatasetSummary:
    """Aggregate dataset statistics."""
    n_samples: int
    n_features: int
    feature_stats: list[FeatureStats]
    class_balance: ClassBalance | None
    correlation: CorrelationReport | None
    completeness: float
    effective_dimensionality: int


def compute_feature_stats(values: np.ndarray, name: str = "feature") -> FeatureStats:
    """Compute summary statistics for one feature vector."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return FeatureStats(
            name=name, mean=0.0, std=0.0, median=0.0,
            q25=0.0, q75=0.0, iqr=0.0, skewness=0.0, kurtosis=0.0,
            min_val=0.0, max_val=0.0, n_missing=len(values), n_unique=0,
            distribution=DistributionType.UNKNOWN,
        )
    q25, median, q75 = np.percentile(finite, [25, 50, 75])
    return FeatureStats(
        name=name,
        mean=float(np.mean(finite)),
        std=float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        median=float(median),
        q25=float(q25),
        q75=float(q75),
        iqr=float(q75 - q25),
        skewness=float(sp_stats.skew(finite)) if len(finite) > 2 else 0.0,
        kurtosis=float(sp_stats.kurtosis(finite)) if len(finite) > 3 else 0.0,
        min_val=float(np.min(finite)),
        max_val=float(np.max(finite)),
        n_missing=int(len(values) - len(finite)),
        n_unique=int(len(np.unique(finite))),
        distribution=_classify_distribution(finite),
    )


def _classify_distribution(x: np.ndarray) -> DistributionType:
    """Heuristic distribution classification via normality + shape tests."""
    if len(x) < 8:
        return DistributionType.UNKNOWN
    _, p_normal = sp_stats.shapiro(x[:5000]) if len(x) > 5000 else sp_stats.shapiro(x)
    if p_normal > 0.05:
        return DistributionType.NORMAL
    kurt = float(sp_stats.kurtosis(x))
    skew = float(sp_stats.skew(x))
    if kurt > 3.0:
        return DistributionType.HEAVY_TAILED
    if abs(skew) > 1.5 and np.all(x > 0):
        return DistributionType.LOGNORMAL
    hist, _ = np.histogram(x, bins=min(50, len(x) // 5))
    peaks = _count_peaks(hist)
    if peaks >= 2:
        return DistributionType.BIMODAL
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    rng = float(np.max(x) - np.min(x))
    if rng > 0 and iqr / rng > 0.55:
        return DistributionType.UNIFORM
    return DistributionType.UNKNOWN


def _count_peaks(hist: np.ndarray) -> int:
    """Count local maxima in a histogram."""
    if len(hist) < 3:
        return 1
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks += 1
    return max(peaks, 1)


def class_balance(labels: Sequence[str]) -> ClassBalance:
    """Analyse label distribution for class imbalance."""
    unique, counts = np.unique(labels, return_counts=True)
    total = int(np.sum(counts))
    proportions = counts / total if total > 0 else counts.astype(float)
    entropy = float(-np.sum(proportions * np.log2(proportions + 1e-15)))
    max_entropy = float(np.log2(len(unique))) if len(unique) > 1 else 1.0
    ir = float(np.max(counts) / (np.min(counts) + 1e-15))
    balanced = ir < 3.0 and entropy / max_entropy > 0.8
    return ClassBalance(
        labels=[str(u) for u in unique],
        counts=[int(c) for c in counts],
        proportions=[float(p) for p in proportions],
        entropy=entropy,
        imbalance_ratio=ir,
        is_balanced=balanced,
    )


def feature_correlation(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = 0.9,
) -> CorrelationReport:
    """Compute pairwise Pearson and Spearman correlations, flag redundancy."""
    n_feat = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_feat)]
    pearson = np.corrcoef(X, rowvar=False)
    spearman, _ = sp_stats.spearmanr(X, axis=0)
    if n_feat == 1:
        spearman = np.array([[1.0]])
    high_pairs: list[tuple[str, str, float]] = []
    redundant: set[str] = set()
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            r = abs(float(pearson[i, j]))
            if r >= threshold:
                high_pairs.append((feature_names[i], feature_names[j], r))
                redundant.add(feature_names[j])
    return CorrelationReport(
        feature_names=list(feature_names),
        pearson=pearson,
        spearman=np.atleast_2d(spearman),
        highly_correlated_pairs=high_pairs,
        redundant_features=sorted(redundant),
    )


def effective_dimensionality(X: np.ndarray, threshold: float = 0.95) -> int:
    """Number of PCA components needed to explain *threshold* fraction of variance."""
    centered = X - np.mean(X, axis=0)
    cov = np.dot(centered.T, centered) / max(X.shape[0] - 1, 1)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    total = float(np.sum(eigvals))
    if total < 1e-15:
        return 0
    cumulative = np.cumsum(eigvals) / total
    idx = int(np.searchsorted(cumulative, threshold))
    return min(idx + 1, len(eigvals))


def dataset_summary(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    labels: Sequence[str] | None = None,
    correlation_threshold: float = 0.9,
) -> DatasetSummary:
    """Full dataset summary: per-feature stats, class balance, correlations."""
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    f_stats = [
        compute_feature_stats(X[:, i], feature_names[i])
        for i in range(n_features)
    ]
    cb = class_balance(labels) if labels is not None else None
    corr = feature_correlation(X, feature_names, correlation_threshold) if n_features > 1 else None
    total_cells = n_samples * n_features
    missing = sum(fs.n_missing for fs in f_stats)
    completeness = 1.0 - missing / total_cells if total_cells > 0 else 0.0
    eff_dim = effective_dimensionality(X) if n_features > 1 else 1
    return DatasetSummary(
        n_samples=n_samples,
        n_features=n_features,
        feature_stats=f_stats,
        class_balance=cb,
        correlation=corr,
        completeness=completeness,
        effective_dimensionality=eff_dim,
    )


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a summary statistic.

    Returns (estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    stat_fn = {"mean": np.mean, "median": np.median, "std": np.std}.get(statistic, np.mean)
    estimate = float(stat_fn(values))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = stat_fn(sample)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot, 100 * alpha))
    upper = float(np.percentile(boot, 100 * (1.0 - alpha)))
    return estimate, lower, upper


def mutual_information_matrix(X: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Estimate pairwise MI between features via histogram binning.

    Returns an (n_features, n_features) matrix of MI values in nats.
    """
    n_feat = X.shape[1]
    mi = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i, n_feat):
            hist_2d, _, _ = np.histogram2d(X[:, i], X[:, j], bins=n_bins)
            pxy = hist_2d / hist_2d.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            mask = pxy > 0
            mi_val = np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))
            mi[i, j] = mi_val
            mi[j, i] = mi_val
    return mi
