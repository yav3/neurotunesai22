"""Feature fusion, selection, compression, and temporal pooling.

Implements early/late fusion strategies, attention-weighted aggregation,
mutual-information feature selection, PCA/SVD compression, multi-resolution
extraction, and permutation-based feature importance — all pure numpy/scipy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable


# ---------------------------------------------------------------------------
# Early fusion
# ---------------------------------------------------------------------------

def zscore_normalize(X: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise along *axis*. Returns (X_norm, mu, sigma)."""
    mu = np.mean(X, axis=axis, keepdims=True)
    sigma = np.std(X, axis=axis, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma, np.squeeze(mu), np.squeeze(sigma)


def minmax_normalize(X: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalise to [0, 1]. Returns (X_norm, mins, maxs)."""
    lo = np.min(X, axis=axis, keepdims=True)
    hi = np.max(X, axis=axis, keepdims=True)
    rng = hi - lo
    rng = np.where(rng < 1e-12, 1.0, rng)
    return (X - lo) / rng, np.squeeze(lo), np.squeeze(hi)


def early_fusion(feature_groups: List[np.ndarray],
                 normalize: str = "zscore") -> np.ndarray:
    """Concatenate feature groups along axis-1 after per-group normalisation.

    Parameters
    ----------
    feature_groups : list of (n_samples, d_i) arrays
    normalize : "zscore" | "minmax" | "none"

    Returns
    -------
    (n_samples, sum(d_i)) fused matrix
    """
    normed = []
    for g in feature_groups:
        if g.ndim == 1:
            g = g.reshape(-1, 1)
        if normalize == "zscore":
            g, _, _ = zscore_normalize(g)
        elif normalize == "minmax":
            g, _, _ = minmax_normalize(g)
        normed.append(g)
    return np.hstack(normed)


# ---------------------------------------------------------------------------
# Late fusion
# ---------------------------------------------------------------------------

def late_fusion_scores(score_vectors: List[np.ndarray],
                       weights: Optional[np.ndarray] = None,
                       method: str = "weighted_mean") -> np.ndarray:
    """Combine per-system score vectors into a single fused score.

    Parameters
    ----------
    score_vectors : list of (n_samples,) arrays — one per subsystem
    weights : optional (n_systems,) weight vector; uniform if None
    method : "weighted_mean" | "max" | "product" | "rank_mean"
    """
    S = np.column_stack(score_vectors)  # (n, k)
    k = S.shape[1]
    if weights is None:
        weights = np.ones(k) / k
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / (np.sum(weights) + 1e-15)

    if method == "weighted_mean":
        return S @ weights
    elif method == "max":
        return np.max(S, axis=1)
    elif method == "product":
        return np.prod(S ** weights, axis=1)
    elif method == "rank_mean":
        ranks = np.zeros_like(S)
        for j in range(k):
            order = np.argsort(np.argsort(S[:, j]))
            ranks[:, j] = order / max(len(order) - 1, 1)
        return ranks @ weights
    else:
        raise ValueError(f"Unknown late-fusion method: {method}")


def learn_fusion_weights(score_vectors: List[np.ndarray],
                         labels: np.ndarray,
                         lr: float = 0.01, epochs: int = 200) -> np.ndarray:
    """Learn late-fusion weights via gradient descent on MSE against labels."""
    S = np.column_stack(score_vectors)
    k = S.shape[1]
    w = np.ones(k) / k
    for _ in range(epochs):
        pred = S @ w
        err = pred - labels
        grad = (2 / len(labels)) * S.T @ err
        w -= lr * grad
        w = np.maximum(w, 0)
        s = np.sum(w)
        if s > 1e-12:
            w /= s
    return w


# ---------------------------------------------------------------------------
# Attention-weighted aggregation
# ---------------------------------------------------------------------------

def attention_weights(Q: np.ndarray, K: np.ndarray,
                      temperature: float = 1.0) -> np.ndarray:
    """Scaled dot-product attention weights (no value projection).

    Parameters
    ----------
    Q : (n, d) query matrix
    K : (n, d) key matrix
    temperature : scaling denominator

    Returns
    -------
    (n, n) attention matrix (rows sum to 1)
    """
    d = Q.shape[1]
    scale = np.sqrt(d) * temperature
    scores = Q @ K.T / max(scale, 1e-10)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_s = np.exp(scores)
    return exp_s / (np.sum(exp_s, axis=1, keepdims=True) + 1e-15)


def attention_aggregate(features: np.ndarray,
                        temperature: float = 1.0) -> np.ndarray:
    """Self-attention–weighted mean of time frames.

    Parameters
    ----------
    features : (T, d) frame-level features

    Returns
    -------
    (d,) aggregated vector
    """
    A = attention_weights(features, features, temperature)
    weighted = A @ features  # (T, d)
    return np.mean(weighted, axis=0)


def multi_head_attention_aggregate(features: np.ndarray, n_heads: int = 4,
                                   temperature: float = 1.0) -> np.ndarray:
    """Multi-head self-attention aggregation."""
    T, d = features.shape
    head_dim = max(d // n_heads, 1)
    heads = []
    for h in range(n_heads):
        start = h * head_dim
        end = min(start + head_dim, d)
        if start >= d:
            break
        sub = features[:, start:end]
        heads.append(attention_aggregate(sub, temperature))
    return np.concatenate(heads)


# ---------------------------------------------------------------------------
# Feature selection — mutual information
# ---------------------------------------------------------------------------

def _entropy(x: np.ndarray, bins: int = 30) -> float:
    counts, _ = np.histogram(x, bins=bins)
    p = counts / (np.sum(counts) + 1e-15)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-30)))


def _joint_entropy(x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:
    H, _, _ = np.histogram2d(x, y, bins=bins)
    p = H / (np.sum(H) + 1e-15)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-30)))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:
    """Estimate MI(X; Y) via histogram binning."""
    return _entropy(x, bins) + _entropy(y, bins) - _joint_entropy(x, y, bins)


def mi_feature_ranking(X: np.ndarray, y: np.ndarray,
                       bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Rank features by MI with target. Returns (sorted_indices, mi_scores)."""
    d = X.shape[1]
    scores = np.array([mutual_information(X[:, j], y, bins) for j in range(d)])
    order = np.argsort(scores)[::-1]
    return order, scores[order]


def select_top_k(X: np.ndarray, y: np.ndarray, k: int,
                 bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Select top-k features by MI. Returns (X_selected, indices)."""
    order, _ = mi_feature_ranking(X, y, bins)
    idx = order[:k]
    return X[:, idx], idx


# ---------------------------------------------------------------------------
# PCA / SVD compression
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    components: np.ndarray        # (k, d) principal axes
    singular_values: np.ndarray   # (k,)
    explained_variance: np.ndarray  # (k,) fraction
    mean: np.ndarray              # (d,) column means

    @property
    def total_explained(self) -> float:
        return float(np.sum(self.explained_variance))


def pca_compress(X: np.ndarray, k: int) -> Tuple[np.ndarray, PCAResult]:
    """Truncated SVD-based PCA. Returns (X_reduced, PCAResult)."""
    mu = np.mean(X, axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = np.sum(S ** 2)
    k = min(k, len(S))
    components = Vt[:k]
    svals = S[:k]
    explained = (svals ** 2) / max(total_var, 1e-15)
    X_reduced = Xc @ components.T
    result = PCAResult(components, svals, explained, mu)
    return X_reduced, result


def pca_reconstruct(X_reduced: np.ndarray, pca: PCAResult) -> np.ndarray:
    """Reconstruct from PCA-compressed representation."""
    return X_reduced @ pca.components + pca.mean


def auto_k_pca(X: np.ndarray, target_variance: float = 0.95) -> Tuple[np.ndarray, PCAResult]:
    """Choose k to reach target cumulative explained variance."""
    mu = np.mean(X, axis=0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = np.sum(S ** 2)
    cumvar = np.cumsum(S ** 2) / max(total_var, 1e-15)
    k = int(np.searchsorted(cumvar, target_variance) + 1)
    k = min(k, len(S))
    return pca_compress(X, k)


# ---------------------------------------------------------------------------
# Temporal pooling
# ---------------------------------------------------------------------------

def temporal_pool_stats(frames: np.ndarray) -> np.ndarray:
    """Pool frame-level features into a single vector via statistics.

    Concatenates [mean, std, min, max, median, skew, kurtosis] per dimension.
    """
    mu = np.mean(frames, axis=0)
    std = np.std(frames, axis=0)
    mn = np.min(frames, axis=0)
    mx = np.max(frames, axis=0)
    med = np.median(frames, axis=0)
    centered = frames - mu
    m3 = np.mean(centered ** 3, axis=0)
    m4 = np.mean(centered ** 4, axis=0)
    safe_std = np.where(std < 1e-12, 1.0, std)
    skew = m3 / (safe_std ** 3)
    kurt = m4 / (safe_std ** 4) - 3.0
    return np.concatenate([mu, std, mn, mx, med, skew, kurt])


def temporal_pool_mean(frames: np.ndarray) -> np.ndarray:
    return np.mean(frames, axis=0)


def temporal_pool_max(frames: np.ndarray) -> np.ndarray:
    return np.max(frames, axis=0)


def temporal_pool_attention(frames: np.ndarray,
                            temperature: float = 1.0) -> np.ndarray:
    """Attention-weighted temporal pooling."""
    return attention_aggregate(frames, temperature)


# ---------------------------------------------------------------------------
# Multi-resolution extraction
# ---------------------------------------------------------------------------

def multi_resolution_extract(signal: np.ndarray, sr: int,
                             frame_fn: Callable[[np.ndarray], np.ndarray],
                             window_sizes: List[int],
                             hop_ratio: float = 0.5) -> np.ndarray:
    """Extract features at multiple window sizes and concatenate.

    Parameters
    ----------
    signal : (n_samples,) audio
    sr : sample rate
    frame_fn : function mapping (window,) → (d,) feature vector
    window_sizes : list of window lengths in samples
    hop_ratio : hop as fraction of window

    Returns
    -------
    (d_total,) concatenation of pooled features from each resolution
    """
    parts = []
    for ws in window_sizes:
        hop = max(1, int(ws * hop_ratio))
        frames = []
        for start in range(0, len(signal) - ws + 1, hop):
            chunk = signal[start:start + ws]
            frames.append(frame_fn(chunk))
        if frames:
            frame_mat = np.array(frames)
            parts.append(temporal_pool_stats(frame_mat))
        else:
            d = len(frame_fn(np.zeros(ws)))
            parts.append(np.zeros(d * 7))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Feature importance — permutation scoring
# ---------------------------------------------------------------------------

def permutation_importance(X: np.ndarray, y: np.ndarray,
                           predict_fn: Callable[[np.ndarray], np.ndarray],
                           metric_fn: Callable[[np.ndarray, np.ndarray], float],
                           n_repeats: int = 5,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Permutation-based feature importance.

    Parameters
    ----------
    X : (n, d) feature matrix
    y : (n,) labels / targets
    predict_fn : X → predictions
    metric_fn : (y_true, y_pred) → scalar (higher = better)
    n_repeats : number of permutation rounds per feature

    Returns
    -------
    (importances_mean, importances_std) each of shape (d,)
    """
    rng = np.random.RandomState(seed)
    baseline = metric_fn(y, predict_fn(X))
    d = X.shape[1]
    drops = np.zeros((d, n_repeats))
    for j in range(d):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            score = metric_fn(y, predict_fn(X_perm))
            drops[j, r] = baseline - score
    return np.mean(drops, axis=1), np.std(drops, axis=1)


# ---------------------------------------------------------------------------
# Convenience: full fusion pipeline
# ---------------------------------------------------------------------------

@dataclass
class FusionPipeline:
    """Configurable fusion pipeline: normalise → select → compress → pool."""
    normalize: str = "zscore"
    select_k: Optional[int] = None
    pca_k: Optional[int] = None
    pca_variance: Optional[float] = None
    pool_method: str = "stats"

    def fit_transform(self, feature_groups: List[np.ndarray],
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        X = early_fusion(feature_groups, self.normalize)
        if self.select_k is not None and y is not None:
            X, _ = select_top_k(X, y, self.select_k)
        if self.pca_variance is not None:
            X, self._pca = auto_k_pca(X, self.pca_variance)
        elif self.pca_k is not None:
            X, self._pca = pca_compress(X, self.pca_k)
        return X

    def pool(self, frame_features: np.ndarray) -> np.ndarray:
        if self.pool_method == "stats":
            return temporal_pool_stats(frame_features)
        elif self.pool_method == "mean":
            return temporal_pool_mean(frame_features)
        elif self.pool_method == "max":
            return temporal_pool_max(frame_features)
        elif self.pool_method == "attention":
            return temporal_pool_attention(frame_features)
        return temporal_pool_stats(frame_features)
