"""Learned music representations: contrastive, masked prediction, multi-scale.

Implements core representation learning objectives for the foundation model:
- Multi-scale temporal pooling (frame → segment → track)
- Codebook quantization with EMA updates (van den Oord et al., 2017)
- Masked acoustic modeling (Baevski et al., 2020; Hsu et al., 2021)
- Barlow Twins redundancy reduction (Zbontar et al., 2021)
- Variance-Invariance-Covariance regularization (VICReg, Bardes et al., 2022)
- Hierarchical feature pyramids for multi-resolution music understanding

Pure numpy. No external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PoolingConfig:
    frame_dim: int = 768
    segment_frames: int = 32
    n_codebook: int = 512
    codebook_dim: int = 64
    ema_decay: float = 0.99
    commitment_cost: float = 0.25
    mask_prob: float = 0.065
    mask_length: int = 10
    temperature: float = 0.1


@dataclass
class RepresentationOutput:
    frame_embeddings: np.ndarray
    segment_embeddings: np.ndarray
    track_embedding: np.ndarray
    codebook_indices: Optional[np.ndarray] = None
    quantization_loss: float = 0.0


# ---------- multi-scale temporal pooling ----------

def mean_pool(features: np.ndarray, segment_size: int) -> np.ndarray:
    """Pool frame-level features into segment-level via non-overlapping mean.

    Args:
        features: (T, D) frame embeddings
        segment_size: frames per segment

    Returns:
        (n_segments, D) segment embeddings
    """
    T, D = features.shape
    n_seg = max(1, T // segment_size)
    trimmed = features[:n_seg * segment_size]
    return trimmed.reshape(n_seg, segment_size, D).mean(axis=1)


def attention_pool(features: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Attention-weighted pooling over time axis.

    Args:
        features: (T, D)
        query: (D,) learned query vector

    Returns:
        (D,) weighted sum
    """
    scores = features @ query  # (T,)
    weights = _softmax(scores)
    return weights @ features  # (D,)


def multi_scale_pool(
    features: np.ndarray,
    segment_size: int = 32,
    query: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-level pooling: frame → segment → track.

    Returns:
        (frame_emb, segment_emb, track_emb)
    """
    frame_emb = features
    segment_emb = mean_pool(features, segment_size)

    if query is not None:
        track_emb = attention_pool(features, query)
    else:
        track_emb = features.mean(axis=0)

    return frame_emb, segment_emb, track_emb


# ---------- codebook quantization (VQ-VAE style) ----------

@dataclass
class Codebook:
    embeddings: np.ndarray  # (K, D)
    usage_count: np.ndarray  # (K,)
    ema_cluster_size: np.ndarray  # (K,)
    ema_sum: np.ndarray  # (K, D)


def init_codebook(n_codes: int, dim: int, seed: int = 42) -> Codebook:
    """Initialize codebook with small random vectors."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(n_codes, dim).astype(np.float64) * 0.01
    return Codebook(
        embeddings=emb,
        usage_count=np.zeros(n_codes),
        ema_cluster_size=np.ones(n_codes),
        ema_sum=emb.copy(),
    )


def quantize(z: np.ndarray, codebook: Codebook) -> Tuple[np.ndarray, np.ndarray, float]:
    """Vector quantization: find nearest codebook entry per frame.

    Args:
        z: (T, D) continuous representations
        codebook: current codebook state

    Returns:
        (quantized, indices, commitment_loss)
    """
    # (T, K) distance matrix
    dists = (
        np.sum(z ** 2, axis=1, keepdims=True)
        - 2 * z @ codebook.embeddings.T
        + np.sum(codebook.embeddings ** 2, axis=1)
    )
    indices = np.argmin(dists, axis=1)  # (T,)
    quantized = codebook.embeddings[indices]  # (T, D)

    commitment_loss = np.mean((z - quantized) ** 2)
    return quantized, indices, commitment_loss


def update_codebook_ema(
    codebook: Codebook,
    z: np.ndarray,
    indices: np.ndarray,
    decay: float = 0.99,
    epsilon: float = 1e-5,
) -> Codebook:
    """Exponential moving average codebook update (van den Oord et al., 2017).

    Straight-through estimator: gradients pass through quantization.
    """
    K, D = codebook.embeddings.shape
    T = z.shape[0]

    one_hot = np.zeros((T, K))
    one_hot[np.arange(T), indices] = 1.0

    cluster_size = one_hot.sum(axis=0)  # (K,)
    cluster_sum = one_hot.T @ z  # (K, D)

    new_ema_size = decay * codebook.ema_cluster_size + (1 - decay) * cluster_size
    new_ema_sum = decay * codebook.ema_sum + (1 - decay) * cluster_sum

    # Laplace smoothing
    n = new_ema_size.sum()
    smoothed = (new_ema_size + epsilon) / (n + K * epsilon) * n
    new_emb = new_ema_sum / smoothed[:, np.newaxis]

    return Codebook(
        embeddings=new_emb,
        usage_count=codebook.usage_count + cluster_size,
        ema_cluster_size=new_ema_size,
        ema_sum=new_ema_sum,
    )


def codebook_perplexity(indices: np.ndarray, n_codes: int) -> float:
    """Codebook utilization as perplexity = exp(entropy).

    Healthy range: close to n_codes means full utilization.
    """
    counts = np.bincount(indices, minlength=n_codes).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(np.exp(entropy))


# ---------- masked acoustic modeling ----------

def generate_mask(
    T: int,
    mask_prob: float = 0.065,
    mask_length: int = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate span-based mask for masked prediction (wav2vec 2.0 style).

    Args:
        T: sequence length
        mask_prob: probability of starting a mask span at each position
        mask_length: length of each mask span

    Returns:
        boolean mask of shape (T,), True = masked
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros(T, dtype=bool)

    n_mask = max(1, int(mask_prob * T / mask_length))
    starts = rng.choice(max(1, T - mask_length), size=n_mask, replace=False)

    for s in starts:
        mask[s:s + mask_length] = True

    return mask


def masked_prediction_loss(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    """L2 loss over masked positions only.

    Args:
        predicted: (T, D) model predictions
        target: (T, D) ground truth features
        mask: (T,) boolean mask
    """
    if not mask.any():
        return 0.0
    diff = predicted[mask] - target[mask]
    return float(np.mean(diff ** 2))


def contrastive_prediction_loss(
    context: np.ndarray,
    targets: np.ndarray,
    negatives: np.ndarray,
    temperature: float = 0.1,
) -> float:
    """InfoNCE loss for masked prediction with distractors.

    Args:
        context: (M, D) context representations at masked positions
        targets: (M, D) true quantized targets
        negatives: (M, N, D) sampled negative targets
        temperature: softmax temperature
    """
    M, D = context.shape
    N = negatives.shape[1]

    pos_sim = np.sum(context * targets, axis=1) / temperature  # (M,)

    neg_sim = np.einsum('md,mnd->mn', context, negatives) / temperature  # (M, N)

    logits = np.concatenate([pos_sim[:, np.newaxis], neg_sim], axis=1)  # (M, 1+N)
    log_softmax = logits[:, 0] - np.log(np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1) + 1e-10) - logits.max(axis=1)

    # Numerically stable log-sum-exp
    max_logits = logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - max_logits), axis=1)) + max_logits.squeeze()
    nce_loss = -(pos_sim - log_sum_exp)

    return float(np.mean(nce_loss))


def sample_negatives(
    features: np.ndarray,
    mask: np.ndarray,
    n_negatives: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample negative examples from unmasked positions.

    Args:
        features: (T, D) quantized features
        mask: (T,) boolean mask
        n_negatives: negatives per masked position

    Returns:
        (M, n_negatives, D) negative samples
    """
    rng = np.random.RandomState(seed)
    masked_idx = np.where(mask)[0]
    M = len(masked_idx)
    T = features.shape[0]

    neg_indices = rng.randint(0, T, size=(M, n_negatives))
    negatives = features[neg_indices]  # (M, n_neg, D)
    return negatives


# ---------- Barlow Twins redundancy reduction ----------

def barlow_twins_loss(
    z_a: np.ndarray,
    z_b: np.ndarray,
    lambda_off: float = 0.005,
) -> float:
    """Barlow Twins loss: decorrelate embedding dimensions (Zbontar et al., 2021).

    Args:
        z_a: (N, D) embeddings from augmentation A (standardized)
        z_b: (N, D) embeddings from augmentation B (standardized)
        lambda_off: weight for off-diagonal terms

    Returns:
        scalar loss
    """
    N, D = z_a.shape

    z_a_norm = (z_a - z_a.mean(axis=0)) / (z_a.std(axis=0) + 1e-6)
    z_b_norm = (z_b - z_b.mean(axis=0)) / (z_b.std(axis=0) + 1e-6)

    cross_corr = (z_a_norm.T @ z_b_norm) / N  # (D, D)

    diag_loss = np.sum((np.diag(cross_corr) - 1) ** 2)

    off_diag = cross_corr.copy()
    np.fill_diagonal(off_diag, 0)
    off_diag_loss = np.sum(off_diag ** 2)

    return float(diag_loss + lambda_off * off_diag_loss)


# ---------- VICReg ----------

def variance_loss(z: np.ndarray, gamma: float = 1.0) -> float:
    """Variance term: prevent embedding collapse (Bardes et al., 2022).

    Hinge loss encouraging std(z_d) >= gamma along each dimension.
    """
    std = np.std(z, axis=0)
    return float(np.mean(np.maximum(0, gamma - std)))


def invariance_loss(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """Invariance term: MSE between augmented views."""
    return float(np.mean((z_a - z_b) ** 2))


def covariance_loss(z: np.ndarray) -> float:
    """Covariance term: decorrelate embedding dimensions."""
    N, D = z.shape
    z_centered = z - z.mean(axis=0)
    cov = (z_centered.T @ z_centered) / (N - 1)
    off_diag = cov.copy()
    np.fill_diagonal(off_diag, 0)
    return float(np.sum(off_diag ** 2) / D)


def vicreg_loss(
    z_a: np.ndarray,
    z_b: np.ndarray,
    lambda_var: float = 25.0,
    lambda_inv: float = 25.0,
    lambda_cov: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """VICReg combined loss (Bardes et al., 2022).

    Returns:
        (total_loss, component_dict)
    """
    v = variance_loss(z_a) + variance_loss(z_b)
    i = invariance_loss(z_a, z_b)
    c = covariance_loss(z_a) + covariance_loss(z_b)

    total = lambda_var * v + lambda_inv * i + lambda_cov * c
    return total, {"variance": v, "invariance": i, "covariance": c}


# ---------- feature pyramid ----------

def feature_pyramid(
    features: np.ndarray,
    scales: Sequence[int] = (1, 2, 4, 8),
) -> List[np.ndarray]:
    """Multi-resolution feature pyramid via average pooling at different scales.

    Args:
        features: (T, D) frame features
        scales: temporal downsampling factors

    Returns:
        list of (T//s, D) feature maps
    """
    pyramid = []
    for s in scales:
        pooled = mean_pool(features, s)
        pyramid.append(pooled)
    return pyramid


def pyramid_merge(pyramid: List[np.ndarray], target_len: int) -> np.ndarray:
    """Merge multi-scale pyramid into unified representation via interpolation.

    Upsamples each level to target_len and concatenates along feature axis.
    """
    merged = []
    for level in pyramid:
        T_l, D = level.shape
        if T_l == target_len:
            merged.append(level)
        else:
            indices = np.linspace(0, T_l - 1, target_len)
            idx_floor = np.floor(indices).astype(int)
            idx_ceil = np.minimum(idx_floor + 1, T_l - 1)
            frac = indices - idx_floor
            upsampled = (1 - frac[:, np.newaxis]) * level[idx_floor] + frac[:, np.newaxis] * level[idx_ceil]
            merged.append(upsampled)
    return np.concatenate(merged, axis=1)


# ---------- representation quality ----------

def representation_entropy(embeddings: np.ndarray, n_bins: int = 50) -> float:
    """Estimate entropy of the embedding distribution via histogram."""
    D = embeddings.shape[1]
    total_entropy = 0.0
    for d in range(D):
        hist, _ = np.histogram(embeddings[:, d], bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (embeddings[:, d].max() - embeddings[:, d].min()) / n_bins
        if bin_width <= 0:
            continue
        total_entropy -= np.sum(hist * np.log(hist + 1e-10) * bin_width)
    return float(total_entropy / D)


def isotropy_score(embeddings: np.ndarray) -> float:
    """Measure how isotropic the embedding space is (Mu et al., 2018).

    1.0 = perfectly isotropic, 0.0 = collapsed to a line.
    Uses the ratio of minimum to maximum singular values.
    """
    centered = embeddings - embeddings.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-10:
        return 0.0
    return float(s[-1] / s[0])


def effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank of the embedding matrix (Roy & Vetterli, 2007).

    exp(entropy of normalized singular values).
    """
    centered = embeddings - embeddings.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def alignment_uniformity(
    z_a: np.ndarray,
    z_b: np.ndarray,
    alpha: float = 2.0,
    t: float = 2.0,
) -> Tuple[float, float]:
    """Alignment and uniformity metrics (Wang & Isola, 2020).

    alignment: expected distance between positive pairs (lower = better)
    uniformity: log of expected pairwise Gaussian potential (lower = better spread)
    """
    z_a_n = z_a / (np.linalg.norm(z_a, axis=1, keepdims=True) + 1e-10)
    z_b_n = z_b / (np.linalg.norm(z_b, axis=1, keepdims=True) + 1e-10)

    alignment = float(np.mean(np.linalg.norm(z_a_n - z_b_n, axis=1) ** alpha))

    all_z = np.concatenate([z_a_n, z_b_n], axis=0)
    sq_dists = np.sum((all_z[:, np.newaxis] - all_z[np.newaxis]) ** 2, axis=2)
    n = all_z.shape[0]
    mask = ~np.eye(n, dtype=bool)
    uniformity = float(np.log(np.mean(np.exp(-t * sq_dists[mask])) + 1e-10))

    return alignment, uniformity


# ---------- helpers ----------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-10)
