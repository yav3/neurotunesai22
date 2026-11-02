"""Attention mechanisms for music understanding.

Implements attention variants suited for musical sequences:
- Standard scaled dot-product and multi-head attention (Vaswani et al., 2017)
- Relative position attention (Shaw et al., 2018; Dai et al., 2019)
- Local/windowed attention for long sequences (Beltagy et al., 2020)
- Cross-attention for multi-modal fusion (audio + text, audio + MIDI)
- Pitch-aware attention with harmonic position encoding
- Causal (autoregressive) masking for generation
- Efficient attention via random feature maps (Performer, Choromanski et al., 2021)

Pure numpy. Forward-pass only (no autograd).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AttentionConfig:
    d_model: int = 512
    n_heads: int = 8
    d_k: Optional[int] = None
    d_v: Optional[int] = None
    dropout_rate: float = 0.0
    max_relative_position: int = 128
    window_size: int = 256
    causal: bool = False

    def __post_init__(self):
        if self.d_k is None:
            self.d_k = self.d_model // self.n_heads
        if self.d_v is None:
            self.d_v = self.d_model // self.n_heads


@dataclass
class AttentionOutput:
    output: np.ndarray
    weights: np.ndarray
    entropy: float


# ---------- core attention ----------

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    temperature: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention (Vaswani et al., 2017).

    Args:
        Q: (..., T_q, d_k)
        K: (..., T_k, d_k)
        V: (..., T_k, d_v)
        mask: (..., T_q, T_k) boolean, True = attend
        temperature: if None, uses sqrt(d_k)

    Returns:
        (output, weights) where output is (..., T_q, d_v)
    """
    d_k = Q.shape[-1]
    scale = temperature if temperature is not None else np.sqrt(d_k)

    scores = Q @ K.swapaxes(-2, -1) / scale  # (..., T_q, T_k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = _softmax_last(scores)
    output = weights @ V

    return output, weights


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    n_heads: int,
    mask: Optional[np.ndarray] = None,
) -> AttentionOutput:
    """Multi-head attention with linear projections.

    Args:
        Q, K, V: (T, d_model)
        W_q, W_k, W_v: (d_model, d_model) projection weights
        W_o: (d_model, d_model) output projection
        n_heads: number of attention heads
        mask: (T_q, T_k) or None
    """
    T_q, d_model = Q.shape
    d_k = d_model // n_heads

    q = (Q @ W_q).reshape(T_q, n_heads, d_k).transpose(1, 0, 2)  # (H, T_q, d_k)
    k = (K @ W_k).reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)
    v = (V @ W_v).reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)

    if mask is not None:
        mask = mask[np.newaxis]  # (1, T_q, T_k)

    out, weights = scaled_dot_product_attention(q, k, v, mask)  # (H, T_q, d_k)

    out = out.transpose(1, 0, 2).reshape(T_q, d_model)  # (T_q, d_model)
    output = out @ W_o

    avg_weights = weights.mean(axis=0)  # (T_q, T_k)
    entropy = _attention_entropy(avg_weights)

    return AttentionOutput(output=output, weights=avg_weights, entropy=entropy)


# ---------- relative position attention ----------

def relative_position_encoding(
    max_len: int,
    d_model: int,
    max_relative: int = 128,
) -> np.ndarray:
    """Compute relative position bias matrix (Shaw et al., 2018).

    Returns:
        (2*max_relative+1, d_model) relative embeddings
    """
    positions = np.arange(-max_relative, max_relative + 1, dtype=np.float64)
    dim = np.arange(d_model, dtype=np.float64)
    angles = positions[:, np.newaxis] / (10000 ** (2 * (dim // 2) / d_model))

    encoding = np.zeros_like(angles)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def relative_attention_logits(
    Q: np.ndarray,
    rel_emb: np.ndarray,
    max_relative: int,
) -> np.ndarray:
    """Compute relative position attention logits.

    Args:
        Q: (T, d_k) queries
        rel_emb: (2*max_relative+1, d_k) relative embeddings

    Returns:
        (T, T) relative position bias
    """
    T = Q.shape[0]
    rel_logits = Q @ rel_emb.T  # (T, 2*max_relative+1)

    bias = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            rel = np.clip(j - i, -max_relative, max_relative) + max_relative
            bias[i, j] = rel_logits[i, rel]
    return bias


def relative_position_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    rel_emb: np.ndarray,
    max_relative: int = 128,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attention with relative position bias.

    Scores = Q·K^T/√d + Q·R^T (content + position).
    """
    d_k = Q.shape[-1]
    content_scores = Q @ K.T / np.sqrt(d_k)
    position_bias = relative_attention_logits(Q, rel_emb, max_relative)

    scores = content_scores + position_bias

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = _softmax_last(scores)
    output = weights @ V
    return output, weights


# ---------- local/windowed attention ----------

def local_attention_mask(T: int, window_size: int) -> np.ndarray:
    """Create a banded attention mask for local (windowed) attention.

    Returns:
        (T, T) boolean mask, True within window
    """
    indices = np.arange(T)
    diff = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    return diff <= window_size // 2


def causal_mask(T: int) -> np.ndarray:
    """Lower-triangular causal mask for autoregressive generation."""
    return np.tril(np.ones((T, T), dtype=bool))


def causal_local_mask(T: int, window_size: int) -> np.ndarray:
    """Combined causal + local mask."""
    local = local_attention_mask(T, window_size)
    causal = np.tril(np.ones((T, T), dtype=bool))
    return local & causal


# ---------- cross-attention ----------

def cross_attention(
    query: np.ndarray,
    context: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    n_heads: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-attention for multi-modal fusion.

    query: (T_q, d_model) — e.g., audio frames
    context: (T_c, d_model) — e.g., text tokens or MIDI events
    """
    T_q, d_model = query.shape
    d_k = d_model // n_heads

    q = (query @ W_q).reshape(T_q, n_heads, d_k).transpose(1, 0, 2)
    k = (context @ W_k).reshape(context.shape[0], n_heads, d_k).transpose(1, 0, 2)
    v = (context @ W_v).reshape(context.shape[0], n_heads, d_k).transpose(1, 0, 2)

    out, weights = scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 0, 2).reshape(T_q, d_model)

    return out, weights.mean(axis=0)


# ---------- pitch-aware attention ----------

def harmonic_position_encoding(
    pitches: np.ndarray,
    d_model: int,
    base_freq: float = 440.0,
) -> np.ndarray:
    """Encode MIDI pitch as sinusoidal position embedding with harmonic structure.

    Encodes both absolute pitch and octave-invariant pitch class.
    """
    T = len(pitches)
    encoding = np.zeros((T, d_model))

    half = d_model // 2
    freq_ratio = 2 ** ((pitches - 69) / 12)  # relative to A4

    dim = np.arange(half)
    for t in range(T):
        angles = freq_ratio[t] / (10000 ** (2 * dim / d_model))
        encoding[t, :half:2] = np.sin(angles[:half // 2 + half % 2])
        encoding[t, 1:half:2] = np.cos(angles[:half // 2])

        pc = pitches[t] % 12
        pc_angles = pc / (10000 ** (2 * dim / d_model))
        encoding[t, half::2] = np.sin(pc_angles[:d_model // 4 + d_model % 4])
        encoding[t, half + 1::2] = np.cos(pc_angles[:(d_model - half) // 2])

    return encoding


def pitch_aware_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    pitches_q: np.ndarray,
    pitches_k: np.ndarray,
    d_model: int,
    harmonic_weight: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attention biased by pitch proximity and harmonic relationships.

    Intervals of octave (12), fifth (7), fourth (5) get positive bias.
    """
    d_k = Q.shape[-1]
    content_scores = Q @ K.T / np.sqrt(d_k)

    intervals = np.abs(pitches_q[:, np.newaxis] - pitches_k[np.newaxis, :]) % 12
    harmonic_intervals = {0, 7, 5, 4, 3}
    harmonic_bias = np.zeros_like(content_scores)
    for iv in harmonic_intervals:
        harmonic_bias += (intervals == iv).astype(float)
    harmonic_bias /= len(harmonic_intervals)

    scores = content_scores + harmonic_weight * harmonic_bias
    weights = _softmax_last(scores)
    output = weights @ V
    return output, weights


# ---------- efficient attention (Performer) ----------

def random_fourier_features(
    x: np.ndarray,
    n_features: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Approximate softmax kernel via random Fourier features (Rahimi & Recht, 2007).

    Used for linear-time attention (Performer, Choromanski et al., 2021).
    """
    rng = np.random.RandomState(seed)
    d = x.shape[-1]
    W = rng.randn(d, n_features) / np.sqrt(d)

    xW = x @ W  # (..., n_features)
    phi = np.concatenate([np.cos(xW), np.sin(xW)], axis=-1) / np.sqrt(n_features)
    return phi


def performer_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    n_features: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Linear-time attention via random feature approximation.

    O(T·d·m) instead of O(T²·d), where m = n_features.
    """
    Q_prime = random_fourier_features(Q, n_features, seed)
    K_prime = random_fourier_features(K, n_features, seed)

    KV = K_prime.T @ V  # (2m, d_v)
    K_sum = K_prime.sum(axis=0)  # (2m,)

    numerator = Q_prime @ KV  # (T_q, d_v)
    denominator = Q_prime @ K_sum  # (T_q,)

    return numerator / (denominator[:, np.newaxis] + 1e-6)


# ---------- attention analysis ----------

def _attention_entropy(weights: np.ndarray) -> float:
    """Average entropy of attention distributions (higher = more diffuse)."""
    w = weights + 1e-10
    entropy_per_query = -np.sum(w * np.log(w), axis=-1)
    return float(np.mean(entropy_per_query))


def attention_head_diversity(head_weights: np.ndarray) -> float:
    """Measure diversity across attention heads via pairwise JS divergence.

    head_weights: (H, T_q, T_k)
    """
    H = head_weights.shape[0]
    total_div = 0.0
    count = 0
    for i in range(H):
        for j in range(i + 1, H):
            m = 0.5 * (head_weights[i] + head_weights[j])
            kl_i = np.sum(head_weights[i] * np.log((head_weights[i] + 1e-10) / (m + 1e-10)), axis=-1)
            kl_j = np.sum(head_weights[j] * np.log((head_weights[j] + 1e-10) / (m + 1e-10)), axis=-1)
            total_div += np.mean(0.5 * kl_i + 0.5 * kl_j)
            count += 1
    return float(total_div / max(count, 1))


def attention_sparsity(weights: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of attention weights below threshold (higher = sparser)."""
    return float(np.mean(weights < threshold))


def effective_attention_span(weights: np.ndarray) -> float:
    """Average effective span: expected distance from query to attended key."""
    T_q, T_k = weights.shape
    positions = np.arange(T_k)
    expected_pos = weights @ positions  # (T_q,)
    query_pos = np.arange(T_q)
    spans = np.abs(expected_pos - query_pos)
    return float(np.mean(spans))


# ---------- helper ----------

def _softmax_last(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)


def init_attention_weights(
    d_model: int,
    n_heads: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Xavier-initialized Q, K, V, O projection matrices."""
    rng = np.random.RandomState(seed)
    scale = np.sqrt(2.0 / (d_model + d_model))
    W_q = rng.randn(d_model, d_model) * scale
    W_k = rng.randn(d_model, d_model) * scale
    W_v = rng.randn(d_model, d_model) * scale
    W_o = rng.randn(d_model, d_model) * scale
    return W_q, W_k, W_v, W_o
