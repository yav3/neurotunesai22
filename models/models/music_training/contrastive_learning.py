"""Contrastive learning for music representations.

Implements NT-Xent (normalized temperature-scaled cross-entropy),
triplet loss, and music-specific augmentation strategies for
self-supervised representation learning.
"""
from __future__ import annotations


import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class AugmentationType(Enum):
    PITCH_SHIFT = "pitch_shift"
    TIME_STRETCH = "time_stretch"
    FREQUENCY_MASK = "frequency_mask"
    TIME_MASK = "time_mask"
    GAUSSIAN_NOISE = "gaussian_noise"
    GAIN = "gain"
    MIXUP = "mixup"


@dataclass(frozen=True)
class ContrastivePair:
    anchor: np.ndarray
    positive: np.ndarray
    augmentation_type: AugmentationType


@dataclass(frozen=True)
class TripletSample:
    anchor: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    margin: float = 0.2


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute NxN cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = embeddings / norms
    return normalized @ normalized.T


def nt_xent_loss(
    z_i: np.ndarray,
    z_j: np.ndarray,
    temperature: float = 0.07,
) -> float:
    """Normalized temperature-scaled cross-entropy loss (SimCLR).

    z_i, z_j: (N, D) matrices of paired embeddings.
    Returns scalar loss.
    """
    n = len(z_i)
    if n == 0:
        return 0.0
    z = np.vstack([z_i, z_j])  # (2N, D)
    sim = pairwise_cosine_similarity(z) / temperature
    sim_max = np.max(sim, axis=1, keepdims=True)
    sim = sim - sim_max
    exp_sim = np.exp(sim)
    np.fill_diagonal(exp_sim, 0.0)
    loss = 0.0
    for k in range(n):
        pos_idx = k + n
        numerator = exp_sim[k, pos_idx]
        denominator = np.sum(exp_sim[k])
        if denominator > 0 and numerator > 0:
            loss -= math.log(numerator / denominator)
        pos_idx_rev = k
        numerator_rev = exp_sim[k + n, pos_idx_rev]
        denominator_rev = np.sum(exp_sim[k + n])
        if denominator_rev > 0 and numerator_rev > 0:
            loss -= math.log(numerator_rev / denominator_rev)
    return loss / (2 * n)


def triplet_loss(
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    margin: float = 0.2,
) -> float:
    """Triplet margin loss: max(0, d(a,p) - d(a,n) + margin).

    Uses L2 distance.
    """
    d_pos = float(np.linalg.norm(anchor - positive))
    d_neg = float(np.linalg.norm(anchor - negative))
    return max(0.0, d_pos - d_neg + margin)


def batch_triplet_loss(
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    margin: float = 0.2,
) -> float:
    """Batch mean triplet loss."""
    n = len(anchors)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += triplet_loss(anchors[i], positives[i], negatives[i], margin)
    return total / n


def hard_negative_mining(
    anchor: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    margin: float = 0.2,
) -> int:
    """Find hardest negative (closest to anchor that still violates margin).

    Returns index of the hardest negative in the negatives array.
    """
    if len(negatives) == 0:
        return -1
    d_pos = float(np.linalg.norm(anchor - positives[0]))
    distances = np.linalg.norm(negatives - anchor, axis=1)
    semi_hard_mask = (distances > d_pos) & (distances < d_pos + margin)
    if np.any(semi_hard_mask):
        candidates = np.where(semi_hard_mask)[0]
        return int(candidates[np.argmin(distances[candidates])])
    return int(np.argmin(distances))


@dataclass
class MusicAugmentor:
    """Apply music-specific augmentations for contrastive pairs."""
    pitch_shift_range: tuple[int, int] = (-2, 2)
    time_stretch_range: tuple[float, float] = (0.9, 1.1)
    freq_mask_width: int = 8
    time_mask_width: int = 16
    noise_std: float = 0.01
    gain_range_db: tuple[float, float] = (-6.0, 6.0)

    def pitch_shift_spectrogram(self, spec: np.ndarray, shift_bins: int) -> np.ndarray:
        """Shift a spectrogram along the frequency axis."""
        result = np.zeros_like(spec)
        if shift_bins > 0:
            result[shift_bins:, :] = spec[:-shift_bins, :]
        elif shift_bins < 0:
            result[:shift_bins, :] = spec[-shift_bins:, :]
        else:
            result = spec.copy()
        return result

    def frequency_mask(self, spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """SpecAugment-style frequency masking."""
        result = spec.copy()
        n_freq = spec.shape[0]
        width = min(self.freq_mask_width, n_freq)
        start = rng.integers(0, n_freq - width + 1)
        result[start:start + width, :] = 0.0
        return result

    def time_mask(self, spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """SpecAugment-style time masking."""
        result = spec.copy()
        n_time = spec.shape[1]
        width = min(self.time_mask_width, n_time)
        start = rng.integers(0, n_time - width + 1)
        result[:, start:start + width] = 0.0
        return result

    def add_noise(self, spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add Gaussian noise."""
        noise = rng.normal(0, self.noise_std, spec.shape)
        return spec + noise

    def apply_gain(self, spec: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply gain in dB to spectrogram."""
        factor = 10 ** (gain_db / 20.0)
        return spec * factor

    def create_pair(
        self,
        spectrogram: np.ndarray,
        aug_type: AugmentationType,
        rng: np.random.Generator | None = None,
    ) -> ContrastivePair:
        """Create a contrastive pair from one spectrogram."""
        if rng is None:
            rng = np.random.default_rng()
        if aug_type == AugmentationType.PITCH_SHIFT:
            shift = rng.integers(self.pitch_shift_range[0], self.pitch_shift_range[1] + 1)
            positive = self.pitch_shift_spectrogram(spectrogram, shift)
        elif aug_type == AugmentationType.FREQUENCY_MASK:
            positive = self.frequency_mask(spectrogram, rng)
        elif aug_type == AugmentationType.TIME_MASK:
            positive = self.time_mask(spectrogram, rng)
        elif aug_type == AugmentationType.GAUSSIAN_NOISE:
            positive = self.add_noise(spectrogram, rng)
        elif aug_type == AugmentationType.GAIN:
            gain = rng.uniform(*self.gain_range_db)
            positive = self.apply_gain(spectrogram, gain)
        else:
            positive = spectrogram.copy()
        return ContrastivePair(
            anchor=spectrogram,
            positive=positive,
            augmentation_type=aug_type,
        )


@dataclass
class ContrastiveTrainingState:
    """Track contrastive learning metrics."""
    epoch: int = 0
    total_loss: float = 0.0
    n_batches: int = 0
    alignment_history: list[float] = field(default_factory=list)
    uniformity_history: list[float] = field(default_factory=list)

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(self.n_batches, 1)

    def update(self, loss: float) -> None:
        self.total_loss += loss
        self.n_batches += 1


def alignment_metric(z_i: np.ndarray, z_j: np.ndarray, alpha: float = 2.0) -> float:
    """Measure how close positive pairs are (Wang & Isola, 2020)."""
    diffs = np.linalg.norm(z_i - z_j, axis=1) ** alpha
    return float(np.mean(diffs))


def uniformity_metric(z: np.ndarray, t: float = 2.0) -> float:
    """Measure how uniformly distributed on the hypersphere (Wang & Isola, 2020)."""
    n = len(z)
    if n < 2:
        return 0.0
    sq_dists = np.sum((z[:, None] - z[None, :]) ** 2, axis=-1)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return float(np.log(np.mean(np.exp(-t * sq_dists[mask]))))
