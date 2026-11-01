"""Data augmentation for music audio features.

All augmentations operate on feature vectors (numpy arrays), not raw audio.
They transform the extracted features to simulate audio-domain perturbations.
"""
from __future__ import annotations

import numpy as np


_CHROMA_FEATURE_INDICES = list(range(12))


def pitch_shift_features(
    features: np.ndarray,
    semitones: int,
    chroma_indices: list[int] | None = None,
) -> np.ndarray:
    result = features.copy()
    idx = chroma_indices or _CHROMA_FEATURE_INDICES
    valid_idx = [i for i in idx if i < result.shape[-1]]

    if result.ndim == 1:
        chroma = result[valid_idx]
        shifted = np.roll(chroma, semitones)
        for i, vi in enumerate(valid_idx):
            result[vi] = shifted[i]
    elif result.ndim == 2:
        for row in range(result.shape[0]):
            chroma = result[row, valid_idx]
            shifted = np.roll(chroma, semitones)
            for i, vi in enumerate(valid_idx):
                result[row, vi] = shifted[i]
    return result


def tempo_scale_features(
    features: np.ndarray,
    scale_factor: float,
    bpm_index: int = 0,
    rhythm_indices: list[int] | None = None,
) -> np.ndarray:
    result = features.copy()
    if result.ndim == 1:
        if bpm_index < len(result):
            result[bpm_index] *= scale_factor
        if rhythm_indices:
            for ri in rhythm_indices:
                if ri < len(result):
                    result[ri] *= scale_factor
    elif result.ndim == 2:
        if bpm_index < result.shape[1]:
            result[:, bpm_index] *= scale_factor
        if rhythm_indices:
            for ri in rhythm_indices:
                if ri < result.shape[1]:
                    result[:, ri] *= scale_factor
    return result


def noise_injection(
    features: np.ndarray,
    noise_scale: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randn(*features.shape) * noise_scale
    return features + noise


def feature_dropout(
    features: np.ndarray,
    dropout_rate: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = rng.random(features.shape) > dropout_rate
    return features * mask


def mixup(
    features_a: np.ndarray,
    features_b: np.ndarray,
    alpha: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    rng = np.random.RandomState(seed)
    lam = rng.beta(alpha, alpha) if alpha > 0 else 0.5
    mixed = lam * features_a + (1 - lam) * features_b
    return mixed, float(lam)
