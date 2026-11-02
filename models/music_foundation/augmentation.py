"""Audio augmentation for self-supervised pretraining.

Implements augmentation strategies that preserve musical structure while
providing sufficient variation for contrastive learning:
- Time-domain: gain jitter, polarity inversion, additive noise, clipping
- Frequency-domain: frequency masking, frequency warping, EQ randomization
- Time-frequency: SpecAugment (Park et al., 2019), mixup (Zhang et al., 2018)
- Music-aware: pitch shift via resampling, tempo stretch, channel swap
- Composition: augmentation chains with probability scheduling

Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class AugType(Enum):
    GAIN = auto()
    POLARITY = auto()
    NOISE = auto()
    CLIP = auto()
    FREQ_MASK = auto()
    TIME_MASK = auto()
    FREQ_WARP = auto()
    PITCH_SHIFT = auto()
    TEMPO_STRETCH = auto()
    MIXUP = auto()
    SPEC_AUGMENT = auto()
    EQ_RANDOM = auto()
    CHANNEL_SWAP = auto()


@dataclass
class AugConfig:
    aug_type: AugType
    probability: float = 0.5
    params: Dict = field(default_factory=dict)


@dataclass
class AugResult:
    signal: np.ndarray
    applied: List[AugType]
    metadata: Dict = field(default_factory=dict)


# ---------- time-domain augmentations ----------

def gain_jitter(signal: np.ndarray, min_db: float = -6.0, max_db: float = 6.0,
                seed: Optional[int] = None) -> np.ndarray:
    """Random gain change in dB."""
    rng = np.random.RandomState(seed)
    gain_db = rng.uniform(min_db, max_db)
    return signal * 10 ** (gain_db / 20.0)


def polarity_inversion(signal: np.ndarray) -> np.ndarray:
    """Flip signal polarity. Preserves all spectral properties."""
    return -signal


def additive_noise(signal: np.ndarray, snr_db: float = 20.0,
                   seed: Optional[int] = None) -> np.ndarray:
    """Add Gaussian noise at specified SNR."""
    rng = np.random.RandomState(seed)
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:
        return signal
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(*signal.shape) * np.sqrt(noise_power)
    return signal + noise


def soft_clip(signal: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Soft clipping via tanh compression above threshold."""
    above = np.abs(signal) > threshold
    clipped = signal.copy()
    clipped[above] = np.sign(signal[above]) * (
        threshold + (1 - threshold) * np.tanh(
            (np.abs(signal[above]) - threshold) / (1 - threshold)
        )
    )
    return clipped


def random_crop(signal: np.ndarray, crop_length: int,
                seed: Optional[int] = None) -> np.ndarray:
    """Random temporal crop."""
    rng = np.random.RandomState(seed)
    if len(signal) <= crop_length:
        pad = np.zeros(crop_length - len(signal))
        return np.concatenate([signal, pad])
    start = rng.randint(0, len(signal) - crop_length)
    return signal[start:start + crop_length]


def fade_in_out(signal: np.ndarray, fade_samples: int = 256) -> np.ndarray:
    """Apply cosine fade-in and fade-out."""
    result = signal.copy()
    n = min(fade_samples, len(signal) // 2)
    fade = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
    result[:n] *= fade
    result[-n:] *= fade[::-1]
    return result


# ---------- frequency-domain augmentations ----------

def frequency_mask(
    spectrogram: np.ndarray,
    n_masks: int = 1,
    max_width: int = 20,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Mask random frequency bands (SpecAugment freq mask, Park et al., 2019)."""
    rng = np.random.RandomState(seed)
    F, T = spectrogram.shape
    result = spectrogram.copy()
    for _ in range(n_masks):
        width = rng.randint(1, max_width + 1)
        start = rng.randint(0, max(1, F - width))
        result[start:start + width, :] = 0
    return result


def time_mask(
    spectrogram: np.ndarray,
    n_masks: int = 1,
    max_width: int = 50,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Mask random time segments (SpecAugment time mask)."""
    rng = np.random.RandomState(seed)
    F, T = spectrogram.shape
    result = spectrogram.copy()
    for _ in range(n_masks):
        width = rng.randint(1, max_width + 1)
        start = rng.randint(0, max(1, T - width))
        result[:, start:start + width] = 0
    return result


def spec_augment(
    spectrogram: np.ndarray,
    n_freq_masks: int = 2,
    freq_width: int = 15,
    n_time_masks: int = 2,
    time_width: int = 35,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Full SpecAugment policy (Park et al., 2019)."""
    result = frequency_mask(spectrogram, n_freq_masks, freq_width, seed)
    seed2 = seed + 1 if seed is not None else None
    result = time_mask(result, n_time_masks, time_width, seed2)
    return result


def frequency_warp(
    spectrogram: np.ndarray,
    warp_factor: float = 1.05,
) -> np.ndarray:
    """Warp frequency axis by a constant factor (vocal tract length perturbation).

    warp_factor > 1 shifts up, < 1 shifts down.
    """
    F, T = spectrogram.shape
    new_freqs = np.arange(F) / warp_factor
    new_freqs = np.clip(new_freqs, 0, F - 1)

    idx_low = np.floor(new_freqs).astype(int)
    idx_high = np.minimum(idx_low + 1, F - 1)
    frac = new_freqs - idx_low

    result = (1 - frac[:, np.newaxis]) * spectrogram[idx_low] + \
             frac[:, np.newaxis] * spectrogram[idx_high]
    return result


def random_eq(
    spectrogram: np.ndarray,
    n_bands: int = 5,
    max_gain_db: float = 6.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Random per-band EQ gain."""
    rng = np.random.RandomState(seed)
    F, T = spectrogram.shape
    band_edges = np.linspace(0, F, n_bands + 1, dtype=int)
    gains_db = rng.uniform(-max_gain_db, max_gain_db, size=n_bands)
    gains_linear = 10 ** (gains_db / 20.0)

    result = spectrogram.copy()
    for i in range(n_bands):
        result[band_edges[i]:band_edges[i + 1]] *= gains_linear[i]
    return result


# ---------- music-aware augmentations ----------

def pitch_shift_resample(
    signal: np.ndarray,
    semitones: float,
    sr: int = 22050,
) -> np.ndarray:
    """Pitch shift via resampling (changes duration; combine with tempo stretch).

    semitones > 0 = shift up, < 0 = shift down.
    """
    from scipy.signal import resample

    factor = 2 ** (-semitones / 12.0)
    new_length = int(len(signal) * factor)
    if new_length < 1:
        return signal
    shifted = resample(signal, new_length)

    if len(shifted) > len(signal):
        return shifted[:len(signal)]
    else:
        pad = np.zeros(len(signal) - len(shifted))
        return np.concatenate([shifted, pad])


def tempo_stretch(
    spectrogram: np.ndarray,
    rate: float = 1.0,
) -> np.ndarray:
    """Time-stretch a spectrogram by interpolation.

    rate > 1 = faster (shorter), < 1 = slower (longer).
    """
    F, T = spectrogram.shape
    new_T = max(1, int(T / rate))
    indices = np.linspace(0, T - 1, new_T)
    idx_low = np.floor(indices).astype(int)
    idx_high = np.minimum(idx_low + 1, T - 1)
    frac = indices - idx_low

    result = (1 - frac[np.newaxis, :]) * spectrogram[:, idx_low] + \
             frac[np.newaxis, :] * spectrogram[:, idx_high]
    return result


def mixup(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    alpha: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Mixup augmentation (Zhang et al., 2018).

    Returns:
        (mixed_signal, lambda) where lambda is the mixing coefficient
    """
    rng = np.random.RandomState(seed)
    lam = rng.beta(alpha, alpha) if alpha > 0 else 0.5

    min_len = min(len(signal_a), len(signal_b))
    mixed = lam * signal_a[:min_len] + (1 - lam) * signal_b[:min_len]
    return mixed, lam


def cutmix_spectrogram(
    spec_a: np.ndarray,
    spec_b: np.ndarray,
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """CutMix on spectrogram: replace a rectangular region (Yun et al., 2019)."""
    rng = np.random.RandomState(seed)
    F, T = spec_a.shape
    lam = rng.beta(alpha, alpha) if alpha > 0 else 0.5

    cut_ratio = np.sqrt(1 - lam)
    cut_f = int(F * cut_ratio)
    cut_t = int(T * cut_ratio)

    cf = rng.randint(0, max(1, F - cut_f))
    ct = rng.randint(0, max(1, T - cut_t))

    result = spec_a.copy()
    result[cf:cf + cut_f, ct:ct + cut_t] = spec_b[cf:cf + cut_f, ct:ct + cut_t]

    actual_lam = 1 - (cut_f * cut_t) / (F * T)
    return result, actual_lam


# ---------- augmentation chain ----------

def build_augmentation_chain(
    configs: List[AugConfig],
) -> Callable:
    """Build a composed augmentation pipeline from configs.

    Each augmentation is applied with its specified probability.
    """
    def chain(signal: np.ndarray, seed: Optional[int] = None) -> AugResult:
        rng = np.random.RandomState(seed)
        result = signal.copy()
        applied = []

        for cfg in configs:
            if rng.random() > cfg.probability:
                continue

            aug_seed = rng.randint(0, 2**31)
            p = cfg.params

            if cfg.aug_type == AugType.GAIN:
                result = gain_jitter(result, p.get('min_db', -6), p.get('max_db', 6), aug_seed)
            elif cfg.aug_type == AugType.POLARITY:
                result = polarity_inversion(result)
            elif cfg.aug_type == AugType.NOISE:
                result = additive_noise(result, p.get('snr_db', 20), aug_seed)
            elif cfg.aug_type == AugType.CLIP:
                result = soft_clip(result, p.get('threshold', 0.8))
            elif cfg.aug_type == AugType.PITCH_SHIFT:
                semi = rng.uniform(p.get('min_semi', -2), p.get('max_semi', 2))
                result = pitch_shift_resample(result, semi)
            else:
                pass

            applied.append(cfg.aug_type)

        return AugResult(signal=result, applied=applied)

    return chain


def default_pretraining_chain() -> Callable:
    """Standard augmentation chain for self-supervised pretraining."""
    configs = [
        AugConfig(AugType.GAIN, 0.8, {'min_db': -4, 'max_db': 4}),
        AugConfig(AugType.NOISE, 0.3, {'snr_db': 25}),
        AugConfig(AugType.POLARITY, 0.5),
        AugConfig(AugType.CLIP, 0.1, {'threshold': 0.9}),
    ]
    return build_augmentation_chain(configs)


def two_view_augmentation(
    signal: np.ndarray,
    chain_fn: Optional[Callable] = None,
    seed: int = 42,
) -> Tuple[AugResult, AugResult]:
    """Generate two augmented views for contrastive learning."""
    if chain_fn is None:
        chain_fn = default_pretraining_chain()
    view_a = chain_fn(signal, seed=seed)
    view_b = chain_fn(signal, seed=seed + 7919)
    return view_a, view_b
