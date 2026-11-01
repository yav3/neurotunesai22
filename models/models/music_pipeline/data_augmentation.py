"""Data augmentation for music ML pipelines: pitch shift, time stretch, noise, mixing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AugmentationType(Enum):
    PITCH_SHIFT = "pitch_shift"
    TIME_STRETCH = "time_stretch"
    NOISE_INJECTION = "noise_injection"
    GAIN_CHANGE = "gain_change"
    POLARITY_INVERT = "polarity_invert"
    CHANNEL_SWAP = "channel_swap"
    LOW_PASS = "low_pass"
    HIGH_PASS = "high_pass"
    RANDOM_EQ = "random_eq"


@dataclass
class AugmentationConfig:
    """Configuration for a single augmentation step."""
    aug_type: AugmentationType
    probability: float = 0.5
    min_param: float = -2.0
    max_param: float = 2.0


def pitch_shift_semitones(signal: np.ndarray, semitones: float, sample_rate: int) -> np.ndarray:
    """Shift pitch by resampling (simple sinc-free approach for augmentation).

    Positive semitones = higher pitch; negative = lower.
    """
    ratio = 2.0 ** (semitones / 12.0)
    n = len(signal)
    new_n = int(n / ratio)
    if new_n < 2:
        return signal[:1]
    indices = np.linspace(0, n - 1, new_n)
    shifted = np.interp(indices, np.arange(n), signal)
    # Resample back to original length
    out_indices = np.linspace(0, len(shifted) - 1, n)
    return np.interp(out_indices, np.arange(len(shifted)), shifted)


def time_stretch(signal: np.ndarray, rate: float) -> np.ndarray:
    """Stretch/compress time by rate (>1 = longer, <1 = shorter) via interpolation."""
    if rate <= 0:
        return signal
    n = len(signal)
    new_n = int(n * rate)
    if new_n < 2:
        return signal[:1]
    indices = np.linspace(0, n - 1, new_n)
    return np.interp(indices, np.arange(n), signal)


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Gaussian noise at a specified SNR in dB."""
    if rng is None:
        rng = np.random.default_rng()
    sig_power = np.mean(signal ** 2)
    if sig_power < 1e-20:
        return signal.copy()
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def gain_change_db(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain change in dB."""
    return signal * (10.0 ** (gain_db / 20.0))


def polarity_invert(signal: np.ndarray) -> np.ndarray:
    """Invert signal polarity."""
    return -signal


def low_pass_simple(signal: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Simple low-pass via frequency-domain zeroing. cutoff_ratio in (0,1)."""
    spectrum = np.fft.rfft(signal)
    n_bins = len(spectrum)
    cutoff_bin = int(n_bins * max(0.01, min(1.0, cutoff_ratio)))
    spectrum[cutoff_bin:] = 0
    return np.fft.irfft(spectrum, n=len(signal))


def high_pass_simple(signal: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Simple high-pass via frequency-domain zeroing. cutoff_ratio in (0,1)."""
    spectrum = np.fft.rfft(signal)
    n_bins = len(spectrum)
    cutoff_bin = int(n_bins * max(0.0, min(0.99, cutoff_ratio)))
    spectrum[:cutoff_bin] = 0
    return np.fft.irfft(spectrum, n=len(signal))


def random_eq(signal: np.ndarray, max_db: float = 6.0, n_bands: int = 5,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply random EQ by splitting spectrum into bands and applying random gains."""
    if rng is None:
        rng = np.random.default_rng()
    spectrum = np.fft.rfft(signal)
    n_bins = len(spectrum)
    gains_db = rng.uniform(-max_db, max_db, n_bands)
    band_size = n_bins // n_bands
    for i, g in enumerate(gains_db):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else n_bins
        spectrum[start:end] *= 10.0 ** (g / 20.0)
    return np.fft.irfft(spectrum, n=len(signal))


def mixup(signal_a: np.ndarray, signal_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Mixup augmentation: linear interpolation of two signals."""
    min_len = min(len(signal_a), len(signal_b))
    return alpha * signal_a[:min_len] + (1.0 - alpha) * signal_b[:min_len]


def spec_augment_mask(n_frames: int, n_freq: int, n_time_masks: int = 1,
                      n_freq_masks: int = 1, time_width: int = 10,
                      freq_width: int = 5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a SpecAugment-style binary mask for a spectrogram (n_freq x n_frames).

    Returns a mask where 1 = keep, 0 = masked.
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = np.ones((n_freq, n_frames), dtype=np.float64)
    for _ in range(n_time_masks):
        start = rng.integers(0, max(1, n_frames - time_width))
        mask[:, start:start + time_width] = 0.0
    for _ in range(n_freq_masks):
        start = rng.integers(0, max(1, n_freq - freq_width))
        mask[start:start + freq_width, :] = 0.0
    return mask


@dataclass
class AugmentationPipeline:
    """Sequence of augmentation steps."""
    steps: list[AugmentationConfig]

    def n_steps(self) -> int:
        return len(self.steps)

    def active_types(self) -> list[AugmentationType]:
        return [s.aug_type for s in self.steps if s.probability > 0]


def apply_pipeline(signal: np.ndarray, pipeline: AugmentationPipeline,
                   sample_rate: int = 44100, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply an augmentation pipeline to a signal."""
    if rng is None:
        rng = np.random.default_rng()
    result = signal.copy()
    for step in pipeline.steps:
        if rng.random() > step.probability:
            continue
        param = rng.uniform(step.min_param, step.max_param)
        if step.aug_type == AugmentationType.PITCH_SHIFT:
            result = pitch_shift_semitones(result, param, sample_rate)
        elif step.aug_type == AugmentationType.TIME_STRETCH:
            rate = max(0.5, min(2.0, 1.0 + param / 10.0))
            stretched = time_stretch(result, rate)
            if len(stretched) > len(result):
                result = stretched[:len(result)]
            else:
                out = np.zeros_like(result)
                out[:len(stretched)] = stretched
                result = out
        elif step.aug_type == AugmentationType.NOISE_INJECTION:
            snr = max(5.0, 30.0 + param * 5.0)
            result = add_noise(result, snr, rng)
        elif step.aug_type == AugmentationType.GAIN_CHANGE:
            result = gain_change_db(result, param)
        elif step.aug_type == AugmentationType.POLARITY_INVERT:
            result = polarity_invert(result)
        elif step.aug_type == AugmentationType.LOW_PASS:
            result = low_pass_simple(result, max(0.1, min(0.9, 0.5 + param / 10.0)))
        elif step.aug_type == AugmentationType.HIGH_PASS:
            result = high_pass_simple(result, max(0.01, min(0.5, 0.1 + param / 20.0)))
        elif step.aug_type == AugmentationType.RANDOM_EQ:
            result = random_eq(result, max_db=abs(param) + 1.0, rng=rng)
    return result
