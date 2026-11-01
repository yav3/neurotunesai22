"""Music-specific data augmentation transforms.

Pitch shifting, time stretching, noise injection, frequency masking,
and polarity inversion for training-set expansion.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AugmentConfig:
    pitch_shift_range: Tuple[int, int] = (-2, 2)  # semitones
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    noise_snr_db: float = 30.0
    freq_mask_max_bands: int = 2
    freq_mask_width: int = 8  # mel bins


def pitch_shift_semitones(
    signal: np.ndarray,
    semitones: int,
    sr: int = 44100,
) -> np.ndarray:
    """Pitch shift via resampling (no time correction)."""
    if semitones == 0:
        return signal.copy()
    factor = 2.0 ** (semitones / 12.0)
    indices = np.arange(0, len(signal), factor)
    indices = indices[indices < len(signal) - 1].astype(int)
    return signal[indices]


def time_stretch_simple(
    signal: np.ndarray,
    rate: float,
) -> np.ndarray:
    """Naive time stretch via linear interpolation."""
    if abs(rate - 1.0) < 1e-6:
        return signal.copy()
    new_len = int(len(signal) / rate)
    old_indices = np.linspace(0, len(signal) - 1, new_len)
    return np.interp(old_indices, np.arange(len(signal)), signal)


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float = 30.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise at specified SNR."""
    rng = np.random.default_rng(seed)
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(max(noise_power, 1e-12)), len(signal))
    return signal + noise


def frequency_mask(
    spectrogram: np.ndarray,
    n_masks: int = 1,
    mask_width: int = 8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply SpecAugment-style frequency masking to a spectrogram."""
    rng = np.random.default_rng(seed)
    n_freq, n_time = spectrogram.shape
    masked = spectrogram.copy()
    for _ in range(n_masks):
        f0 = rng.integers(0, max(1, n_freq - mask_width))
        masked[f0:f0 + mask_width, :] = 0.0
    return masked


def time_mask(
    spectrogram: np.ndarray,
    n_masks: int = 1,
    mask_width: int = 16,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply SpecAugment-style time masking."""
    rng = np.random.default_rng(seed)
    n_freq, n_time = spectrogram.shape
    masked = spectrogram.copy()
    for _ in range(n_masks):
        t0 = rng.integers(0, max(1, n_time - mask_width))
        masked[:, t0:t0 + mask_width] = 0.0
    return masked


def polarity_inversion(signal: np.ndarray) -> np.ndarray:
    """Invert signal polarity (phase flip)."""
    return -signal


def random_gain(
    signal: np.ndarray,
    min_db: float = -6.0,
    max_db: float = 6.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply random gain in dB range."""
    rng = np.random.default_rng(seed)
    gain_db = rng.uniform(min_db, max_db)
    return signal * 10 ** (gain_db / 20.0)


def random_eq_band(
    signal: np.ndarray,
    sr: int = 44100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Random single-band EQ augmentation via FFT."""
    rng = np.random.default_rng(seed)
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    fft = np.fft.rfft(signal)

    center_hz = rng.uniform(100, sr / 2 - 100)
    width_hz = rng.uniform(50, 500)
    gain_db = rng.uniform(-6, 6)
    gain_lin = 10 ** (gain_db / 20.0)

    mask = np.exp(-0.5 * ((freqs - center_hz) / width_hz) ** 2)
    eq_curve = 1.0 + (gain_lin - 1.0) * mask
    return np.fft.irfft(fft * eq_curve, n=n)


def augment_pipeline(
    signal: np.ndarray,
    config: Optional[AugmentConfig] = None,
    sr: int = 44100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Run a random augmentation pipeline."""
    if config is None:
        config = AugmentConfig()
    rng = np.random.default_rng(seed)

    # random pitch shift
    semitones = rng.integers(config.pitch_shift_range[0], config.pitch_shift_range[1] + 1)
    out = pitch_shift_semitones(signal, semitones, sr)

    # random time stretch
    rate = rng.uniform(*config.time_stretch_range)
    out = time_stretch_simple(out, rate)

    # random noise
    out = add_gaussian_noise(out, config.noise_snr_db, seed=rng.integers(0, 2**31))

    # random gain
    out = random_gain(out, seed=rng.integers(0, 2**31))

    return out
