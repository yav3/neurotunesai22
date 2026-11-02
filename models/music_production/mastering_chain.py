"""Mastering chain: multi-band processing, loudness targeting, stereo enhancement."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MasteringStage(Enum):
    EQ = "eq"
    MULTIBAND_COMPRESS = "multiband_compress"
    STEREO_ENHANCE = "stereo_enhance"
    LIMIT = "limit"
    DITHER = "dither"


@dataclass
class BandConfig:
    low_hz: float
    high_hz: float
    threshold_db: float = -12.0
    ratio: float = 3.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    makeup_db: float = 0.0


@dataclass
class MasteringConfig:
    target_lufs: float = -14.0
    true_peak_limit_dbtp: float = -1.0
    bands: list[BandConfig] = field(default_factory=lambda: [
        BandConfig(20, 250, -18.0, 2.5, 20.0, 200.0, 1.0),
        BandConfig(250, 2000, -14.0, 3.0, 10.0, 100.0, 0.0),
        BandConfig(2000, 8000, -12.0, 3.5, 5.0, 80.0, 0.5),
        BandConfig(8000, 20000, -10.0, 2.0, 3.0, 60.0, 0.0),
    ])
    stereo_width: float = 1.0
    dither_bits: int = 16


@dataclass
class MasteringResult:
    output: np.ndarray
    input_lufs: float
    output_lufs: float
    gain_applied_db: float
    true_peak_dbtp: float
    band_gains_db: list[float]


def _rms_db(signal: np.ndarray) -> float:
    ms = np.mean(signal ** 2)
    return 10.0 * np.log10(max(ms, 1e-10))


def _lufs_approx(signal: np.ndarray, sr: int) -> float:
    ms = np.mean(signal ** 2)
    return -0.691 + 10.0 * np.log10(max(ms, 1e-10))


def _simple_bandpass(signal: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    S = np.fft.rfft(signal)
    mask = ((freqs >= low) & (freqs <= high)).astype(float)
    rolloff = 12
    for i, f in enumerate(freqs):
        if f < low:
            mask[i] = max(0.0, 1.0 - (low - f) / rolloff) if low - f < rolloff else 0.0
        elif f > high:
            mask[i] = max(0.0, 1.0 - (f - high) / rolloff) if f - high < rolloff else 0.0
    return np.fft.irfft(S * mask, n)


def _compress_band(signal: np.ndarray, config: BandConfig, sr: int) -> tuple[np.ndarray, float]:
    level = _rms_db(signal)
    over = level - config.threshold_db
    if over <= 0:
        gain = config.makeup_db
    else:
        reduction = over * (1.0 - 1.0 / config.ratio)
        gain = config.makeup_db - reduction
    factor = 10.0 ** (gain / 20.0)
    return signal * factor, gain


def apply_multiband_compression(
    signal: np.ndarray, sr: int, bands: list[BandConfig],
) -> tuple[np.ndarray, list[float]]:
    output = np.zeros_like(signal)
    gains = []
    for band in bands:
        band_signal = _simple_bandpass(signal, sr, band.low_hz, band.high_hz)
        compressed, gain = _compress_band(band_signal, band, sr)
        output += compressed
        gains.append(gain)
    return output, gains


def stereo_enhance(left: np.ndarray, right: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    side *= width
    return mid + side, mid - side


def true_peak(signal: np.ndarray, oversample: int = 4) -> float:
    n = len(signal)
    padded = np.zeros(n * oversample)
    padded[::oversample] = signal
    kernel_size = oversample * 8 + 1
    kernel = np.sinc(np.linspace(-4, 4, kernel_size))
    kernel *= np.hanning(kernel_size)
    kernel /= np.sum(kernel)
    upsampled = np.convolve(padded, kernel, mode="same")
    peak = np.max(np.abs(upsampled))
    return 20.0 * np.log10(max(peak, 1e-10))


def limiter(signal: np.ndarray, ceiling_db: float, sr: int) -> np.ndarray:
    ceiling = 10.0 ** (ceiling_db / 20.0)
    output = np.copy(signal)
    peak = np.max(np.abs(output))
    if peak > ceiling:
        output *= ceiling / peak
    return output


def dither(signal: np.ndarray, bits: int = 16) -> np.ndarray:
    step = 2.0 / (2 ** bits)
    noise = np.random.triangular(-step, 0, step, len(signal))
    return signal + noise


def master(signal: np.ndarray, sr: int, config: MasteringConfig | None = None) -> MasteringResult:
    if config is None:
        config = MasteringConfig()
    input_lufs = _lufs_approx(signal, sr)
    processed, band_gains = apply_multiband_compression(signal, sr, config.bands)
    processed = limiter(processed, config.true_peak_limit_dbtp, sr)
    current_lufs = _lufs_approx(processed, sr)
    gain_needed = config.target_lufs - current_lufs
    gain_needed = np.clip(gain_needed, -12.0, 12.0)
    factor = 10.0 ** (gain_needed / 20.0)
    processed *= factor
    processed = limiter(processed, config.true_peak_limit_dbtp, sr)
    if config.dither_bits < 24:
        processed = dither(processed, config.dither_bits)
    output_lufs = _lufs_approx(processed, sr)
    tp = true_peak(processed)
    return MasteringResult(
        output=processed, input_lufs=input_lufs, output_lufs=output_lufs,
        gain_applied_db=gain_needed, true_peak_dbtp=tp, band_gains_db=band_gains,
    )
