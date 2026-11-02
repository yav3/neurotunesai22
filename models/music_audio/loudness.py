"""Perceptual loudness: LUFS approximation, loudness range, true peak."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoudnessResult:
    integrated_lufs: float
    short_term_lufs: np.ndarray
    momentary_lufs: np.ndarray
    loudness_range_lu: float
    true_peak_dbtp: float
    sample_peak_dbfs: float


def _k_weight_filter(signal: np.ndarray, sr: int) -> np.ndarray:
    """Approximate ITU-R BS.1770 K-weighting via two biquad stages.

    Stage 1: high-shelf boost (+4 dB at high frequencies).
    Stage 2: high-pass at ~38 Hz (head-related transfer).
    """
    x = signal.astype(float)
    if sr <= 0:
        return x

    fc_shelf = 1500.0
    gain_db = 4.0
    w0 = 2.0 * math.pi * fc_shelf / sr
    A = 10.0 ** (gain_db / 40.0)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / 0.7071 - 1) + 2)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha

    y = _biquad(x, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    fc_hp = 38.0
    w0_hp = 2.0 * math.pi * fc_hp / sr
    cos_hp = math.cos(w0_hp)
    sin_hp = math.sin(w0_hp)
    alpha_hp = sin_hp / (2.0 * 0.5)

    b0_hp = (1 + cos_hp) / 2
    b1_hp = -(1 + cos_hp)
    b2_hp = (1 + cos_hp) / 2
    a0_hp = 1 + alpha_hp
    a1_hp = -2 * cos_hp
    a2_hp = 1 - alpha_hp

    return _biquad(y, b0_hp / a0_hp, b1_hp / a0_hp, b2_hp / a0_hp, a1_hp / a0_hp, a2_hp / a0_hp)


def _biquad(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        y[i] = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x[i]
        y2, y1 = y1, y[i]
    return y


def _block_rms_db(signal: np.ndarray, block_size: int, hop: int) -> np.ndarray:
    n = len(signal)
    n_blocks = max(1, (n - block_size) // hop + 1)
    result = np.full(n_blocks, -70.0)
    for i in range(n_blocks):
        start = i * hop
        block = signal[start:start + block_size]
        ms = float(np.mean(block ** 2))
        if ms > 1e-30:
            result[i] = -0.691 + 10.0 * math.log10(ms)
    return result


def integrated_loudness(signal: np.ndarray, sr: int) -> float:
    """Approximate integrated loudness in LUFS (mono signal)."""
    filtered = _k_weight_filter(signal, sr)
    block_size = int(0.4 * sr)
    hop = int(0.1 * sr)
    if block_size < 1 or hop < 1:
        return -70.0
    blocks = _block_rms_db(filtered, block_size, hop)

    abs_threshold = -70.0
    above = blocks[blocks > abs_threshold]
    if len(above) == 0:
        return -70.0
    relative_threshold = float(np.mean(above)) - 10.0
    gated = above[above > relative_threshold]
    if len(gated) == 0:
        return -70.0
    return float(np.mean(gated))


def true_peak_dbtp(signal: np.ndarray, oversample: int = 4) -> float:
    """Estimate true peak via linear interpolation oversampling."""
    if len(signal) == 0:
        return -np.inf
    n = len(signal)
    x = np.arange(n)
    x_up = np.linspace(0, n - 1, n * oversample)
    upsampled = np.interp(x_up, x, signal.astype(float))
    peak = float(np.max(np.abs(upsampled)))
    if peak < 1e-30:
        return -70.0
    return float(20.0 * math.log10(peak))


def measure_loudness(signal: np.ndarray, sr: int) -> LoudnessResult:
    """Full loudness measurement suite."""
    filtered = _k_weight_filter(signal, sr)

    block_400ms = max(int(0.4 * sr), 1)
    hop_100ms = max(int(0.1 * sr), 1)
    momentary_blocks = _block_rms_db(filtered, block_400ms, hop_100ms)

    block_3s = max(int(3.0 * sr), 1)
    hop_1s = max(int(1.0 * sr), 1)
    short_term_blocks = _block_rms_db(filtered, block_3s, hop_1s)

    il = integrated_loudness(signal, sr)

    lra = 0.0
    if len(short_term_blocks) >= 2:
        valid = short_term_blocks[short_term_blocks > -70.0]
        if len(valid) >= 2:
            lra = float(np.percentile(valid, 95) - np.percentile(valid, 10))

    tp = true_peak_dbtp(signal)
    sp = float(20.0 * math.log10(max(float(np.max(np.abs(signal))), 1e-30)))

    return LoudnessResult(
        integrated_lufs=il,
        short_term_lufs=short_term_blocks,
        momentary_lufs=momentary_blocks,
        loudness_range_lu=lra,
        true_peak_dbtp=tp,
        sample_peak_dbfs=sp,
    )
