"""Loudness normalization and metering per EBU R128 / ITU-R BS.1770.

K-weighted integrated loudness, loudness range (LRA), true-peak
detection, and target-LUFS normalization.  Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class MeterType(Enum):
    PEAK = "peak"
    RMS = "rms"
    LUFS = "lufs"
    TRUE_PEAK = "true_peak"


@dataclass
class LoudnessResult:
    integrated_lufs: float
    range_lu: float
    peak_dbfs: float
    true_peak_dbfs: float


def k_weighting_coefficients(sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two-stage K-weighting filter coefficients (pre-emphasis + RLB).

    Returns (stage1_coeffs, stage2_coeffs) each as [b0,b1,b2,a1,a2].
    Approximation valid for sr >= 44100.
    """
    f0 = 1681.97
    Q = 0.7071
    G = 3.999
    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.5
    a0 = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0
    b1 = 2.0 * (K * K - Vh) / a0
    b2 = (Vh - Vb * K / Q + K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    stage1 = np.array([b0, b1, b2, a1, a2])

    f1 = 38.13
    Q2 = 0.5003
    K2 = np.tan(np.pi * f1 / sr)
    a0_2 = 1.0 + K2 / Q2 + K2 * K2
    b0_2 = 1.0 / a0_2
    b1_2 = -2.0 / a0_2
    b2_2 = 1.0 / a0_2
    a1_2 = 2.0 * (K2 * K2 - 1.0) / a0_2
    a2_2 = (1.0 - K2 / Q2 + K2 * K2) / a0_2
    stage2 = np.array([b0_2, b1_2, b2_2, a1_2, a2_2])

    return stage1, stage2


def _apply_biquad(signal: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    b0, b1, b2, a1, a2 = coeffs
    n = len(signal)
    out = np.zeros(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        x0 = signal[i]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return out


def apply_k_weighting(signal: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Apply K-weighting filter to a mono signal."""
    s1, s2 = k_weighting_coefficients(sr)
    return _apply_biquad(_apply_biquad(signal, s1), s2)


def momentary_loudness(signal: np.ndarray, sr: int = 48000,
                       block_ms: int = 400) -> np.ndarray:
    """Compute 400 ms gated loudness blocks (LUFS per block)."""
    weighted = apply_k_weighting(signal, sr)
    block_n = int(sr * block_ms / 1000)
    hop = block_n  # non-overlapping for momentary
    n_blocks = max(1, (len(weighted) - block_n) // hop + 1)
    lufs = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        block = weighted[start:start + block_n]
        ms = float(np.mean(block ** 2))
        lufs[i] = -0.691 + 10.0 * np.log10(max(ms, 1e-20))
    return lufs


def short_term_loudness(signal: np.ndarray, sr: int = 48000,
                        block_ms: int = 3000) -> np.ndarray:
    """3-second overlapping loudness blocks."""
    weighted = apply_k_weighting(signal, sr)
    block_n = int(sr * block_ms / 1000)
    hop = int(sr * 0.1)  # 100 ms hop
    n_blocks = max(1, (len(weighted) - block_n) // hop + 1)
    lufs = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        block = weighted[start:start + block_n]
        ms = float(np.mean(block ** 2))
        lufs[i] = -0.691 + 10.0 * np.log10(max(ms, 1e-20))
    return lufs


def loudness_gate(momentary_values: np.ndarray,
                  absolute_threshold: float = -70.0,
                  relative_offset: float = -10.0) -> np.ndarray:
    """EBU R128 two-stage gate. Returns mask of ungated blocks."""
    abs_mask = momentary_values >= absolute_threshold
    if not np.any(abs_mask):
        return np.zeros(len(momentary_values), dtype=bool)
    relative_threshold = float(np.mean(
        10 ** (momentary_values[abs_mask] / 10.0)
    ))
    relative_threshold_lufs = 10.0 * np.log10(max(relative_threshold, 1e-20)) + relative_offset
    return momentary_values >= relative_threshold_lufs


def integrated_loudness(signal: np.ndarray, sr: int = 48000) -> float:
    """EBU R128 integrated loudness (LUFS) with two-stage gating."""
    mom = momentary_loudness(signal, sr, block_ms=400)
    mask = loudness_gate(mom)
    if not np.any(mask):
        return -70.0
    power = 10 ** (mom[mask] / 10.0)
    return float(10.0 * np.log10(max(np.mean(power), 1e-20)))


def loudness_range(signal: np.ndarray, sr: int = 48000) -> float:
    """Loudness range (LRA) in LU per EBU R128."""
    stl = short_term_loudness(signal, sr, block_ms=3000)
    mask = loudness_gate(stl, absolute_threshold=-70.0, relative_offset=-20.0)
    if np.sum(mask) < 2:
        return 0.0
    gated = np.sort(stl[mask])
    lo = gated[int(len(gated) * 0.10)]
    hi = gated[int(len(gated) * 0.95)]
    return float(hi - lo)


def true_peak(signal: np.ndarray, sr: int = 48000,
              oversample: int = 4) -> float:
    """Inter-sample true peak via linear interpolation upsampling."""
    n = len(signal)
    x = np.arange(n)
    x_up = np.linspace(0, n - 1, n * oversample)
    upsampled = np.interp(x_up, x, signal)
    return float(np.max(np.abs(upsampled)))


def normalize_loudness(signal: np.ndarray, sr: int = 48000,
                       target_lufs: float = -14.0) -> np.ndarray:
    """Normalize a signal to a target integrated LUFS."""
    current = integrated_loudness(signal, sr)
    if current <= -70.0:
        return signal.copy()
    gain_db = target_lufs - current
    gain_linear = 10 ** (gain_db / 20.0)
    return signal * gain_linear


def rms_db(signal: np.ndarray) -> float:
    """RMS level in dBFS."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    return 20.0 * np.log10(max(rms, 1e-20))


def peak_db(signal: np.ndarray) -> float:
    """Sample peak in dBFS."""
    pk = float(np.max(np.abs(signal)))
    return 20.0 * np.log10(max(pk, 1e-20))


def crest_factor(signal: np.ndarray) -> float:
    """Crest factor: peak/RMS ratio in dB."""
    return peak_db(signal) - rms_db(signal)


def measure_loudness(signal: np.ndarray, sr: int = 48000) -> LoudnessResult:
    """Full loudness measurement returning integrated, range, peak, true peak."""
    return LoudnessResult(
        integrated_lufs=integrated_loudness(signal, sr),
        range_lu=loudness_range(signal, sr),
        peak_dbfs=peak_db(signal),
        true_peak_dbfs=20.0 * np.log10(max(true_peak(signal, sr), 1e-20)),
    )


def dynamic_range_meter(signal: np.ndarray, sr: int = 48000,
                        window_ms: int = 3000) -> np.ndarray:
    """Time-varying dynamic range (peak - RMS per window) in dB."""
    win_n = int(sr * window_ms / 1000)
    hop = win_n // 2
    n_frames = max(1, (len(signal) - win_n) // hop + 1)
    dr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        block = signal[start:start + win_n]
        pk = float(np.max(np.abs(block)))
        rms = float(np.sqrt(np.mean(block ** 2)))
        pk_db = 20.0 * np.log10(max(pk, 1e-20))
        rms_db_val = 20.0 * np.log10(max(rms, 1e-20))
        dr[i] = pk_db - rms_db_val
    return dr
