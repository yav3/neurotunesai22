"""Dynamic range analysis and processing.

Crest factor, LUFS-based loudness, dynamic range compression modelling,
loudness normalisation, and micro-dynamic analysis.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DynamicRangeStats:
    peak_db: float
    rms_db: float
    crest_factor_db: float
    dynamic_range_db: float
    loudness_lufs: float


def rms_level(signal: np.ndarray) -> float:
    """Root mean square level."""
    return float(np.sqrt(np.mean(signal ** 2)))


def peak_level(signal: np.ndarray) -> float:
    return float(np.max(np.abs(signal)))


def crest_factor(signal: np.ndarray) -> float:
    """Crest factor = peak / RMS, in dB."""
    rms = rms_level(signal)
    peak = peak_level(signal)
    if rms < 1e-12:
        return 0.0
    return 20.0 * np.log10(peak / rms)


def lufs_approximate(signal: np.ndarray, sr: int = 44100) -> float:
    """Approximate integrated loudness in LUFS (ITU-R BS.1770 simplified).

    Uses K-weighted RMS over 400 ms gating windows with −70 LUFS absolute gate.
    """
    # K-weighting: high-shelf at 1681 Hz + high-pass at 38 Hz (simplified)
    from scipy.signal import butter, sosfilt

    nyq = sr / 2.0
    # Stage 1: high-shelf approximation
    sos_hs = butter(2, 1681.0 / nyq, btype='high', output='sos')
    weighted = sosfilt(sos_hs, signal) * 1.5 + signal * 0.5  # approximate shelf boost

    # Stage 2: high-pass at 38 Hz
    if 38.0 / nyq < 1.0:
        sos_hp = butter(2, 38.0 / nyq, btype='high', output='sos')
        weighted = sosfilt(sos_hp, weighted)

    # Gating with 400 ms blocks, 75% overlap
    block_samples = int(0.4 * sr)
    hop = block_samples // 4
    n_blocks = (len(weighted) - block_samples) // hop + 1
    if n_blocks < 1:
        ms = np.mean(weighted ** 2)
        return -0.691 + 10.0 * np.log10(max(ms, 1e-12))

    block_loudness = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        block = weighted[start:start + block_samples]
        block_loudness[i] = np.mean(block ** 2)

    # Absolute gate: −70 LUFS
    abs_gate = 10 ** ((-70 + 0.691) / 10)
    gated = block_loudness[block_loudness > abs_gate]
    if len(gated) == 0:
        return -70.0

    # Relative gate: −10 dB below ungated mean
    mean_above_abs = np.mean(gated)
    rel_gate = mean_above_abs * 10 ** (-10 / 10)
    final = gated[gated > rel_gate]
    if len(final) == 0:
        return -70.0

    return -0.691 + 10.0 * np.log10(np.mean(final))


def dynamic_range_dr(signal: np.ndarray, sr: int = 44100) -> float:
    """Estimate dynamic range (DR) using the Pleasurize Music Foundation method.

    Approximate: 20 * log10(RMS of top 20% / RMS of bottom 20%).
    """
    frame_len = int(3.0 * sr)  # 3-second frames
    hop = frame_len
    n_frames = max(1, len(signal) // hop)
    rms_values = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_len]
        rms_values[i] = rms_level(frame)

    sorted_rms = np.sort(rms_values)
    n = len(sorted_rms)
    top_20 = sorted_rms[int(0.8 * n):]
    bot_20 = sorted_rms[:max(1, int(0.2 * n))]

    rms_top = np.mean(top_20) if len(top_20) > 0 else 1e-12
    rms_bot = np.mean(bot_20) if len(bot_20) > 0 else 1e-12
    if rms_bot < 1e-12:
        return 60.0
    return 20.0 * np.log10(rms_top / rms_bot)


def loudness_normalise(
    signal: np.ndarray,
    target_lufs: float = -14.0,
    sr: int = 44100,
) -> np.ndarray:
    """Normalise signal to target integrated loudness."""
    current = lufs_approximate(signal, sr)
    if current < -69:
        return signal
    gain_db = target_lufs - current
    gain_linear = 10 ** (gain_db / 20.0)
    normalised = signal * gain_linear
    # true-peak limiter at −1 dBFS
    peak = np.max(np.abs(normalised))
    ceiling = 10 ** (-1.0 / 20.0)
    if peak > ceiling:
        normalised *= ceiling / peak
    return normalised


def analyze_dynamic_range(signal: np.ndarray, sr: int = 44100) -> DynamicRangeStats:
    """Full dynamic range analysis."""
    pk = peak_level(signal)
    rms = rms_level(signal)
    cf = crest_factor(signal)
    dr = dynamic_range_dr(signal, sr)
    lufs = lufs_approximate(signal, sr)
    return DynamicRangeStats(
        peak_db=20.0 * np.log10(max(pk, 1e-12)),
        rms_db=20.0 * np.log10(max(rms, 1e-12)),
        crest_factor_db=cf,
        dynamic_range_db=dr,
        loudness_lufs=lufs,
    )
