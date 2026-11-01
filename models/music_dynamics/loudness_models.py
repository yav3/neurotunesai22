"""Perceptual loudness: Stevens' power law, EBU R128, equal-loudness contours.

Implements loudness measurement aligned with psychoacoustic standards:
phon/sone conversion, momentary/short-term/integrated loudness (EBU R128
simplified), loudness range (LRA), and dynamic range metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoudnessProfile:
    peak_db: float
    rms_db: float
    crest_factor_db: float
    dynamic_range_db: float
    loudness_sone: float
    loudness_phon: float


@dataclass(frozen=True)
class EBULoudness:
    momentary_lufs: float
    short_term_lufs: float
    integrated_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float


def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def amplitude_to_db(amp: float) -> float:
    if amp < 1e-12:
        return -120.0
    return 20.0 * math.log10(amp)


def phon_to_sone(phon: float) -> float:
    """Convert loudness in phons to sones (Stevens' power law)."""
    if phon < 40.0:
        return max(0.0, (phon / 40.0) ** 2.86 * 1.0)
    return 2.0 ** ((phon - 40.0) / 10.0)


def sone_to_phon(sone: float) -> float:
    """Convert loudness in sones to phons."""
    if sone <= 0.0:
        return 0.0
    if sone < 1.0:
        return 40.0 * (sone ** (1.0 / 2.86))
    return 40.0 + 10.0 * math.log2(sone)


def a_weighting(freq_hz: float) -> float:
    """A-weighting filter magnitude at a given frequency (IEC 61672)."""
    f2 = freq_hz ** 2
    num = 12194.0 ** 2 * f2 ** 2
    den = ((f2 + 20.6 ** 2) *
           math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
           (f2 + 12194.0 ** 2))
    if den < 1e-30:
        return 0.0
    ra = num / den
    return 20.0 * math.log10(max(ra, 1e-12)) + 2.0


def k_weighting_filter(freqs: np.ndarray) -> np.ndarray:
    """K-weighting filter for EBU R128 loudness measurement.

    Two-stage filter: high-shelf at 1681 Hz (+3.999 dB) then
    high-pass at 38 Hz (second order). Approximated in frequency domain.
    """
    gains = np.ones(len(freqs))
    shelf_freq = 1681.0
    shelf_gain_db = 3.999
    for i, f in enumerate(freqs):
        if f < 1.0:
            gains[i] = 0.0
            continue
        shelf = 1.0 + (10.0 ** (shelf_gain_db / 20.0) - 1.0) * (f ** 2 / (f ** 2 + shelf_freq ** 2))
        hp = f ** 2 / (f ** 2 + 38.0 ** 2)
        gains[i] = shelf * hp ** 2
    return gains


def momentary_loudness(
    signal: np.ndarray,
    sr: float,
    window_ms: float = 400.0,
    hop_ms: float = 100.0,
) -> np.ndarray:
    """Momentary loudness (EBU R128): mean square over 400 ms windows."""
    win_samples = int(sr * window_ms / 1000.0)
    hop_samples = int(sr * hop_ms / 1000.0)
    if win_samples < 1 or len(signal) < win_samples:
        return np.array([-70.0])
    n_frames = (len(signal) - win_samples) // hop_samples + 1
    loudness = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_samples
        frame = signal[start:start + win_samples]
        ms = np.mean(frame ** 2)
        loudness[i] = -0.691 + 10.0 * math.log10(max(ms, 1e-12))
    return loudness


def integrated_loudness(momentary_values: np.ndarray) -> float:
    """Gated integrated loudness per EBU R128.

    Two-pass gating: absolute gate at -70 LUFS, then relative gate at -10 LU
    below the ungated mean.
    """
    abs_gate = -70.0
    above_abs = momentary_values[momentary_values > abs_gate]
    if len(above_abs) == 0:
        return -70.0
    ungated_mean = 10.0 * math.log10(np.mean(10.0 ** (above_abs / 10.0)))
    rel_gate = ungated_mean - 10.0
    above_rel = above_abs[above_abs > rel_gate]
    if len(above_rel) == 0:
        return -70.0
    return float(-0.691 + 10.0 * math.log10(np.mean(10.0 ** (above_rel / 10.0))))


def loudness_range(momentary_values: np.ndarray) -> float:
    """Loudness Range (LRA) per EBU R128: difference between 95th and 10th percentile."""
    abs_gate = -70.0
    above = momentary_values[momentary_values > abs_gate]
    if len(above) < 2:
        return 0.0
    ungated_mean = 10.0 * math.log10(np.mean(10.0 ** (above / 10.0)))
    rel_gate = ungated_mean - 20.0
    gated = above[above > rel_gate]
    if len(gated) < 2:
        return 0.0
    p10 = float(np.percentile(gated, 10))
    p95 = float(np.percentile(gated, 95))
    return p95 - p10


def compute_ebu_loudness(signal: np.ndarray, sr: float) -> EBULoudness:
    """Full EBU R128 loudness analysis of a signal."""
    mom = momentary_loudness(signal, sr, window_ms=400.0, hop_ms=100.0)
    short = momentary_loudness(signal, sr, window_ms=3000.0, hop_ms=100.0)
    integ = integrated_loudness(mom)
    lra = loudness_range(mom)
    true_peak = amplitude_to_db(float(np.max(np.abs(signal))))
    return EBULoudness(
        momentary_lufs=float(np.mean(mom)) if len(mom) > 0 else -70.0,
        short_term_lufs=float(np.mean(short)) if len(short) > 0 else -70.0,
        integrated_lufs=integ,
        loudness_range_lu=lra,
        true_peak_dbtp=true_peak,
    )


def compute_loudness_profile(signal: np.ndarray, sr: float) -> LoudnessProfile:
    """Compute a LoudnessProfile with peak, RMS, crest factor, dynamic range, sones."""
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak_db = amplitude_to_db(peak)
    rms_db = amplitude_to_db(rms)
    crest_db = peak_db - rms_db
    frame_len = int(sr * 0.05)
    if frame_len < 1:
        frame_len = 1
    n_frames = max(1, len(signal) // frame_len)
    frame_rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * frame_len:(i + 1) * frame_len]
        frame_rms[i] = amplitude_to_db(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
    dynamic_range = float(np.percentile(frame_rms, 95) - np.percentile(frame_rms, 5))
    phon_approx = max(0.0, rms_db + 83.0)
    sone = phon_to_sone(phon_approx)
    return LoudnessProfile(
        peak_db=peak_db, rms_db=rms_db, crest_factor_db=crest_db,
        dynamic_range_db=dynamic_range, loudness_sone=sone,
        loudness_phon=phon_approx,
    )
