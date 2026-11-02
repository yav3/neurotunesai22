"""Loudness war analysis: metrics for over-compression and loudness maximization.

Quantifies the degree of dynamic range destruction typical of the "loudness
war" era (≈1995–present).  Metrics include crest factor trends, peak-to-
loudness ratio (PLR), peak signal ratio (PSR), inter-sample peak detection,
waveform clipping analysis, and temporal loudness fatigue estimation.

Pure numpy/scipy — no external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence
from scipy.signal import resample_poly


@dataclass
class ClippingStats:
    clipped_samples: int
    total_samples: int
    clipping_ratio: float
    max_consecutive_clip: int
    clip_regions: List[Tuple[int, int]]


@dataclass
class InterSamplePeak:
    position: int
    true_peak_db: float
    sample_peak_db: float
    overshoot_db: float


@dataclass
class LoudnessWarProfile:
    crest_factor_db: float
    plr_db: float
    psr_db: float
    dynamic_spread_db: float
    clipping: ClippingStats
    inter_sample_peaks: List[InterSamplePeak]
    rms_db: float
    peak_db: float
    loudness_fatigue_index: float
    era_classification: str


def rms_level(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def peak_level(signal: np.ndarray) -> float:
    return float(np.max(np.abs(signal)))


def to_db(value: float, ref: float = 1.0) -> float:
    if value < 1e-12:
        return -120.0
    return float(20.0 * np.log10(value / ref))


def crest_factor(signal: np.ndarray) -> float:
    rms = rms_level(signal)
    peak = peak_level(signal)
    if rms < 1e-12:
        return 0.0
    return float(peak / rms)


def crest_factor_db(signal: np.ndarray) -> float:
    cf = crest_factor(signal)
    if cf < 1e-12:
        return 0.0
    return float(20.0 * np.log10(cf))


def windowed_crest_factor(
    signal: np.ndarray, window_size: int = 4096, hop: int = 2048,
) -> np.ndarray:
    n_frames = max(1, (len(signal) - window_size) // hop + 1)
    cfs = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start : start + window_size]
        cfs[i] = crest_factor_db(frame)
    return cfs


def peak_to_loudness_ratio(signal: np.ndarray, sr: int = 44100) -> float:
    """PLR: difference between true peak and integrated loudness (EBU R128 approximation)."""
    peak = peak_level(signal)
    peak_db = to_db(peak)
    block_size = int(0.4 * sr)
    hop = int(0.1 * sr)
    if len(signal) < block_size:
        rms = rms_level(signal)
        return peak_db - to_db(rms)
    n_blocks = max(1, (len(signal) - block_size) // hop + 1)
    block_loudness = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        block = signal[start : start + block_size]
        block_loudness[i] = np.mean(block ** 2)
    abs_threshold = -70.0
    above_abs = block_loudness[block_loudness > 10 ** (abs_threshold / 10)]
    if len(above_abs) == 0:
        return 0.0
    gamma_r = np.mean(above_abs)
    relative_threshold_lin = gamma_r * 10 ** (-10.0 / 10.0)
    gated = above_abs[above_abs >= relative_threshold_lin]
    if len(gated) == 0:
        return 0.0
    integrated_loudness_db = to_db(float(np.sqrt(np.mean(gated))))
    return peak_db - integrated_loudness_db


def peak_signal_ratio(signal: np.ndarray) -> float:
    """PSR: ratio of peak to RMS in dB — lower means more compressed."""
    rms = rms_level(signal)
    peak = peak_level(signal)
    return to_db(peak) - to_db(rms)


def detect_clipping(
    signal: np.ndarray, threshold: float = 0.99,
) -> ClippingStats:
    clipped_mask = np.abs(signal) >= threshold
    clipped_count = int(np.sum(clipped_mask))
    total = len(signal)
    ratio = clipped_count / max(total, 1)
    max_consec = 0
    current = 0
    regions: List[Tuple[int, int]] = []
    region_start = -1
    for i, c in enumerate(clipped_mask):
        if c:
            if current == 0:
                region_start = i
            current += 1
            max_consec = max(max_consec, current)
        else:
            if current > 0:
                regions.append((region_start, i - 1))
            current = 0
    if current > 0:
        regions.append((region_start, len(signal) - 1))
    return ClippingStats(
        clipped_samples=clipped_count,
        total_samples=total,
        clipping_ratio=ratio,
        max_consecutive_clip=max_consec,
        clip_regions=regions,
    )


def detect_inter_sample_peaks(
    signal: np.ndarray, oversample_factor: int = 4, threshold_db: float = 0.0,
) -> List[InterSamplePeak]:
    """Find inter-sample peaks that exceed 0 dBFS when reconstructed."""
    if len(signal) < 4:
        return []
    oversampled = resample_poly(signal, oversample_factor, 1)
    peaks: List[InterSamplePeak] = []
    sample_peak = peak_level(signal)
    sample_peak_db = to_db(sample_peak)
    for i in range(1, len(oversampled) - 1):
        val = abs(oversampled[i])
        if val > abs(oversampled[i - 1]) and val > abs(oversampled[i + 1]):
            true_peak_db = to_db(val)
            if true_peak_db > threshold_db:
                orig_pos = int(i / oversample_factor)
                overshoot = true_peak_db - sample_peak_db
                peaks.append(InterSamplePeak(
                    position=orig_pos,
                    true_peak_db=true_peak_db,
                    sample_peak_db=sample_peak_db,
                    overshoot_db=overshoot,
                ))
    peaks.sort(key=lambda p: -p.true_peak_db)
    return peaks[:50]


def dynamic_spread(signal: np.ndarray, sr: int = 44100) -> float:
    """Difference between 95th and 5th percentile of short-time RMS in dB."""
    window = int(0.05 * sr)
    hop = window // 2
    n_frames = max(1, (len(signal) - window) // hop + 1)
    rms_vals = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start : start + window]
        rms_vals[i] = float(np.sqrt(np.mean(frame ** 2)))
    rms_vals = rms_vals[rms_vals > 1e-10]
    if len(rms_vals) < 2:
        return 0.0
    db_vals = 20.0 * np.log10(rms_vals)
    return float(np.percentile(db_vals, 95) - np.percentile(db_vals, 5))


def loudness_fatigue_index(
    signal: np.ndarray, sr: int = 44100, block_sec: float = 10.0,
) -> float:
    """Estimate perceptual fatigue: fraction of blocks above -10 dBFS RMS.

    High values (>0.7) correlate with listener fatigue from sustained loudness.
    """
    block_size = int(block_sec * sr)
    if len(signal) < block_size:
        rms = rms_level(signal)
        return 1.0 if to_db(rms) > -10.0 else 0.0
    n_blocks = len(signal) // block_size
    loud_count = 0
    for i in range(n_blocks):
        block = signal[i * block_size : (i + 1) * block_size]
        if to_db(rms_level(block)) > -10.0:
            loud_count += 1
    return loud_count / max(n_blocks, 1)


def classify_era(crest_db: float, plr_db: float) -> str:
    """Classify production era by dynamic characteristics.

    Based on Vickers (2010) and Deruty & Tardieu (2014) loudness war taxonomy.
    """
    if crest_db > 14.0 and plr_db > 18.0:
        return "pre-loudness-war"
    if crest_db > 10.0 and plr_db > 12.0:
        return "early-compression"
    if crest_db > 6.0:
        return "moderate-loudness-war"
    return "severe-loudness-war"


def analyze_loudness_war(
    signal: np.ndarray, sr: int = 44100,
) -> LoudnessWarProfile:
    """Full loudness-war analysis of an audio signal."""
    cf_db = crest_factor_db(signal)
    plr = peak_to_loudness_ratio(signal, sr)
    psr = peak_signal_ratio(signal)
    spread = dynamic_spread(signal, sr)
    clipping = detect_clipping(signal)
    isp = detect_inter_sample_peaks(signal)
    rms = to_db(rms_level(signal))
    peak = to_db(peak_level(signal))
    fatigue = loudness_fatigue_index(signal, sr)
    era = classify_era(cf_db, plr)
    return LoudnessWarProfile(
        crest_factor_db=cf_db,
        plr_db=plr,
        psr_db=psr,
        dynamic_spread_db=spread,
        clipping=clipping,
        inter_sample_peaks=isp,
        rms_db=rms,
        peak_db=peak,
        loudness_fatigue_index=fatigue,
        era_classification=era,
    )


def batch_crest_factor_trend(
    signals: Sequence[np.ndarray], years: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Compute crest factor for a batch; optionally fit a linear trend over years.

    Returns (crest_factors, years_array, slope_db_per_year or None).
    """
    cfs = np.array([crest_factor_db(s) for s in signals])
    if years is not None:
        yrs = np.array(years, dtype=float)
        if len(yrs) >= 2:
            coeffs = np.polyfit(yrs, cfs, 1)
            return cfs, yrs, float(coeffs[0])
        return cfs, yrs, None
    return cfs, np.arange(len(signals), dtype=float), None
