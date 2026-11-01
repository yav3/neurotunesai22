"""Mastering quality metrics: loudness, dynamic range, stereo width, spectral balance.

Implements ITU-R BS.1770 LUFS measurement, EBU R128 loudness range,
true-peak detection via oversampling, and genre-aware spectral balance scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class LoudnessStandard(Enum):
    EBU_R128 = "ebu_r128"
    ATSC_A85 = "atsc_a85"
    ARIB_TR_B32 = "arib_tr_b32"


TARGET_LUFS = {
    LoudnessStandard.EBU_R128: -23.0,
    LoudnessStandard.ATSC_A85: -24.0,
    LoudnessStandard.ARIB_TR_B32: -24.0,
}

GENRE_SPECTRAL_TARGETS: dict[str, np.ndarray] = {
    "pop":       np.array([0.18, 0.22, 0.28, 0.20, 0.12]),
    "rock":      np.array([0.15, 0.25, 0.30, 0.20, 0.10]),
    "classical": np.array([0.10, 0.18, 0.30, 0.25, 0.17]),
    "hip_hop":   np.array([0.25, 0.25, 0.25, 0.15, 0.10]),
    "jazz":      np.array([0.12, 0.20, 0.28, 0.23, 0.17]),
    "edm":       np.array([0.22, 0.24, 0.26, 0.18, 0.10]),
}

BAND_EDGES_HZ = [20.0, 250.0, 2000.0, 6000.0, 12000.0, 20000.0]


# ---------------------------------------------------------------------------
# K-weighting filter (ITU-R BS.1770)
# ---------------------------------------------------------------------------

def _k_weight_coefficients(sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Pre-filter + RLB weighting coefficients for K-weighting at given sample rate."""
    # Stage 1: high-shelf (pre-filter) — +4 dB above ~1.5 kHz
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    denom = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / denom
    b1 = 2.0 * (K * K - Vh) / denom
    b2 = (Vh - Vb * K / Q + K * K) / denom
    a1 = 2.0 * (K * K - 1.0) / denom
    a2 = (1.0 - K / Q + K * K) / denom
    pre_b = np.array([b0, b1, b2])
    pre_a = np.array([1.0, a1, a2])

    # Stage 2: highpass (RLB weighting) — ~60 Hz
    f0_hp = 38.13547087602444
    Q_hp = 0.5003270373238773
    K_hp = np.tan(np.pi * f0_hp / sr)
    denom_hp = 1.0 + K_hp / Q_hp + K_hp * K_hp
    hp_b = np.array([1.0, -2.0, 1.0]) / denom_hp
    hp_a = np.array([1.0,
                     2.0 * (K_hp * K_hp - 1.0) / denom_hp,
                     (1.0 - K_hp / Q_hp + K_hp * K_hp) / denom_hp])

    return np.convolve(pre_b, hp_b), np.convolve(pre_a, hp_a)


def k_weight(audio: np.ndarray, sr: float) -> np.ndarray:
    """Apply K-weighting filter to audio signal."""
    b, a = _k_weight_coefficients(sr)
    return sig.lfilter(b, a, audio)


# ---------------------------------------------------------------------------
# LUFS measurement (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------

@dataclass
class LUFSResult:
    integrated: float
    short_term: np.ndarray   # 3-second windows, 1-second hop
    momentary: np.ndarray    # 400 ms windows
    loudness_range: float    # LRA (EBU R128)
    max_short_term: float
    max_momentary: float


def _gated_mean_square(block_powers: np.ndarray,
                       abs_threshold: float = -70.0) -> float:
    """Two-stage gated loudness per BS.1770-4."""
    # Absolute gate
    above_abs = block_powers[block_powers > abs_threshold]
    if len(above_abs) == 0:
        return -np.inf

    # Relative gate: -10 dB below ungated mean
    ungated_mean = 10.0 * np.log10(np.mean(10.0 ** (above_abs / 10.0)) + 1e-30)
    rel_threshold = ungated_mean - 10.0
    above_rel = above_abs[above_abs > rel_threshold]
    if len(above_rel) == 0:
        return -np.inf

    return 10.0 * np.log10(np.mean(10.0 ** (above_rel / 10.0)) + 1e-30)


def measure_lufs(audio: np.ndarray, sr: float,
                 n_channels: int = 1) -> LUFSResult:
    """Measure integrated, short-term, and momentary LUFS.

    Args:
        audio: mono (N,) or interleaved multi-channel (N*C,)
        sr: sample rate
        n_channels: number of channels (1=mono, 2=stereo)
    """
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        weighted = np.column_stack([k_weight(audio[:, c], sr)
                                    for c in range(n_channels)])
        channel_weights = np.ones(n_channels)
        if n_channels >= 5:
            channel_weights[3:5] = 1.41  # surround +1.5 dB
        mean_sq = np.sum(channel_weights * np.mean(weighted ** 2, axis=0))
    else:
        weighted = k_weight(audio.ravel(), sr)
        mean_sq = np.mean(weighted ** 2)

    integrated_lufs = -0.691 + 10.0 * np.log10(mean_sq + 1e-30)

    # Short-term (3 s window, 1 s hop)
    st_samples = int(3.0 * sr)
    st_hop = int(1.0 * sr)
    st_powers = []
    flat_w = weighted.ravel() if weighted.ndim > 1 else weighted
    for start in range(0, len(flat_w) - st_samples + 1, st_hop):
        block = flat_w[start:start + st_samples]
        p = -0.691 + 10.0 * np.log10(np.mean(block ** 2) + 1e-30)
        st_powers.append(p)
    st_arr = np.array(st_powers) if st_powers else np.array([-np.inf])

    # Momentary (400 ms window)
    m_samples = int(0.4 * sr)
    m_hop = int(0.1 * sr)
    m_powers = []
    for start in range(0, len(flat_w) - m_samples + 1, m_hop):
        block = flat_w[start:start + m_samples]
        p = -0.691 + 10.0 * np.log10(np.mean(block ** 2) + 1e-30)
        m_powers.append(p)
    m_arr = np.array(m_powers) if m_powers else np.array([-np.inf])

    # LRA (EBU R128): difference between 95th and 10th percentile of short-term
    finite_st = st_arr[np.isfinite(st_arr)]
    if len(finite_st) >= 2:
        lra = float(np.percentile(finite_st, 95) - np.percentile(finite_st, 10))
    else:
        lra = 0.0

    return LUFSResult(
        integrated=float(integrated_lufs),
        short_term=st_arr,
        momentary=m_arr,
        loudness_range=lra,
        max_short_term=float(np.max(st_arr)),
        max_momentary=float(np.max(m_arr)),
    )


# ---------------------------------------------------------------------------
# True-peak detection (4x oversampling)
# ---------------------------------------------------------------------------

def true_peak_dbtp(audio: np.ndarray, sr: float,
                   oversample: int = 4) -> float:
    """True peak in dBTP via polyphase upsampling."""
    n = len(audio)
    up = np.zeros(n * oversample)
    up[::oversample] = audio
    # Low-pass FIR for interpolation
    nyq = sr * oversample / 2.0
    cutoff = sr / 2.0
    numtaps = min(64 * oversample + 1, len(up))
    if numtaps % 2 == 0:
        numtaps += 1
    h = sig.firwin(numtaps, cutoff / nyq)
    up_filtered = sig.fftconvolve(up, h, mode="same") * oversample
    peak = float(np.max(np.abs(up_filtered)))
    return 20.0 * np.log10(peak + 1e-30)


def inter_sample_peaks(audio: np.ndarray, sr: float,
                       oversample: int = 4) -> np.ndarray:
    """Detect inter-sample peaks that exceed 0 dBFS."""
    n = len(audio)
    up = np.zeros(n * oversample)
    up[::oversample] = audio
    nyq = sr * oversample / 2.0
    cutoff = sr / 2.0
    numtaps = min(64 * oversample + 1, len(up))
    if numtaps % 2 == 0:
        numtaps += 1
    h = sig.firwin(numtaps, cutoff / nyq)
    up_filtered = sig.fftconvolve(up, h, mode="same") * oversample

    peaks_mask = np.abs(up_filtered) > 1.0
    if not np.any(peaks_mask):
        return np.array([], dtype=int)

    peak_indices = np.where(peaks_mask)[0]
    return peak_indices // oversample  # map back to original sample positions


# ---------------------------------------------------------------------------
# Dynamic range metrics
# ---------------------------------------------------------------------------

@dataclass
class DynamicRangeResult:
    crest_factor_db: float
    peak_to_rms_db: float
    dynamic_range_db: float   # DR metric (unweighted)
    psr_db: float             # peak-to-short-term-loudness ratio


def dynamic_range(audio: np.ndarray, sr: float,
                  block_ms: float = 3000.0) -> DynamicRangeResult:
    """Compute dynamic range metrics."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    crest_db = 20.0 * np.log10(peak / (rms + 1e-30) + 1e-30)
    peak_rms_db = 20.0 * np.log10(peak + 1e-30) - 20.0 * np.log10(rms + 1e-30)

    # Block-wise RMS for DR metric
    block_samples = int(block_ms / 1000.0 * sr)
    n_blocks = max(1, len(audio) // block_samples)
    block_rms = np.array([
        np.sqrt(np.mean(audio[i * block_samples:(i + 1) * block_samples] ** 2))
        for i in range(n_blocks)
    ])
    block_rms_db = 20.0 * np.log10(block_rms + 1e-30)

    # DR: 2nd-highest peak dB - RMS of top 20% loudest blocks
    sorted_rms = np.sort(block_rms_db)[::-1]
    top_20 = sorted_rms[:max(1, len(sorted_rms) // 5)]
    dr = float(20.0 * np.log10(peak + 1e-30) - np.mean(top_20))

    # PSR (peak to short-term loudness ratio)
    psr = peak_rms_db  # simplified; full version uses LUFS short-term

    return DynamicRangeResult(
        crest_factor_db=float(crest_db),
        peak_to_rms_db=float(peak_rms_db),
        dynamic_range_db=float(dr),
        psr_db=float(psr),
    )


# ---------------------------------------------------------------------------
# Stereo width analysis
# ---------------------------------------------------------------------------

@dataclass
class StereoWidthResult:
    correlation: float          # -1 (out of phase) to +1 (mono)
    mid_side_ratio_db: float    # 0 = pure mid, large = wide
    stereo_width: float         # 0..1 normalized width
    balance: float              # -1 (left) to +1 (right)
    phase_coherent: bool


def stereo_width(left: np.ndarray, right: np.ndarray) -> StereoWidthResult:
    """Analyze stereo width, balance, and phase coherence."""
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    mid_energy = float(np.mean(mid ** 2))
    side_energy = float(np.mean(side ** 2))

    ms_ratio_db = 10.0 * np.log10((side_energy + 1e-30) / (mid_energy + 1e-30))

    # Pearson correlation
    corr_num = float(np.sum(left * right))
    corr_den = np.sqrt(float(np.sum(left ** 2)) * float(np.sum(right ** 2))) + 1e-30
    correlation = corr_num / corr_den

    # Width: 0 = mono, 1 = fully decorrelated
    width = float(np.clip(1.0 - abs(correlation), 0.0, 1.0))

    # Balance
    l_energy = float(np.mean(left ** 2))
    r_energy = float(np.mean(right ** 2))
    total = l_energy + r_energy + 1e-30
    balance = (r_energy - l_energy) / total

    return StereoWidthResult(
        correlation=float(correlation),
        mid_side_ratio_db=float(ms_ratio_db),
        stereo_width=width,
        balance=float(balance),
        phase_coherent=correlation > 0.0,
    )


# ---------------------------------------------------------------------------
# Spectral balance scoring
# ---------------------------------------------------------------------------

@dataclass
class SpectralBalanceResult:
    band_energies: np.ndarray    # energy per band (5 bands)
    band_fractions: np.ndarray   # normalized to sum=1
    genre_deviation: float       # L2 distance from genre target
    band_labels: list[str]


def spectral_balance(audio: np.ndarray, sr: float,
                     genre: str = "pop",
                     n_fft: int = 4096) -> SpectralBalanceResult:
    """Score spectral balance against genre-specific targets."""
    bands = BAND_EDGES_HZ
    labels = ["sub", "low", "mid", "high_mid", "air"]

    # Compute magnitude spectrum
    S = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    band_energies = np.zeros(5)
    for i in range(5):
        lo, hi = bands[i], bands[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        band_energies[i] = float(np.sum(S[mask] ** 2))

    total = band_energies.sum() + 1e-30
    fractions = band_energies / total

    target = GENRE_SPECTRAL_TARGETS.get(genre, GENRE_SPECTRAL_TARGETS["pop"])
    deviation = float(np.linalg.norm(fractions - target))

    return SpectralBalanceResult(
        band_energies=band_energies,
        band_fractions=fractions,
        genre_deviation=deviation,
        band_labels=labels,
    )


# ---------------------------------------------------------------------------
# DC offset and clipping detection
# ---------------------------------------------------------------------------

def dc_offset(audio: np.ndarray) -> float:
    """Measure DC offset as fraction of full scale."""
    return float(np.mean(audio))


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at or above clipping threshold."""
    n_clipped = int(np.sum(np.abs(audio) >= threshold))
    return n_clipped / max(1, len(audio))


# ---------------------------------------------------------------------------
# Composite mastering report
# ---------------------------------------------------------------------------

@dataclass
class MasteringReport:
    lufs: LUFSResult
    true_peak_dbtp: float
    dynamic_range: DynamicRangeResult
    stereo: StereoWidthResult | None
    spectral: SpectralBalanceResult
    dc_offset: float
    clipping_ratio: float
    passes_broadcast: bool    # within broadcast loudness spec


def mastering_report(audio: np.ndarray, sr: float,
                     n_channels: int = 1,
                     genre: str = "pop",
                     standard: LoudnessStandard = LoudnessStandard.EBU_R128,
                     ) -> MasteringReport:
    """Generate a comprehensive mastering quality report."""
    if n_channels == 1:
        mono = audio
        lufs = measure_lufs(mono, sr, 1)
        sw = None
    else:
        reshaped = audio.reshape(-1, n_channels)
        mono = reshaped.mean(axis=1)
        lufs = measure_lufs(audio, sr, n_channels)
        sw = stereo_width(reshaped[:, 0], reshaped[:, 1]) if n_channels >= 2 else None

    tp = true_peak_dbtp(mono, sr)
    dr = dynamic_range(mono, sr)
    sb = spectral_balance(mono, sr, genre=genre)
    dc = dc_offset(mono)
    clip = clipping_ratio(mono)

    target = TARGET_LUFS[standard]
    tolerance = 1.0  # ±1 LU
    passes = (abs(lufs.integrated - target) <= tolerance) and (tp <= -1.0)

    return MasteringReport(
        lufs=lufs,
        true_peak_dbtp=tp,
        dynamic_range=dr,
        stereo=sw,
        spectral=sb,
        dc_offset=dc,
        clipping_ratio=clip,
        passes_broadcast=passes,
    )
