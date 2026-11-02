"""Mix analysis: diagnostic metering for mixing and pre-mastering.

Spectral balance, LUFS loudness (ITU-R BS.1770-4), stereo imaging,
frequency masking detection, and composite mix scoring. Frequency band
conventions follow Izhaki (2017) 'Mixing Audio'. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence


# ---------------------------------------------------------------------------
# Izhaki (2017) standard frequency bands
# ---------------------------------------------------------------------------
STANDARD_BANDS: List[Tuple[str, float, float]] = [
    ("sub",       20,    60),
    ("low",       60,   250),
    ("low_mid",  250,   500),
    ("mid",      500,  2000),
    ("upper_mid", 2000, 4000),
    ("presence",  4000, 6000),
    ("brilliance", 6000, 20000),
]


@dataclass
class MixBand:
    """Energy profile for a single frequency band."""
    name: str
    low_freq: float
    high_freq: float
    energy: float
    peak_db: float
    crest_factor_db: float


@dataclass
class MaskingRegion:
    """Frequency region where two sources compete."""
    low_freq: float
    high_freq: float
    overlap_energy_db: float
    ratio_db: float


@dataclass
class MixReport:
    """Composite mix diagnostic."""
    bands: List[MixBand]
    crest_factor_db: float
    headroom_db: float
    dc_offset: float
    stereo_width: float
    stereo_balance_db: float
    correlation: float
    score: float


# ---------------------------------------------------------------------------
# Core metering
# ---------------------------------------------------------------------------

def frequency_band_energy(
    magnitude_spectrum: np.ndarray,
    frequencies: np.ndarray,
    low: float,
    high: float,
) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if not mask.any():
        return 0.0
    return float(np.sum(magnitude_spectrum[mask] ** 2))


def spectral_balance(
    magnitude_spectrum: np.ndarray,
    frequencies: np.ndarray,
    bands: Optional[List[Tuple[str, float, float]]] = None,
) -> List[MixBand]:
    if bands is None:
        bands = STANDARD_BANDS
    result: List[MixBand] = []
    for name, lo, hi in bands:
        mask = (frequencies >= lo) & (frequencies < hi)
        if not mask.any():
            result.append(MixBand(name, lo, hi, 0.0, -np.inf, 0.0))
            continue
        seg = magnitude_spectrum[mask]
        energy = float(np.sum(seg ** 2))
        pk = float(np.max(np.abs(seg)))
        pk_db = float(20.0 * np.log10(pk + 1e-12))
        rms = float(np.sqrt(np.mean(seg ** 2)))
        cf = float(20.0 * np.log10((pk + 1e-12) / (rms + 1e-12)))
        result.append(MixBand(name, lo, hi, energy, pk_db, cf))
    return result


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def crest_factor(signal: np.ndarray) -> float:
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 1e-12:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def headroom(signal: np.ndarray) -> float:
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-12:
        return np.inf
    return float(-20.0 * np.log10(peak))


def dc_offset_detector(signal: np.ndarray) -> float:
    return float(np.mean(signal))


# ---------------------------------------------------------------------------
# ITU-R BS.1770-4 LUFS loudness
# ---------------------------------------------------------------------------

def _k_weight_filter(signal: np.ndarray, sr: float) -> np.ndarray:
    """Two-stage K-weighting: high-shelf + high-pass (simplified FIR)."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec = np.fft.rfft(signal)
    shelf_gain = np.ones_like(freqs)
    shelf_gain[freqs > 0] = 1.0 + 1.5 * (1.0 - np.exp(-freqs[freqs > 0] / 3000.0))
    hp = np.ones_like(freqs)
    hp[freqs < 80] = 0.0
    fade = (freqs >= 38) & (freqs < 80)
    hp[fade] = (freqs[fade] - 38) / 42.0
    spec *= shelf_gain * hp
    return np.fft.irfft(spec, n=n)


def _block_rms(signal: np.ndarray, sr: float, block_ms: float) -> np.ndarray:
    block_len = int(sr * block_ms / 1000.0)
    if block_len < 1 or len(signal) < block_len:
        rms = float(np.sqrt(np.mean(signal ** 2)))
        return np.array([rms]) if rms > 0 else np.array([1e-12])
    hop = block_len // 4
    n_blocks = max(1, (len(signal) - block_len) // hop + 1)
    rms_vals = np.empty(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        blk = signal[start : start + block_len]
        rms_vals[i] = np.sqrt(np.mean(blk ** 2))
    return rms_vals


def lufs_momentary(signal: np.ndarray, sr: float = 44100.0) -> float:
    weighted = _k_weight_filter(signal, sr)
    rms_vals = _block_rms(weighted, sr, 400.0)
    max_rms = float(np.max(rms_vals))
    if max_rms < 1e-12:
        return -70.0
    return float(-0.691 + 10.0 * np.log10(max_rms ** 2))


def lufs_integrated(signal: np.ndarray, sr: float = 44100.0) -> float:
    weighted = _k_weight_filter(signal, sr)
    rms_vals = _block_rms(weighted, sr, 400.0)
    power = rms_vals ** 2
    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    gated = power[power > abs_gate]
    if len(gated) == 0:
        return -70.0
    rel_threshold = np.mean(gated) * 10.0 ** (-10.0 / 10.0)
    final = gated[gated > rel_threshold]
    if len(final) == 0:
        return -70.0
    return float(-0.691 + 10.0 * np.log10(np.mean(final)))


def loudness_range(signal: np.ndarray, sr: float = 44100.0) -> float:
    weighted = _k_weight_filter(signal, sr)
    rms_vals = _block_rms(weighted, sr, 3000.0)
    power = rms_vals ** 2
    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    gated = power[power > abs_gate]
    if len(gated) < 2:
        return 0.0
    rel_threshold = np.mean(gated) * 10.0 ** (-20.0 / 10.0)
    final = gated[gated > rel_threshold]
    if len(final) < 2:
        return 0.0
    loudness_db = 10.0 * np.log10(final + 1e-20)
    p10 = float(np.percentile(loudness_db, 10))
    p95 = float(np.percentile(loudness_db, 95))
    return max(p95 - p10, 0.0)


# ---------------------------------------------------------------------------
# Stereo imaging
# ---------------------------------------------------------------------------

def mix_stereo_width(left: np.ndarray, right: np.ndarray) -> float:
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    mid_e = float(np.sum(mid ** 2))
    side_e = float(np.sum(side ** 2))
    if mid_e < 1e-12:
        return 1.0 if side_e > 1e-12 else 0.0
    return float(side_e / (mid_e + side_e))


def mix_stereo_balance(left: np.ndarray, right: np.ndarray) -> float:
    l_e = float(np.sum(left ** 2))
    r_e = float(np.sum(right ** 2))
    if l_e < 1e-12 and r_e < 1e-12:
        return 0.0
    return float(10.0 * np.log10((l_e + 1e-12) / (r_e + 1e-12)))


def mix_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0:
        return 0.0
    num = float(np.sum(left * right))
    denom = np.sqrt(float(np.sum(left ** 2)) * float(np.sum(right ** 2)))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Masking detection
# ---------------------------------------------------------------------------

def masking_detector(
    mag_a: np.ndarray,
    mag_b: np.ndarray,
    frequencies: np.ndarray,
    bands: Optional[List[Tuple[str, float, float]]] = None,
    threshold_db: float = 3.0,
) -> List[MaskingRegion]:
    if bands is None:
        bands = STANDARD_BANDS
    regions: List[MaskingRegion] = []
    for _, lo, hi in bands:
        mask = (frequencies >= lo) & (frequencies < hi)
        if not mask.any():
            continue
        ea = float(np.sum(mag_a[mask] ** 2))
        eb = float(np.sum(mag_b[mask] ** 2))
        if ea < 1e-12 or eb < 1e-12:
            continue
        ea_db = 10.0 * np.log10(ea + 1e-12)
        eb_db = 10.0 * np.log10(eb + 1e-12)
        ratio = abs(ea_db - eb_db)
        if ratio < threshold_db:
            overlap = 10.0 * np.log10(min(ea, eb) + 1e-12)
            regions.append(MaskingRegion(lo, hi, overlap, ratio))
    return regions


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def mix_score(
    signal: np.ndarray,
    sr: float = 44100.0,
    left: Optional[np.ndarray] = None,
    right: Optional[np.ndarray] = None,
    target_lufs: float = -14.0,
    target_crest_db: float = 12.0,
) -> MixReport:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mag = np.abs(np.fft.rfft(signal))
    bands = spectral_balance(mag, freqs)

    cf = crest_factor(signal)
    hr = headroom(signal)
    dc = dc_offset_detector(signal)

    if left is not None and right is not None:
        sw = mix_stereo_width(left, right)
        sb = mix_stereo_balance(left, right)
        corr = mix_correlation(left, right)
    else:
        sw, sb, corr = 0.0, 0.0, 1.0

    energies = np.array([b.energy for b in bands])
    total_e = energies.sum()
    if total_e > 0:
        proportions = energies / total_e
        ideal = np.array([0.08, 0.20, 0.18, 0.22, 0.15, 0.10, 0.07])
        if len(proportions) == len(ideal):
            balance_score = 1.0 - float(np.sqrt(np.mean((proportions - ideal) ** 2))) * 5.0
        else:
            balance_score = 0.5
    else:
        balance_score = 0.0

    crest_score = 1.0 - min(abs(cf - target_crest_db) / target_crest_db, 1.0)
    headroom_score = 1.0 if hr >= 1.0 else max(hr, 0.0)
    stereo_score = 1.0 - abs(sb) / 6.0 if abs(sb) < 6.0 else 0.0

    composite = float(np.clip(
        0.30 * max(balance_score, 0.0)
        + 0.25 * max(crest_score, 0.0)
        + 0.20 * max(headroom_score, 0.0)
        + 0.15 * max(stereo_score, 0.0)
        + 0.10 * max(corr, 0.0),
        0.0, 1.0,
    ))

    return MixReport(
        bands=bands,
        crest_factor_db=cf,
        headroom_db=hr,
        dc_offset=dc,
        stereo_width=sw,
        stereo_balance_db=sb,
        correlation=corr,
        score=composite,
    )
