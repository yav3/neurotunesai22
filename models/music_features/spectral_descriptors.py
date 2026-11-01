"""Low-level spectral feature extraction.

Implements standard audio spectral descriptors from MPEG-7, Peeters (2004),
and librosa conventions. All functions operate on magnitude spectra
(numpy arrays), not raw audio — the caller handles windowing/FFT.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct


# ── Frequency helpers ────────────────────────────────────────────────────

def _freq_bins(n_fft: int, sr: float = 22050.0) -> np.ndarray:
    return np.linspace(0, sr / 2, n_fft // 2 + 1)


def _hz_to_bark(hz: np.ndarray) -> np.ndarray:
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ── Spectral shape descriptors ──────────────────────────────────────────

def spectral_centroid(mag: np.ndarray, sr: float = 22050.0) -> float:
    """First spectral moment — brightness indicator (Hz)."""
    freqs = _freq_bins(2 * (len(mag) - 1), sr)[:len(mag)]
    total = np.sum(mag)
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * mag) / total)


def spectral_bandwidth(mag: np.ndarray, sr: float = 22050.0, p: int = 2) -> float:
    """p-th order spectral bandwidth around the centroid (Hz)."""
    freqs = _freq_bins(2 * (len(mag) - 1), sr)[:len(mag)]
    total = np.sum(mag)
    if total < 1e-12:
        return 0.0
    cent = np.sum(freqs * mag) / total
    return float((np.sum(mag * np.abs(freqs - cent) ** p) / total) ** (1.0 / p))


def spectral_rolloff(mag: np.ndarray, sr: float = 22050.0, threshold: float = 0.85) -> float:
    """Frequency below which *threshold* fraction of spectral energy lies."""
    cumsum = np.cumsum(mag ** 2)
    target = threshold * cumsum[-1] if cumsum[-1] > 0 else 0.0
    idx = np.searchsorted(cumsum, target)
    freqs = _freq_bins(2 * (len(mag) - 1), sr)[:len(mag)]
    return float(freqs[min(idx, len(freqs) - 1)])


def spectral_flatness(mag: np.ndarray) -> float:
    """Wiener entropy — 1.0 = white noise, 0.0 = pure tone."""
    mag_pos = mag[mag > 0]
    if len(mag_pos) < 2:
        return 0.0
    log_mean = np.mean(np.log(mag_pos + 1e-20))
    return float(np.exp(log_mean) / (np.mean(mag_pos) + 1e-20))


def spectral_slope(mag: np.ndarray, sr: float = 22050.0) -> float:
    """Linear regression slope of magnitude vs frequency."""
    freqs = _freq_bins(2 * (len(mag) - 1), sr)[:len(mag)]
    if len(freqs) < 2:
        return 0.0
    fm = np.mean(freqs)
    mm = np.mean(mag)
    num = np.sum((freqs - fm) * (mag - mm))
    den = np.sum((freqs - fm) ** 2)
    if abs(den) < 1e-20:
        return 0.0
    return float(num / den)


def spectral_decrease(mag: np.ndarray) -> float:
    """Spectral decrease — rate of decrease of spectral amplitude (MPEG-7)."""
    n = len(mag)
    if n < 2:
        return 0.0
    k = np.arange(1, n)
    total = np.sum(mag[1:])
    if total < 1e-12:
        return 0.0
    return float(np.sum((mag[1:] - mag[0]) / k) / total)


def spectral_crest_factor(mag: np.ndarray) -> float:
    """Ratio of spectral peak to spectral mean — tonality indicator."""
    mean_val = np.mean(mag)
    if mean_val < 1e-12:
        return 0.0
    return float(np.max(mag) / mean_val)


# ── Band-based features ─────────────────────────────────────────────────

def bark_band_energies(mag: np.ndarray, sr: float = 22050.0, n_bands: int = 24) -> np.ndarray:
    """Energy in *n_bands* Bark-scale critical bands."""
    freqs = _freq_bins(2 * (len(mag) - 1), sr)[:len(mag)]
    barks = _hz_to_bark(freqs)
    max_bark = barks[-1]
    edges = np.linspace(0, max_bark, n_bands + 1)
    energies = np.zeros(n_bands)
    power = mag ** 2
    for i in range(n_bands):
        mask = (barks >= edges[i]) & (barks < edges[i + 1])
        if np.any(mask):
            energies[i] = np.sum(power[mask])
    return energies


def mel_filterbank(n_mels: int, n_fft: int, sr: float = 22050.0) -> np.ndarray:
    """Triangular mel-scale filterbank matrix (n_mels, n_fft//2+1)."""
    n_freq = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freq)
    mel_lo = _hz_to_mel(np.array([0.0]))[0]
    mel_hi = _hz_to_mel(np.array([sr / 2.0]))[0]
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freq))
    for m in range(n_mels):
        lo, mid, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        for k in range(n_freq):
            if lo <= freqs[k] < mid and (mid - lo) > 0:
                fb[m, k] = (freqs[k] - lo) / (mid - lo)
            elif mid <= freqs[k] <= hi and (hi - mid) > 0:
                fb[m, k] = (hi - freqs[k]) / (hi - mid)
    return fb


def mfcc(mag: np.ndarray, sr: float = 22050.0, n_mfcc: int = 13, n_mels: int = 40) -> np.ndarray:
    """Mel-frequency cepstral coefficients via DCT-II of log-mel spectrum."""
    n_fft = 2 * (len(mag) - 1)
    fb = mel_filterbank(n_mels, n_fft, sr)
    mel_spec = fb @ (mag ** 2)
    log_mel = np.log(mel_spec + 1e-10)
    return dct(log_mel, type=2, norm='ortho')[:n_mfcc]


def spectral_contrast(mag: np.ndarray, sr: float = 22050.0, n_bands: int = 6,
                       quantile: float = 0.02) -> np.ndarray:
    """Peak-valley contrast in octave-spaced frequency bands."""
    n_freq = len(mag)
    freqs = _freq_bins(2 * (n_freq - 1), sr)[:n_freq]
    base = 200.0
    contrasts = np.zeros(n_bands)
    for b in range(n_bands):
        lo = base * (2 ** b)
        hi = base * (2 ** (b + 1))
        mask = (freqs >= lo) & (freqs < hi)
        band = mag[mask]
        if len(band) < 2:
            continue
        band_sorted = np.sort(band)
        k = max(1, int(quantile * len(band_sorted)))
        valley = np.mean(band_sorted[:k])
        peak = np.mean(band_sorted[-k:])
        contrasts[b] = np.log1p(peak) - np.log1p(valley)
    return contrasts


# ── Aggregate feature vector ────────────────────────────────────────────

def spectral_feature_vector(mag: np.ndarray, sr: float = 22050.0,
                             n_mfcc: int = 13) -> np.ndarray:
    """Concatenated spectral descriptor vector for a single frame."""
    feats = [
        spectral_centroid(mag, sr),
        spectral_bandwidth(mag, sr),
        spectral_rolloff(mag, sr),
        spectral_flatness(mag),
        spectral_slope(mag, sr),
        spectral_decrease(mag),
        spectral_crest_factor(mag),
    ]
    feats.extend(mfcc(mag, sr, n_mfcc).tolist())
    return np.array(feats, dtype=float)
