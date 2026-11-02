"""Spectral analysis: spectral descriptors, spectral shape features, formant analysis.

Implements spectral centroid, spread, skewness, kurtosis, flux, rolloff,
flatness, crest, decrease, slope, and formant tracking for music analysis.
"""
from __future__ import annotations


import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralDescriptors:
    centroid_hz: float
    spread_hz: float
    skewness: float
    kurtosis: float
    flatness: float
    crest: float
    rolloff_hz: float
    slope: float
    decrease: float
    flux: float


def spectral_centroid(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    """First moment of the spectrum: sum(f * |X(f)|) / sum(|X(f)|)."""
    total = np.sum(magnitudes)
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * magnitudes) / total)


def spectral_spread(magnitudes: np.ndarray, freqs: np.ndarray, centroid: float) -> float:
    """Second central moment (standard deviation) of the spectrum."""
    total = np.sum(magnitudes)
    if total < 1e-12:
        return 0.0
    variance = np.sum(magnitudes * (freqs - centroid) ** 2) / total
    return float(np.sqrt(max(variance, 0.0)))


def spectral_skewness(magnitudes: np.ndarray, freqs: np.ndarray, centroid: float, spread: float) -> float:
    """Third standardized central moment."""
    if spread < 1e-12:
        return 0.0
    total = np.sum(magnitudes)
    if total < 1e-12:
        return 0.0
    return float(np.sum(magnitudes * ((freqs - centroid) / spread) ** 3) / total)


def spectral_kurtosis(magnitudes: np.ndarray, freqs: np.ndarray, centroid: float, spread: float) -> float:
    """Fourth standardized central moment minus 3 (excess kurtosis)."""
    if spread < 1e-12:
        return 0.0
    total = np.sum(magnitudes)
    if total < 1e-12:
        return 0.0
    return float(np.sum(magnitudes * ((freqs - centroid) / spread) ** 4) / total - 3.0)


def spectral_flatness(magnitudes: np.ndarray) -> float:
    """Wiener entropy: geometric_mean / arithmetic_mean of |X(f)|.

    1.0 = white noise, 0.0 = pure tone.
    """
    mag = np.asarray(magnitudes, dtype=float)
    if len(mag) == 0 or np.all(mag <= 0):
        return 0.0
    log_mean = np.mean(np.log(mag + 1e-20))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(mag)
    if arith_mean < 1e-20:
        return 0.0
    return float(geo_mean / arith_mean)


def spectral_crest(magnitudes: np.ndarray) -> float:
    """Peak / mean of the magnitude spectrum."""
    mean_val = np.mean(magnitudes)
    if mean_val < 1e-20:
        return 0.0
    return float(np.max(magnitudes) / mean_val)


def spectral_rolloff(magnitudes: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
    """Frequency below which `threshold` fraction of spectral energy is contained."""
    total_energy = np.sum(magnitudes ** 2)
    if total_energy < 1e-20:
        return 0.0
    cumulative = np.cumsum(magnitudes ** 2)
    idx = np.searchsorted(cumulative, threshold * total_energy)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def spectral_slope(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    """Linear regression slope of magnitude vs frequency."""
    n = len(freqs)
    if n < 2:
        return 0.0
    f_mean = np.mean(freqs)
    m_mean = np.mean(magnitudes)
    num = np.sum((freqs - f_mean) * (magnitudes - m_mean))
    den = np.sum((freqs - f_mean) ** 2)
    if den < 1e-20:
        return 0.0
    return float(num / den)


def spectral_decrease(magnitudes: np.ndarray) -> float:
    """Rate of decrease of the spectral envelope (ITU-R BS.1387)."""
    if len(magnitudes) < 2:
        return 0.0
    k = np.arange(1, len(magnitudes))
    num = np.sum((magnitudes[1:] - magnitudes[0]) / k)
    den = np.sum(magnitudes[1:])
    if den < 1e-20:
        return 0.0
    return float(num / den)


def spectral_flux(current: np.ndarray, previous: np.ndarray) -> float:
    """L2 norm of frame-to-frame spectral difference (half-wave rectified)."""
    diff = current - previous
    diff_hw = np.maximum(diff, 0.0)
    return float(np.sqrt(np.sum(diff_hw ** 2)))


def compute_spectral_descriptors(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    previous_magnitudes: np.ndarray | None = None,
) -> SpectralDescriptors:
    """Compute all spectral shape descriptors for one frame."""
    cent = spectral_centroid(magnitudes, freqs)
    spr = spectral_spread(magnitudes, freqs, cent)
    skew = spectral_skewness(magnitudes, freqs, cent, spr)
    kurt = spectral_kurtosis(magnitudes, freqs, cent, spr)
    flat = spectral_flatness(magnitudes)
    crest = spectral_crest(magnitudes)
    rolloff = spectral_rolloff(magnitudes, freqs)
    slope = spectral_slope(magnitudes, freqs)
    decrease = spectral_decrease(magnitudes)
    flux = 0.0
    if previous_magnitudes is not None:
        flux = spectral_flux(magnitudes, previous_magnitudes)
    return SpectralDescriptors(
        centroid_hz=cent, spread_hz=spr, skewness=skew, kurtosis=kurt,
        flatness=flat, crest=crest, rolloff_hz=rolloff, slope=slope,
        decrease=decrease, flux=flux,
    )


@dataclass(frozen=True)
class Formant:
    frequency_hz: float
    bandwidth_hz: float
    amplitude_db: float


def estimate_formants(
    spectrum_db: np.ndarray,
    freqs: np.ndarray,
    n_formants: int = 4,
    min_freq: float = 90.0,
    max_freq: float = 5500.0,
) -> list[Formant]:
    """Simple peak-picking formant estimator from a smoothed spectrum."""
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    spec = spectrum_db[mask]
    f = freqs[mask]
    if len(spec) < 5:
        return []
    kernel_size = min(15, len(spec) // 2)
    if kernel_size < 3:
        return []
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(spec, kernel, mode="same")
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append((f[i], smoothed[i]))
    peaks.sort(key=lambda x: x[1], reverse=True)
    formants: list[Formant] = []
    for freq_hz, amp_db in peaks[:n_formants]:
        bw = freq_hz * 0.1
        formants.append(Formant(frequency_hz=freq_hz, bandwidth_hz=bw, amplitude_db=amp_db))
    formants.sort(key=lambda x: x.frequency_hz)
    return formants


def harmonic_to_noise_ratio(magnitudes: np.ndarray, freqs: np.ndarray, f0: float) -> float:
    """Estimate HNR given a magnitude spectrum and fundamental frequency.

    Harmonic bins are those within 5% of integer multiples of f0.
    """
    if f0 <= 0 or len(magnitudes) == 0:
        return 0.0
    harmonic_energy = 0.0
    total_energy = float(np.sum(magnitudes ** 2))
    if total_energy < 1e-20:
        return 0.0
    for h in range(1, int(freqs[-1] / f0) + 1):
        target = h * f0
        tol = 0.05 * target
        mask = np.abs(freqs - target) <= tol
        harmonic_energy += float(np.sum(magnitudes[mask] ** 2))
    noise_energy = total_energy - harmonic_energy
    if noise_energy < 1e-20:
        return 60.0
    return float(10.0 * math.log10(harmonic_energy / noise_energy))
