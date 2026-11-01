"""Timbral descriptors: brightness, warmth, roughness, and related spectral qualities."""
from __future__ import annotations

import numpy as np


def spectral_centroid_from_spectrum(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    """Spectral centroid (brightness) from a single magnitude spectrum."""
    total = np.sum(magnitude)
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * magnitude) / total)


def spectral_spread(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    """Spectral spread (bandwidth around centroid)."""
    centroid = spectral_centroid_from_spectrum(magnitude, freqs)
    total = np.sum(magnitude)
    if total < 1e-12:
        return 0.0
    return float(np.sqrt(np.sum(magnitude * (freqs - centroid) ** 2) / total))


def spectral_skewness(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    """Third central moment of the spectral distribution."""
    centroid = spectral_centroid_from_spectrum(magnitude, freqs)
    spread = spectral_spread(magnitude, freqs)
    if spread < 1e-12:
        return 0.0
    total = np.sum(magnitude)
    return float(np.sum(magnitude * ((freqs - centroid) / spread) ** 3) / total)


def spectral_kurtosis(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    """Fourth central moment (peakedness) of the spectral distribution."""
    centroid = spectral_centroid_from_spectrum(magnitude, freqs)
    spread = spectral_spread(magnitude, freqs)
    if spread < 1e-12:
        return 0.0
    total = np.sum(magnitude)
    return float(np.sum(magnitude * ((freqs - centroid) / spread) ** 4) / total)


def spectral_flatness(magnitude: np.ndarray) -> float:
    """Spectral flatness (Wiener entropy): geometric / arithmetic mean.

    1.0 = white noise, near 0 = tonal.
    """
    mag = np.maximum(magnitude, 1e-20)
    geo = np.exp(np.mean(np.log(mag)))
    arith = np.mean(mag)
    if arith < 1e-20:
        return 0.0
    return float(geo / arith)


def spectral_crest(magnitude: np.ndarray) -> float:
    """Spectral crest factor: peak / mean."""
    mean_val = np.mean(magnitude)
    if mean_val < 1e-20:
        return 0.0
    return float(np.max(magnitude) / mean_val)


def brightness(magnitude: np.ndarray, freqs: np.ndarray, cutoff: float = 1500.0) -> float:
    """Brightness: energy ratio above cutoff frequency."""
    total = np.sum(magnitude ** 2)
    if total < 1e-20:
        return 0.0
    above = np.sum(magnitude[freqs >= cutoff] ** 2)
    return float(above / total)


def warmth(magnitude: np.ndarray, freqs: np.ndarray, low: float = 100.0, high: float = 500.0) -> float:
    """Warmth: energy ratio in the low-mid band."""
    total = np.sum(magnitude ** 2)
    if total < 1e-20:
        return 0.0
    band = np.sum(magnitude[(freqs >= low) & (freqs <= high)] ** 2)
    return float(band / total)


def roughness_sethares(partials_freq: np.ndarray, partials_amp: np.ndarray) -> float:
    """Sensory dissonance (roughness) via Sethares' model.

    Pairwise interaction of partials using the Plomp-Levelt curve.
    """
    n = len(partials_freq)
    if n < 2:
        return 0.0
    b1, b2 = 3.5, 5.75
    s1, s2 = 0.0207, 18.96
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            f_min = min(partials_freq[i], partials_freq[j])
            if f_min < 1.0:
                continue
            f_max = max(partials_freq[i], partials_freq[j])
            s = 0.24 / (s1 * f_min + s2)
            diff = f_max - f_min
            amp_prod = partials_amp[i] * partials_amp[j]
            d = amp_prod * (np.exp(-b1 * s * diff) - np.exp(-b2 * s * diff))
            total += d
    return float(total)


def inharmonicity(partials_freq: np.ndarray, f0: float) -> float:
    """Inharmonicity: mean deviation of partials from integer multiples of f0."""
    if f0 <= 0 or len(partials_freq) == 0:
        return 0.0
    deviations = []
    for f in partials_freq:
        harmonic_number = round(f / f0)
        if harmonic_number < 1:
            harmonic_number = 1
        ideal = harmonic_number * f0
        deviations.append(abs(f - ideal) / ideal)
    return float(np.mean(deviations))


def odd_even_ratio(partials_amp: np.ndarray) -> float:
    """Ratio of odd-harmonic to even-harmonic energy.

    >1 = clarinet-like (odd dominant); <1 = oboe-like (even present).
    """
    if len(partials_amp) < 2:
        return 1.0
    odd_energy = np.sum(partials_amp[0::2] ** 2)
    even_energy = np.sum(partials_amp[1::2] ** 2)
    if even_energy < 1e-20:
        return float('inf') if odd_energy > 0 else 1.0
    return float(odd_energy / even_energy)
