"""Equal-loudness contours (ISO 226:2003 approximation) and phon/sone conversion.

Implements Fletcher-Munson / Robinson-Dadson curves via the ISO 226
analytical model — pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


ISO226_FREQS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500,
])

_AF = np.array([
    0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349,
    0.330, 0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250,
    0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254,
    0.271, 0.301,
])

_LU = np.array([
    -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2,
    -4.5, -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7,
    -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1,
])

_TF = np.array([
    78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
    14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7,
    -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
])


def equal_loudness_contour(phon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_hz, spl_db) for a given loudness level in phons.

    ISO 226:2003 analytical approximation.
    """
    phon = np.clip(phon, 0, 90)
    Af = 4.47e-3 * (10 ** (0.025 * phon) - 1.15) + (0.4 * 10 ** ((_TF + _LU) / 10 - 9)) ** _AF
    Lp = (10.0 / _AF) * np.log10(Af) - _LU + 94.0
    return ISO226_FREQS.copy(), Lp


def phon_to_sone(phon: float) -> float:
    if phon < 40:
        return max(0.0, (phon / 40.0) ** 2.642 * 1.0)
    return 2.0 ** ((phon - 40.0) / 10.0)


def sone_to_phon(sone: float) -> float:
    if sone <= 0:
        return 0.0
    if sone < 1.0:
        return 40.0 * sone ** (1.0 / 2.642)
    return 40.0 + 10.0 * np.log2(sone)


def spl_to_phon(spl_db: float, freq_hz: float) -> float:
    """Approximate conversion from SPL to phons at a given frequency."""
    freqs, _ = equal_loudness_contour(40)
    idx = np.argmin(np.abs(freqs - freq_hz))
    best_phon = 40.0
    best_diff = 1e6
    for phon in np.arange(0, 91, 1):
        _, spl_curve = equal_loudness_contour(phon)
        diff = abs(spl_curve[idx] - spl_db)
        if diff < best_diff:
            best_diff = diff
            best_phon = phon
    return float(best_phon)


@dataclass
class LoudnessSpectrum:
    frequencies: np.ndarray
    spl_db: np.ndarray

    def perceived_loudness(self) -> float:
        total_sone = 0.0
        for i in range(len(self.frequencies)):
            phon = spl_to_phon(self.spl_db[i], self.frequencies[i])
            total_sone += phon_to_sone(phon)
        phon_total = sone_to_phon(total_sone)
        return float(phon_total)


def a_weighting(freq: np.ndarray) -> np.ndarray:
    f2 = freq ** 2
    ra = (12194 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2) *
        np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
        (f2 + 12194 ** 2)
    )
    a_db = 20 * np.log10(ra + 1e-20) + 2.0
    return a_db


def c_weighting(freq: np.ndarray) -> np.ndarray:
    f2 = freq ** 2
    rc = (12194 ** 2 * f2) / (
        (f2 + 20.6 ** 2) * (f2 + 12194 ** 2)
    )
    c_db = 20 * np.log10(rc + 1e-20) + 0.06
    return c_db


def apply_weighting(spectrum_db: np.ndarray, freqs: np.ndarray,
                    curve: str = "A") -> np.ndarray:
    if curve.upper() == "A":
        w = a_weighting(freqs)
    elif curve.upper() == "C":
        w = c_weighting(freqs)
    else:
        return spectrum_db.copy()
    return spectrum_db + w


def loudness_compensation(spectrum_db: np.ndarray, freqs: np.ndarray,
                          playback_spl: float = 70.0,
                          reference_spl: float = 85.0) -> np.ndarray:
    """Compensate for low-level playback using equal-loudness contours."""
    delta_phon = playback_spl - reference_spl
    _, ref_curve = equal_loudness_contour(reference_spl)
    _, play_curve = equal_loudness_contour(max(0, playback_spl))
    compensation = np.interp(freqs, ISO226_FREQS, ref_curve - play_curve)
    return spectrum_db + compensation


def frequency_specific_loudness(audio: np.ndarray, sr: int,
                                n_fft: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Compute specific loudness across critical bands."""
    w = np.hanning(min(len(audio), n_fft))
    x = audio[:n_fft] * w if len(audio) >= n_fft else np.pad(audio, (0, n_fft - len(audio))) * np.hanning(n_fft)
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    spl = 20 * np.log10(X + 1e-12) + 94
    bark_edges = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                           1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                           4400, 5300, 6400, 7700, 9500, 12000, 15500])
    n_bands = len(bark_edges) - 1
    band_loudness = np.zeros(n_bands)
    band_centers = np.zeros(n_bands)
    for b in range(n_bands):
        mask = (freqs >= bark_edges[b]) & (freqs < bark_edges[b + 1])
        if np.any(mask):
            band_spl = 10 * np.log10(np.sum(10 ** (spl[mask] / 10)) + 1e-20)
            center = (bark_edges[b] + bark_edges[b + 1]) / 2
            band_centers[b] = center
            band_loudness[b] = phon_to_sone(spl_to_phon(float(band_spl), center))
    return band_centers, band_loudness
