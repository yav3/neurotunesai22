"""Psychoacoustic loudness: equal-loudness contours (ISO 226), specific loudness."""
from __future__ import annotations
import numpy as np


ISO_226_FREQS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500,
])

THRESHOLD_SPL = np.array([
    78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
    14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.2,
    -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
])


class EqualLoudness:
    def threshold_at_freq(self, freq: float) -> float:
        if freq < ISO_226_FREQS[0] or freq > ISO_226_FREQS[-1]:
            return 80.0
        return float(np.interp(freq, ISO_226_FREQS, THRESHOLD_SPL))

    def phon_to_sone(self, phon: float) -> float:
        if phon < 40:
            return max(2.0 ** ((phon - 40.0) / 10.0), 0.0)
        return 2.0 ** ((phon - 40.0) / 10.0)

    def sone_to_phon(self, sone: float) -> float:
        if sone <= 0:
            return 0.0
        return 40.0 + 10.0 * np.log2(max(sone, 1e-12))

    def loudness_level(self, spl: float, freq: float) -> float:
        threshold = self.threshold_at_freq(freq)
        if spl <= threshold:
            return 0.0
        ref_threshold = self.threshold_at_freq(1000.0)
        correction = threshold - ref_threshold
        return max(spl - correction, 0.0)


class SpecificLoudness:
    def __init__(self, n_bands: int = 24):
        self.n_bands = n_bands
        self._eq = EqualLoudness()

    def compute(self, excitation: np.ndarray, center_freqs: np.ndarray) -> np.ndarray:
        n = min(len(excitation), len(center_freqs), self.n_bands)
        specific = np.zeros(n)
        for i in range(n):
            spl = 10.0 * np.log10(max(excitation[i], 1e-12))
            phon = self._eq.loudness_level(spl, float(center_freqs[i]))
            specific[i] = self._eq.phon_to_sone(phon)
        return specific

    def total_loudness(self, excitation: np.ndarray, center_freqs: np.ndarray) -> float:
        specific = self.compute(excitation, center_freqs)
        return float(np.sum(specific))

    def loudness_in_sone(self, excitation: np.ndarray, center_freqs: np.ndarray) -> float:
        return self.total_loudness(excitation, center_freqs)
