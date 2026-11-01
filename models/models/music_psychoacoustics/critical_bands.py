"""Critical band models: Bark scale, ERB scale, auditory filters."""
from __future__ import annotations
import numpy as np


def hz_to_bark(f: float | np.ndarray) -> float | np.ndarray:
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def bark_to_hz(z: float | np.ndarray) -> float | np.ndarray:
    z = np.asarray(z, dtype=float)
    result = np.zeros_like(z)
    for i in range(5):
        result = 600.0 * np.sinh(z / 6.0)
        z_check = hz_to_bark(result)
        result = result * z / np.maximum(z_check, 1e-12)
    return float(result) if result.ndim == 0 else result


def hz_to_erb(f: float | np.ndarray) -> float | np.ndarray:
    f = np.asarray(f, dtype=float)
    return 21.4 * np.log10(0.00437 * f + 1.0)


def erb_to_hz(e: float | np.ndarray) -> float | np.ndarray:
    e = np.asarray(e, dtype=float)
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def erb_bandwidth(f: float | np.ndarray) -> float | np.ndarray:
    f = np.asarray(f, dtype=float)
    return 24.7 * (0.00437 * f + 1.0)


class CriticalBandAnalyzer:
    BARK_CENTERS = [
        50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170,
        1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800,
        5800, 7000, 8500, 10500, 13500,
    ]

    def band_index(self, freq: float) -> int:
        z = float(hz_to_bark(freq))
        return max(0, min(int(z), 23))

    def band_overlap(self, f1: float, f2: float) -> float:
        z1 = float(hz_to_bark(f1))
        z2 = float(hz_to_bark(f2))
        dist = abs(z2 - z1)
        if dist >= 1.0:
            return 0.0
        return 1.0 - dist

    def spectral_spread(self, freqs: list[float]) -> float:
        if len(freqs) < 2:
            return 0.0
        barks = [float(hz_to_bark(f)) for f in freqs]
        return max(barks) - min(barks)


class GammatoneFilterbank:
    def __init__(self, n_filters: int = 64, f_low: float = 50.0, f_high: float = 8000.0):
        self.n_filters = n_filters
        erb_low = float(hz_to_erb(f_low))
        erb_high = float(hz_to_erb(f_high))
        erb_points = np.linspace(erb_low, erb_high, n_filters)
        self.center_freqs = erb_to_hz(erb_points)
        self.bandwidths = erb_bandwidth(self.center_freqs)

    def filter_response(self, f: np.ndarray, channel: int) -> np.ndarray:
        fc = self.center_freqs[channel]
        bw = self.bandwidths[channel]
        b = 1.019 * 2 * np.pi * bw
        g = ((2 * np.pi * bw) ** 4) / (
            (2 * np.pi * (f - fc)) ** 2 + b**2
        ) ** 2
        return g / (g.max() + 1e-12)

    def excitation_pattern(self, spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        excitation = np.zeros(self.n_filters)
        for i in range(self.n_filters):
            weights = self.filter_response(freqs, i)
            excitation[i] = float(np.sum(weights * spectrum**2))
        return excitation
