"""Roughness and sensory dissonance: Plomp-Levelt, Sethares, Hutchinson-Knopoff."""
from __future__ import annotations
import numpy as np


def plomp_levelt_curve(freq_diff: float, f_avg: float) -> float:
    """Plomp & Levelt (1965) dissonance curve for a partial pair.
    Returns dissonance ∈ [0, 1]. Maximum near 0.25 critical bandwidth."""
    if f_avg < 1.0:
        return 0.0
    s = 0.24 / (0.0207 * f_avg + 18.96)
    x = abs(freq_diff) * s
    return 4.0 * x * np.exp(1.0 - 4.0 * x) if x > 0 else 0.0


def critical_bandwidth(freq: float) -> float:
    """Greenwood (1961) critical bandwidth in Hz."""
    return 25.0 + 75.0 * (1.0 + 1.4e-6 * freq**2) ** 0.69


def sethares_dissonance(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Sethares (1993) total sensory dissonance for a spectrum.
    freqs: partial frequencies (Hz), amps: amplitudes (linear)."""
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    n = len(freqs)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            f_lo, f_hi = min(freqs[i], freqs[j]), max(freqs[i], freqs[j])
            a_prod = amps[i] * amps[j]
            s = 0.24 / (0.0207 * f_lo + 18.96)
            fdiff = (f_hi - f_lo) * s
            d = a_prod * (np.exp(-3.5 * fdiff) - np.exp(-5.75 * fdiff))
            total += max(0.0, d)
    return total


def hutchinson_knopoff(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Hutchinson & Knopoff (1978) aggregate roughness via CBW model."""
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    n = len(freqs)
    if n < 2:
        return 0.0
    roughness = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            f_avg = (freqs[i] + freqs[j]) / 2.0
            cbw = critical_bandwidth(f_avg)
            fdiff = abs(freqs[j] - freqs[i])
            ratio = fdiff / cbw if cbw > 0 else 0.0
            g = ratio * np.exp(1.0 - ratio) if ratio > 0 else 0.0
            roughness += amps[i] * amps[j] * g
    return roughness


class DissonanceCurve:
    """Dissonance curve for a given timbre (set of partials)."""

    def __init__(self, partials: list[float], amps: list[float] | None = None):
        self.partials = np.array(partials, dtype=float)
        if amps is None:
            self.amps = 1.0 / np.arange(1, len(partials) + 1, dtype=float)
        else:
            self.amps = np.array(amps, dtype=float)

    def evaluate(self, ratio: float, base_freq: float = 261.63) -> float:
        """Dissonance at a given frequency ratio above base_freq."""
        freqs1 = self.partials * base_freq
        freqs2 = self.partials * base_freq * ratio
        all_f = np.concatenate([freqs1, freqs2])
        all_a = np.concatenate([self.amps, self.amps])
        return sethares_dissonance(all_f, all_a)

    def curve(self, ratios: np.ndarray, base_freq: float = 261.63) -> np.ndarray:
        """Dissonance values for an array of frequency ratios."""
        return np.array([self.evaluate(r, base_freq) for r in ratios])

    def local_minima(self, n_points: int = 1000, octave: float = 2.0,
                     base_freq: float = 261.63) -> list[tuple[float, float]]:
        """Find consonant intervals (local dissonance minima) up to octave."""
        ratios = np.linspace(1.0, octave, n_points)
        vals = self.curve(ratios, base_freq)
        minima = []
        for i in range(1, len(vals) - 1):
            if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
                minima.append((float(ratios[i]), float(vals[i])))
        return minima


def interval_dissonance_12tet(interval: int, n_harmonics: int = 6) -> float:
    """Sensory dissonance for a 12-TET interval (0-12 semitones).
    Uses harmonic series timbre."""
    ratio = 2.0 ** (interval / 12.0)
    partials = np.arange(1, n_harmonics + 1, dtype=float)
    amps = 1.0 / partials
    dc = DissonanceCurve(partials.tolist(), amps.tolist())
    return dc.evaluate(ratio)


def consonance_ranking_12tet(n_harmonics: int = 6) -> list[tuple[int, float]]:
    """Rank 12-TET intervals by sensory consonance (ascending dissonance)."""
    pairs = [(i, interval_dissonance_12tet(i, n_harmonics)) for i in range(13)]
    pairs.sort(key=lambda x: x[1])
    return pairs


class RoughnessAnalyzer:
    """Frame-by-frame roughness analysis for spectral time series."""

    def __init__(self, method: str = "sethares"):
        self._fn = sethares_dissonance if method == "sethares" else hutchinson_knopoff

    def frame_roughness(self, freqs: np.ndarray, amps: np.ndarray) -> float:
        return self._fn(freqs, amps)

    def trajectory(self, spectra: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Roughness trajectory over a sequence of (freqs, amps) frames."""
        return np.array([self._fn(f, a) for f, a in spectra])

    def mean_roughness(self, spectra: list[tuple[np.ndarray, np.ndarray]]) -> float:
        traj = self.trajectory(spectra)
        return float(traj.mean()) if len(traj) > 0 else 0.0

    def roughness_range(self, spectra: list[tuple[np.ndarray, np.ndarray]]) -> float:
        traj = self.trajectory(spectra)
        return float(traj.max() - traj.min()) if len(traj) > 0 else 0.0
