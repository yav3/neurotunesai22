"""Psychoacoustic models: Bark/ERB/mel scales, equal-loudness contours, masking, loudness."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def hz_to_mel(f: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def hz_to_bark(f: float | np.ndarray) -> float | np.ndarray:
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def bark_to_hz(z: float | np.ndarray) -> float | np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1960.0 * (z + 0.53) / (26.28 - z)


def hz_to_erb(f: float | np.ndarray) -> float | np.ndarray:
    f = np.asarray(f, dtype=float)
    return 21.4 * np.log10(4.37e-3 * f + 1.0)


def erb_to_hz(e: float | np.ndarray) -> float | np.ndarray:
    e = np.asarray(e, dtype=float)
    return (10.0 ** (e / 21.4) - 1.0) / 4.37e-3


def critical_bandwidth(fc: float) -> float:
    return 25.0 + 75.0 * (1.0 + 1.4e-6 * fc**2) ** 0.69


@dataclass
class EqualLoudnessPoint:
    frequency: float
    spl_db: float
    phon: float


def equal_loudness_contour(phon: float, freqs: np.ndarray | None = None) -> np.ndarray:
    if freqs is None:
        freqs = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                          315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                          3150, 4000, 5000, 6300, 8000, 10000, 12500], dtype=float)
    af = 4.47e-3 * (10.0 ** (0.025 * phon) - 1.15) + (0.4 * 10.0 ** (
        ((_threshold_quiet(freqs) + 0.005 * phon) / 40.0) - 1.0)) ** 0.1
    spl = (10.0 / np.log(2.0)) * np.log(af) - 0.005 * phon + 94.0
    spl = np.clip(spl, _threshold_quiet(freqs), 130.0)
    idx_1k = np.argmin(np.abs(freqs - 1000.0))
    offset = phon - spl[idx_1k]
    spl = spl + offset
    return np.column_stack([freqs, spl])


def _threshold_quiet(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    return 3.64 * (f / 1000.0) ** (-0.8) - 6.5 * np.exp(
        -0.6 * (f / 1000.0 - 3.3) ** 2
    ) + 1e-3 * (f / 1000.0) ** 4


def a_weighting(f: float | np.ndarray) -> float | np.ndarray:
    f = np.asarray(f, dtype=float)
    f2 = f ** 2
    num = 12194.0**2 * f2**2
    den = ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) *
           (f2 + 12194.0**2))
    ra = num / np.where(den == 0, 1e-30, den)
    return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def c_weighting(f: float | np.ndarray) -> float | np.ndarray:
    f = np.asarray(f, dtype=float)
    f2 = f ** 2
    num = 12194.0**2 * f2
    den = (f2 + 20.6**2) * (f2 + 12194.0**2)
    rc = num / np.where(den == 0, 1e-30, den)
    return 20.0 * np.log10(np.maximum(rc, 1e-30)) + 0.062


@dataclass
class MaskingResult:
    masker_freq: float
    masker_level: float
    masked_threshold: np.ndarray


def simultaneous_masking(masker_freq: float, masker_level_db: float,
                         probe_freqs: np.ndarray) -> MaskingResult:
    z_masker = hz_to_bark(masker_freq)
    z_probes = hz_to_bark(probe_freqs)
    dz = z_probes - z_masker
    slope_lower = 27.0
    slope_upper = max(5.0, -24.0 + 0.23 * masker_freq / 1000.0 + 0.2 * masker_level_db)
    spread = np.where(dz < 0, slope_lower * np.abs(dz), slope_upper * dz)
    threshold = masker_level_db - spread
    threshold = np.clip(threshold, _threshold_quiet(probe_freqs), masker_level_db)
    return MaskingResult(masker_freq, masker_level_db, threshold)


def temporal_masking(time_ms: float | np.ndarray, masker_level_db: float,
                     pre: bool = False) -> float | np.ndarray:
    t = np.asarray(time_ms, dtype=float)
    if pre:
        decay = masker_level_db * np.exp(-t / 15.0)
    else:
        decay = masker_level_db * np.exp(-t / 50.0)
    return np.maximum(decay, 0.0)


def spl_to_sone(spl_db: float, freq: float = 1000.0) -> float:
    if spl_db <= 0:
        return 0.0
    phon = spl_db + float(a_weighting(freq))
    if phon < 40:
        return 2.0 ** ((phon - 40.0) / 10.0) if phon > 0 else 0.0
    return 2.0 ** ((phon - 40.0) / 10.0)


def sone_to_phon(sone: float) -> float:
    if sone <= 0:
        return 0.0
    return 40.0 + 10.0 * np.log2(sone)


def phon_to_sone(phon: float) -> float:
    if phon < 40:
        return 2.0 ** ((phon - 40.0) / 10.0) if phon > 0 else 0.0
    return 2.0 ** ((phon - 40.0) / 10.0)
