"""Parametric EQ design: biquad filters, frequency response, auto-EQ matching."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class EQBandType(Enum):
    LOW_SHELF = "low_shelf"
    HIGH_SHELF = "high_shelf"
    PEAK = "peak"
    LOW_PASS = "low_pass"
    HIGH_PASS = "high_pass"
    BAND_PASS = "band_pass"
    NOTCH = "notch"


@dataclass(frozen=True)
class EQBand:
    band_type: EQBandType
    frequency_hz: float
    gain_db: float = 0.0
    q_factor: float = 0.707


def biquad_coefficients(band: EQBand, sr: float = 44100.0) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2.0 * np.pi * band.frequency_hz / sr
    alpha = np.sin(w0) / (2.0 * band.q_factor)
    A = 10.0 ** (band.gain_db / 40.0)

    if band.band_type == EQBandType.PEAK:
        b = np.array([1.0 + alpha * A, -2.0 * np.cos(w0), 1.0 - alpha * A])
        a = np.array([1.0 + alpha / A, -2.0 * np.cos(w0), 1.0 - alpha / A])
    elif band.band_type == EQBandType.LOW_SHELF:
        sq = 2.0 * np.sqrt(A) * alpha
        b = np.array([
            A * ((A + 1) - (A - 1) * np.cos(w0) + sq),
            2.0 * A * ((A - 1) - (A + 1) * np.cos(w0)),
            A * ((A + 1) - (A - 1) * np.cos(w0) - sq),
        ])
        a = np.array([
            (A + 1) + (A - 1) * np.cos(w0) + sq,
            -2.0 * ((A - 1) + (A + 1) * np.cos(w0)),
            (A + 1) + (A - 1) * np.cos(w0) - sq,
        ])
    elif band.band_type == EQBandType.HIGH_SHELF:
        sq = 2.0 * np.sqrt(A) * alpha
        b = np.array([
            A * ((A + 1) + (A - 1) * np.cos(w0) + sq),
            -2.0 * A * ((A - 1) + (A + 1) * np.cos(w0)),
            A * ((A + 1) + (A - 1) * np.cos(w0) - sq),
        ])
        a = np.array([
            (A + 1) - (A - 1) * np.cos(w0) + sq,
            2.0 * ((A - 1) - (A + 1) * np.cos(w0)),
            (A + 1) - (A - 1) * np.cos(w0) - sq,
        ])
    elif band.band_type == EQBandType.LOW_PASS:
        b = np.array([(1.0 - np.cos(w0)) / 2.0, 1.0 - np.cos(w0), (1.0 - np.cos(w0)) / 2.0])
        a = np.array([1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha])
    elif band.band_type == EQBandType.HIGH_PASS:
        b = np.array([(1.0 + np.cos(w0)) / 2.0, -(1.0 + np.cos(w0)), (1.0 + np.cos(w0)) / 2.0])
        a = np.array([1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha])
    elif band.band_type == EQBandType.BAND_PASS:
        b = np.array([alpha, 0.0, -alpha])
        a = np.array([1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha])
    elif band.band_type == EQBandType.NOTCH:
        b = np.array([1.0, -2.0 * np.cos(w0), 1.0])
        a = np.array([1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha])
    else:
        b = np.array([1.0, 0.0, 0.0])
        a = np.array([1.0, 0.0, 0.0])

    b = b / a[0]
    a = a / a[0]
    return b, a


def apply_biquad(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    n = len(signal)
    out = np.zeros(n)
    x1, x2, y1, y2 = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        x0 = signal[i]
        y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        out[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return out


def frequency_response(band: EQBand, freqs: np.ndarray,
                       sr: float = 44100.0) -> np.ndarray:
    b, a = biquad_coefficients(band, sr)
    w = 2.0 * np.pi * freqs / sr
    z = np.exp(1j * w)
    h = (b[0] + b[1] * z ** -1 + b[2] * z ** -2) / (a[0] + a[1] * z ** -1 + a[2] * z ** -2)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-10))


def spectral_tilt(spectrum_db: np.ndarray, freqs: np.ndarray) -> float:
    mask = freqs > 0
    if np.sum(mask) < 2:
        return 0.0
    log_f = np.log2(freqs[mask])
    s = spectrum_db[mask]
    coeffs = np.polyfit(log_f, s, 1)
    return float(coeffs[0])


class ParametricEQ:
    def __init__(self, bands: list[EQBand] | None = None, sr: float = 44100.0):
        self.bands = bands or []
        self.sr = sr

    def add_band(self, band: EQBand) -> None:
        self.bands.append(band)

    def process(self, signal: np.ndarray) -> np.ndarray:
        out = signal.copy()
        for band in self.bands:
            b, a = biquad_coefficients(band, self.sr)
            out = apply_biquad(out, b, a)
        return out

    def response(self, freqs: np.ndarray) -> np.ndarray:
        total = np.zeros(len(freqs))
        for band in self.bands:
            total += frequency_response(band, freqs, self.sr)
        return total


def auto_eq_match(source_spectrum: np.ndarray, target_spectrum: np.ndarray,
                  freqs: np.ndarray, n_bands: int = 5,
                  sr: float = 44100.0) -> list[EQBand]:
    diff = target_spectrum - source_spectrum
    bands: list[EQBand] = []
    residual = diff.copy()
    for _ in range(n_bands):
        idx = int(np.argmax(np.abs(residual)))
        gain = float(residual[idx])
        freq = float(freqs[idx])
        if abs(gain) < 0.5 or freq <= 0:
            break
        q = 2.0
        band = EQBand(EQBandType.PEAK, freq, gain, q)
        resp = frequency_response(band, freqs, sr)
        residual -= resp
        bands.append(band)
    return bands
