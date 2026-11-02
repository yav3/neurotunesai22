"""Audio filter design using the Audio EQ Cookbook (Robert Bristow-Johnson)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    ALLPASS = "allpass"
    PEAK = "peak"
    LOWSHELF = "lowshelf"
    HIGHSHELF = "highshelf"


@dataclass
class BiquadCoeffs:
    b0: float
    b1: float
    b2: float
    a0: float
    a1: float
    a2: float

    def normalized(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([self.b0 / self.a0, self.b1 / self.a0, self.b2 / self.a0]),
            np.array([1.0, self.a1 / self.a0, self.a2 / self.a0]),
        )

    def frequency_response(self, freqs: np.ndarray, sr: float) -> np.ndarray:
        b, a = self.normalized()
        w = 2 * np.pi * freqs / sr
        z = np.exp(1j * w)
        num = b[0] + b[1] * z ** -1 + b[2] * z ** -2
        den = a[0] + a[1] * z ** -1 + a[2] * z ** -2
        return num / np.where(np.abs(den) < 1e-30, 1e-30, den)


def biquad_design(
    ftype: FilterType, fc: float, sr: float,
    q: float = 0.7071, gain_db: float = 0.0,
) -> BiquadCoeffs:
    w0 = 2 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2 * q) if q > 0 else sin_w0

    A = 10 ** (gain_db / 40) if gain_db != 0 else 1.0

    if ftype == FilterType.LOWPASS:
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif ftype == FilterType.HIGHPASS:
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif ftype == FilterType.BANDPASS:
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif ftype == FilterType.BANDSTOP:
        b0 = 1.0
        b1 = -2 * cos_w0
        b2 = 1.0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif ftype == FilterType.ALLPASS:
        b0 = 1 - alpha
        b1 = -2 * cos_w0
        b2 = 1 + alpha
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif ftype == FilterType.PEAK:
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    elif ftype == FilterType.LOWSHELF:
        sq = 2 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + sq)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - sq)
        a0 = (A + 1) + (A - 1) * cos_w0 + sq
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - sq
    elif ftype == FilterType.HIGHSHELF:
        sq = 2 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + sq)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - sq)
        a0 = (A + 1) - (A - 1) * cos_w0 + sq
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - sq
    else:
        b0, b1, b2, a0, a1, a2 = 1, 0, 0, 1, 0, 0

    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def apply_biquad(signal: np.ndarray, coeffs: BiquadCoeffs) -> np.ndarray:
    b, a = coeffs.normalized()
    n = len(signal)
    out = np.zeros(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        x0 = signal[i]
        y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        out[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return out


def cascade_response(
    coeffs_list: list[BiquadCoeffs], freqs: np.ndarray, sr: float,
) -> np.ndarray:
    h = np.ones(len(freqs), dtype=complex)
    for c in coeffs_list:
        h *= c.frequency_response(freqs, sr)
    return h


def parametric_eq(bands: list[dict], sr: float) -> list[BiquadCoeffs]:
    return [
        biquad_design(
            FilterType(band.get("type", "peak")),
            band["fc"], sr,
            band.get("q", 1.0),
            band.get("gain_db", 0.0),
        )
        for band in bands
    ]


def a_weighting(freqs: np.ndarray) -> np.ndarray:
    f2 = freqs ** 2
    num = 12194 ** 2 * f2 ** 2
    den = ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
           * (f2 + 12194 ** 2))
    den = np.where(den == 0, 1e-30, den)
    ra = num / den
    ref = (12194 ** 4 * 1000 ** 4) / (
        (1000 ** 2 + 20.6 ** 2) * math.sqrt((1000 ** 2 + 107.7 ** 2) * (1000 ** 2 + 737.9 ** 2))
        * (1000 ** 2 + 12194 ** 2)
    )
    ra /= ref
    return 20 * np.log10(np.maximum(ra, 1e-20))


def c_weighting(freqs: np.ndarray) -> np.ndarray:
    f2 = freqs ** 2
    num = 12194 ** 2 * f2
    den = (f2 + 20.6 ** 2) * (f2 + 12194 ** 2)
    den = np.where(den == 0, 1e-30, den)
    rc = num / den
    ref = 12194 ** 2 * 1000 ** 2 / ((1000 ** 2 + 20.6 ** 2) * (1000 ** 2 + 12194 ** 2))
    rc /= ref
    return 20 * np.log10(np.maximum(rc, 1e-20))
