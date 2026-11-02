"""Filter design: resonant filters, equalizers, filter envelopes,
and spectral shaping for synthesis.

Implements standard IIR biquad filters (Bristow-Johnson cookbook),
state-variable filters, Moog ladder approximation, and
parametric EQ. Pure numpy/scipy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    ALLPASS = "allpass"
    PEAK = "peak"
    LOW_SHELF = "low_shelf"
    HIGH_SHELF = "high_shelf"


@dataclass
class BiquadCoeffs:
    b0: float
    b1: float
    b2: float
    a0: float
    a1: float
    a2: float


@dataclass
class FilterState:
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0


@dataclass
class SVFState:
    ic1eq: float = 0.0
    ic2eq: float = 0.0


def biquad_lowpass(fc: float, sr: float, q: float = 0.707) -> BiquadCoeffs:
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def biquad_highpass(fc: float, sr: float, q: float = 0.707) -> BiquadCoeffs:
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def biquad_bandpass(fc: float, sr: float, q: float = 1.0) -> BiquadCoeffs:
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def biquad_notch(fc: float, sr: float, q: float = 1.0) -> BiquadCoeffs:
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1.0
    b1 = -2.0 * cos_w0
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def biquad_peak(fc: float, sr: float, gain_db: float,
                q: float = 1.0) -> BiquadCoeffs:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def biquad_low_shelf(fc: float, sr: float, gain_db: float) -> BiquadCoeffs:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 1.0 - 1.0) + 2.0)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def biquad_high_shelf(fc: float, sr: float, gain_db: float) -> BiquadCoeffs:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 1.0 - 1.0) + 2.0)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    return BiquadCoeffs(b0, b1, b2, a0, a1, a2)


def apply_biquad(signal: np.ndarray, coeffs: BiquadCoeffs) -> np.ndarray:
    n = len(signal)
    output = np.zeros(n)
    s = FilterState()
    inv_a0 = 1.0 / coeffs.a0
    for i in range(n):
        x = float(signal[i])
        y = (coeffs.b0 * x + coeffs.b1 * s.x1 + coeffs.b2 * s.x2
             - coeffs.a1 * s.y1 - coeffs.a2 * s.y2) * inv_a0
        s.x2 = s.x1
        s.x1 = x
        s.y2 = s.y1
        s.y1 = y
        output[i] = y
    return output


def cascade_biquads(signal: np.ndarray,
                    coeffs_list: List[BiquadCoeffs]) -> np.ndarray:
    out = signal.copy()
    for c in coeffs_list:
        out = apply_biquad(out, c)
    return out


def svf_tick(sample: float, g: float, k: float,
             state: SVFState) -> Tuple[float, float, float]:
    v3 = sample - state.ic2eq
    v1 = state.ic1eq * 1.0 + g * (v3 - state.ic1eq) / (1.0 + g * (g + k))
    v2 = state.ic2eq + g * v1
    state.ic1eq = 2.0 * v1 - state.ic1eq
    state.ic2eq = 2.0 * v2 - state.ic2eq
    lp = v2
    bp = v1
    hp = sample - k * v1 - v2
    return lp, bp, hp


def svf_filter(signal: np.ndarray, fc: float, sr: float,
               q: float = 0.707,
               filter_type: FilterType = FilterType.LOWPASS) -> np.ndarray:
    g = np.tan(np.pi * fc / sr)
    k = 1.0 / q
    state = SVFState()
    n = len(signal)
    output = np.zeros(n)
    for i in range(n):
        lp, bp, hp = svf_tick(float(signal[i]), g, k, state)
        if filter_type == FilterType.LOWPASS:
            output[i] = lp
        elif filter_type == FilterType.HIGHPASS:
            output[i] = hp
        elif filter_type == FilterType.BANDPASS:
            output[i] = bp
        elif filter_type == FilterType.NOTCH:
            output[i] = lp + hp
        else:
            output[i] = lp
    return output


def moog_ladder(signal: np.ndarray, fc: float, sr: float,
                resonance: float = 0.5) -> np.ndarray:
    n = len(signal)
    output = np.zeros(n)
    s = np.zeros(4)
    g = 1.0 - np.exp(-2.0 * np.pi * fc / sr)
    k = 4.0 * resonance
    for i in range(n):
        x = float(signal[i]) - k * s[3]
        x = np.tanh(x)
        for j in range(4):
            y = s[j] + g * (np.tanh(x) - np.tanh(s[j]))
            s[j] = y
            x = y
        output[i] = s[3]
    return output


def parametric_eq(signal: np.ndarray, sr: float,
                  bands: List[Tuple[float, float, float]]) -> np.ndarray:
    coeffs = []
    for fc, gain_db, q in bands:
        coeffs.append(biquad_peak(fc, sr, gain_db, q))
    return cascade_biquads(signal, coeffs)


def frequency_response(coeffs: BiquadCoeffs, sr: float,
                       n_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    freqs = np.linspace(0, sr / 2, n_points)
    w = 2.0 * np.pi * freqs / sr
    z = np.exp(1j * w)
    num = coeffs.b0 + coeffs.b1 * z ** -1 + coeffs.b2 * z ** -2
    den = coeffs.a0 + coeffs.a1 * z ** -1 + coeffs.a2 * z ** -2
    H = num / den
    return freqs, np.abs(H)


def filter_envelope(signal: np.ndarray, sr: float,
                    fc_start: float, fc_end: float,
                    q: float = 2.0) -> np.ndarray:
    n = len(signal)
    output = np.zeros(n)
    fc_curve = np.linspace(fc_start, fc_end, n)
    state = SVFState()
    for i in range(n):
        g = np.tan(np.pi * fc_curve[i] / sr)
        k = 1.0 / q
        lp, _, _ = svf_tick(float(signal[i]), g, k, state)
        output[i] = lp
    return output
