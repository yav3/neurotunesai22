"""Spectral processing: parametric EQ, spectral whitening, spectral gating, morphing.

Low-level spectral manipulation primitives for audio processing: parametric
EQ bands (peaking, shelf, notch), spectral whitening/flattening, spectral
noise gating, spectral centroid/spread tracking, and cross-synthesis.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class FilterType(str, Enum):
    PEAKING = "peaking"
    LOW_SHELF = "low_shelf"
    HIGH_SHELF = "high_shelf"
    NOTCH = "notch"
    BANDPASS = "bandpass"


@dataclass
class EQBand:
    filter_type: FilterType
    frequency: float   # Hz
    gain_db: float     # dB (for peaking/shelf)
    q: float = 1.0     # quality factor


def biquad_coefficients(band: EQBand, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute biquad filter coefficients (b, a) for an EQ band.

    Peaking EQ from Audio EQ Cookbook (Robert Bristow-Johnson).
    """
    w0 = 2.0 * np.pi * band.frequency / sr
    alpha = np.sin(w0) / (2.0 * band.q)
    A = 10.0 ** (band.gain_db / 40.0)

    if band.filter_type == FilterType.PEAKING:
        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A
    elif band.filter_type == FilterType.LOW_SHELF:
        sq = 2.0 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + sq)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - sq)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + sq
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - sq
    elif band.filter_type == FilterType.HIGH_SHELF:
        sq = 2.0 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + sq)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - sq)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + sq
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - sq
    elif band.filter_type == FilterType.NOTCH:
        b0 = 1.0
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha
    else:  # BANDPASS
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_biquad(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply biquad filter via direct form II transposed."""
    n = len(signal)
    out = np.zeros(n)
    z1, z2 = 0.0, 0.0
    for i in range(n):
        x = signal[i]
        y = b[0] * x + z1
        z1 = b[1] * x - a[1] * y + z2
        z2 = b[2] * x - a[2] * y
        out[i] = y
    return out


def parametric_eq(signal: np.ndarray, sr: float,
                  bands: List[EQBand]) -> np.ndarray:
    """Apply a chain of parametric EQ bands."""
    out = signal.copy()
    for band in bands:
        b, a = biquad_coefficients(band, sr)
        out = apply_biquad(out, b, a)
    return out


def spectral_whitening(magnitude: np.ndarray,
                       smoothing: int = 5) -> np.ndarray:
    """Flatten spectral envelope: divide by smoothed magnitude."""
    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        envelope = np.convolve(magnitude, kernel, mode='same')
    else:
        envelope = magnitude.copy()
    envelope = np.maximum(envelope, 1e-10)
    return magnitude / envelope


def spectral_gate(magnitude: np.ndarray, phase: np.ndarray,
                  threshold_db: float = -40.0) -> Tuple[np.ndarray, np.ndarray]:
    """Zero out spectral bins below threshold."""
    mag_db = 20.0 * np.log10(magnitude + 1e-10)
    mask = mag_db >= threshold_db
    return magnitude * mask, phase


def spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    """Spectral centroid in Hz."""
    total = magnitude.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * magnitude) / total)


def spectral_spread(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    """Standard deviation of frequency distribution."""
    total = magnitude.sum()
    if total < 1e-12:
        return 0.0
    centroid = np.sum(freqs * magnitude) / total
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / total))


def spectral_flux(mag_current: np.ndarray, mag_previous: np.ndarray) -> float:
    """Half-wave rectified spectral flux."""
    diff = mag_current - mag_previous
    return float(np.sum(np.maximum(diff, 0.0) ** 2))


def spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray,
                     rolloff_percent: float = 0.85) -> float:
    """Frequency below which rolloff_percent of energy lies."""
    total = np.sum(magnitude ** 2)
    if total < 1e-12:
        return 0.0
    cumsum = np.cumsum(magnitude ** 2)
    idx = np.searchsorted(cumsum, rolloff_percent * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def cross_synthesis(mag_carrier: np.ndarray, phase_carrier: np.ndarray,
                    mag_modulator: np.ndarray,
                    mix: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-synthesis: carrier phase + modulator magnitude."""
    blended_mag = (1.0 - mix) * mag_carrier + mix * mag_modulator
    return blended_mag, phase_carrier


def spectral_freeze(magnitude: np.ndarray, phase: np.ndarray,
                    n_frames: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Repeat a spectral frame to create a freeze effect."""
    mag_out = np.tile(magnitude, (n_frames, 1)) if magnitude.ndim == 1 else magnitude
    if phase.ndim == 1:
        phase_increment = phase.copy()
        phases = [phase.copy()]
        for _ in range(n_frames - 1):
            phase_increment = phase_increment + phase
            phases.append(phase_increment % (2 * np.pi))
        phase_out = np.array(phases)
    else:
        phase_out = phase
    return mag_out, phase_out


def harmonic_percussive_ratio(harmonic_energy: float,
                              percussive_energy: float) -> float:
    """Ratio of harmonic to percussive energy."""
    total = harmonic_energy + percussive_energy
    if total < 1e-12:
        return 0.5
    return harmonic_energy / total
