"""Wavelet-based music analysis.

Continuous and discrete wavelet transforms for multi-resolution time-frequency
analysis of audio.  Morlet, Mexican-hat, and Paul mother wavelets.  Scaleogram
feature extraction (spectral centroid per scale, energy distribution, ridge
detection).  Pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve


class WaveletType(Enum):
    MORLET = auto()
    MEXICAN_HAT = auto()
    PAUL = auto()


@dataclass
class ScaleogramResult:
    coefficients: np.ndarray   # (n_scales, n_samples)
    scales: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray          # |coefficients|^2


@dataclass
class WaveletFeatures:
    spectral_centroid: np.ndarray  # per time frame
    energy_per_scale: np.ndarray   # (n_scales,)
    ridges: List[np.ndarray]       # list of ridge lines (scale indices over time)
    total_energy: float


def morlet_wavelet(n: int, scale: float, omega0: float = 6.0) -> np.ndarray:
    """Complex Morlet wavelet ψ(t) = π^(-1/4) · e^(iω₀t) · e^(-t²/2)."""
    t = np.arange(n, dtype=float) - n / 2
    t = t / scale
    norm = np.pi ** (-0.25) / np.sqrt(scale)
    return norm * np.exp(1j * omega0 * t) * np.exp(-t ** 2 / 2)


def mexican_hat_wavelet(n: int, scale: float) -> np.ndarray:
    """Mexican hat (Ricker) wavelet: (2/√3) · π^(-1/4) · (1-t²) · e^(-t²/2)."""
    t = np.arange(n, dtype=float) - n / 2
    t = t / scale
    norm = (2.0 / np.sqrt(3.0)) * np.pi ** (-0.25) / np.sqrt(scale)
    return norm * (1.0 - t ** 2) * np.exp(-t ** 2 / 2)


def paul_wavelet(n: int, scale: float, order: int = 4) -> np.ndarray:
    """Paul wavelet of given order."""
    t = np.arange(n, dtype=float) - n / 2
    t = t / scale
    from math import factorial
    norm_const = (2 ** order * factorial(order)) / np.sqrt(
        np.pi * factorial(2 * order)
    ) / np.sqrt(scale)
    return norm_const * (1 - 1j * t) ** (-(order + 1))


def _get_wavelet_func(wtype: WaveletType):
    if wtype == WaveletType.MORLET:
        return morlet_wavelet
    elif wtype == WaveletType.MEXICAN_HAT:
        return mexican_hat_wavelet
    elif wtype == WaveletType.PAUL:
        return paul_wavelet
    raise ValueError(f"Unknown wavelet type: {wtype}")


def cwt(
    signal: np.ndarray,
    scales: np.ndarray,
    wavelet: WaveletType = WaveletType.MORLET,
    omega0: float = 6.0,
) -> np.ndarray:
    """Continuous wavelet transform via FFT convolution.

    Returns complex coefficient matrix (n_scales, n_samples).
    """
    n = len(signal)
    wfunc = _get_wavelet_func(wavelet)
    coeffs = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        wavelet_len = min(int(10 * s), n)
        if wavelet_len < 4:
            wavelet_len = 4
        if wavelet == WaveletType.MORLET:
            w = morlet_wavelet(wavelet_len, s, omega0)
        elif wavelet == WaveletType.MEXICAN_HAT:
            w = mexican_hat_wavelet(wavelet_len, s)
        else:
            w = paul_wavelet(wavelet_len, s)
        conv = fftconvolve(signal, np.conj(w[::-1]), mode="same")
        coeffs[i, :] = conv
    return coeffs


def scaleogram(
    signal: np.ndarray,
    sr: float = 22050.0,
    n_scales: int = 64,
    fmin: float = 30.0,
    fmax: float = 8000.0,
    wavelet: WaveletType = WaveletType.MORLET,
    omega0: float = 6.0,
) -> ScaleogramResult:
    """Compute scaleogram (wavelet power spectrum) with geometric scale spacing."""
    frequencies = np.geomspace(fmax, fmin, n_scales)
    scales = omega0 * sr / (2 * np.pi * frequencies)
    coeffs = cwt(signal, scales, wavelet, omega0)
    power = np.abs(coeffs) ** 2
    return ScaleogramResult(
        coefficients=coeffs,
        scales=scales,
        frequencies=frequencies,
        power=power,
    )


def wavelet_spectral_centroid(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Spectral centroid per time frame from wavelet power."""
    freqs = frequencies[:, None]
    total = power.sum(axis=0)
    total = np.maximum(total, 1e-10)
    return (power * freqs).sum(axis=0) / total


def energy_per_scale(power: np.ndarray) -> np.ndarray:
    """Total energy per wavelet scale."""
    return power.sum(axis=1)


def detect_ridges(
    power: np.ndarray,
    min_length: int = 10,
) -> List[np.ndarray]:
    """Ridge detection via greedy scale-tracking on power maxima."""
    n_scales, n_time = power.shape
    ridges: List[List[int]] = []
    active: List[List[int]] = []

    for t in range(n_time):
        col = power[:, t]
        # local maxima in scale dimension
        peaks: List[int] = []
        for s in range(1, n_scales - 1):
            if col[s] > col[s - 1] and col[s] > col[s + 1]:
                peaks.append(s)
        if not peaks and n_scales > 0:
            peaks = [int(np.argmax(col))]

        used_peaks: set = set()
        new_active: List[List[int]] = []
        for ridge in active:
            last_s = ridge[-1]
            best_p = -1
            best_d = float("inf")
            for p in peaks:
                if p not in used_peaks:
                    d = abs(p - last_s)
                    if d < best_d and d <= 2:
                        best_d = d
                        best_p = p
            if best_p >= 0:
                ridge.append(best_p)
                used_peaks.add(best_p)
                new_active.append(ridge)
            else:
                if len(ridge) >= min_length:
                    ridges.append(ridge)

        for p in peaks:
            if p not in used_peaks:
                new_active.append([p])
        active = new_active

    for ridge in active:
        if len(ridge) >= min_length:
            ridges.append(ridge)

    return [np.array(r) for r in ridges]


def extract_wavelet_features(
    signal: np.ndarray,
    sr: float = 22050.0,
    n_scales: int = 64,
    wavelet: WaveletType = WaveletType.MORLET,
) -> WaveletFeatures:
    """Full wavelet feature extraction pipeline."""
    sg = scaleogram(signal, sr, n_scales, wavelet=wavelet)
    sc = wavelet_spectral_centroid(sg.power, sg.frequencies)
    eps = energy_per_scale(sg.power)
    ridges = detect_ridges(sg.power)
    total = float(sg.power.sum())
    return WaveletFeatures(
        spectral_centroid=sc,
        energy_per_scale=eps,
        ridges=ridges,
        total_energy=total,
    )


def dwt_decompose(
    signal: np.ndarray,
    n_levels: int = 4,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Simple discrete wavelet transform using Haar wavelet (pure numpy).

    Returns list of (approximation, detail) coefficient pairs per level.
    """
    levels: List[Tuple[np.ndarray, np.ndarray]] = []
    current = signal.astype(float)
    for _ in range(n_levels):
        if len(current) < 2:
            break
        n = len(current) - len(current) % 2
        even = current[:n:2]
        odd = current[1:n:2]
        approx = (even + odd) / np.sqrt(2)
        detail = (even - odd) / np.sqrt(2)
        levels.append((approx, detail))
        current = approx
    return levels


def dwt_energy_distribution(levels: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Energy distribution across DWT decomposition levels."""
    energies = []
    for approx, detail in levels:
        energies.append(float(np.sum(detail ** 2)))
    # Add final approximation energy
    if levels:
        energies.append(float(np.sum(levels[-1][0] ** 2)))
    total = sum(energies)
    if total > 0:
        return np.array(energies) / total
    return np.array(energies)
