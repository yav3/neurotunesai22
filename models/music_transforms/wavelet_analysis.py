"""Continuous Wavelet Transform for music time-frequency analysis."""
from __future__ import annotations

import numpy as np


def morlet_wavelet(n: int, w0: float = 6.0) -> np.ndarray:
    t = np.linspace(-4, 4, n)
    envelope = np.exp(-0.5 * t ** 2)
    carrier = np.exp(1j * w0 * t)
    correction = np.exp(-0.5 * w0 ** 2)
    return np.pi ** (-0.25) * envelope * (carrier - correction)


def cwt_transform(signal: np.ndarray, scales: np.ndarray, w0: float = 6.0) -> np.ndarray:
    n = len(signal)
    coeffs = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        if s < 1e-10:
            continue
        wavelet_len = min(int(10 * s), n)
        if wavelet_len < 4:
            wavelet_len = 4
        t = np.linspace(-5, 5, wavelet_len)
        psi = np.pi ** (-0.25) * np.exp(-0.5 * t ** 2) * np.exp(1j * w0 * t)
        psi /= np.sqrt(s)
        padded = np.pad(signal, (wavelet_len // 2, wavelet_len // 2), mode='reflect')
        for j in range(n):
            seg = padded[j:j + wavelet_len]
            coeffs[i, j] = np.sum(seg * np.conj(psi))
    return coeffs


def scalogram(cwt_coeffs: np.ndarray) -> np.ndarray:
    return np.abs(cwt_coeffs) ** 2


def scale_to_frequency(scales: np.ndarray, sr: float, w0: float = 6.0) -> np.ndarray:
    return w0 * sr / (2.0 * np.pi * scales)


def wavelet_ridge_extraction(cwt_coeffs: np.ndarray, scales: np.ndarray,
                              penalty: float = 1.0) -> np.ndarray:
    power = np.abs(cwt_coeffs) ** 2
    n_times = power.shape[1]
    ridge = np.zeros(n_times, dtype=int)
    ridge[0] = np.argmax(power[:, 0])
    for t in range(1, n_times):
        best_s = 0
        best_val = -np.inf
        for s in range(len(scales)):
            val = power[s, t] - penalty * abs(s - ridge[t - 1])
            if val > best_val:
                best_val = val
                best_s = s
        ridge[t] = best_s
    return ridge


def instantaneous_frequency_from_cwt(cwt_coeffs: np.ndarray, scales: np.ndarray,
                                      sr: float, w0: float = 6.0) -> np.ndarray:
    ridge = wavelet_ridge_extraction(cwt_coeffs, scales)
    freqs = scale_to_frequency(scales, sr, w0)
    return freqs[ridge]


def cross_wavelet_coherence(cwt1: np.ndarray, cwt2: np.ndarray,
                             smooth_width: int = 5) -> np.ndarray:
    cross = cwt1 * np.conj(cwt2)
    kernel = np.ones(smooth_width) / smooth_width
    s12 = np.zeros_like(cross, dtype=complex)
    s11 = np.zeros_like(cross, dtype=float)
    s22 = np.zeros_like(cross, dtype=float)
    for i in range(cross.shape[0]):
        s12[i] = np.convolve(cross[i], kernel, mode='same')
        s11[i] = np.convolve(np.abs(cwt1[i]) ** 2, kernel, mode='same')
        s22[i] = np.convolve(np.abs(cwt2[i]) ** 2, kernel, mode='same')
    denom = np.sqrt(s11 * s22)
    denom = np.where(denom > 1e-20, denom, 1e-20)
    return np.abs(s12) ** 2 / (denom ** 2)
