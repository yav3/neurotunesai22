"""Constant-Q transform: logarithmic frequency resolution for pitch analysis.

Implements the CQT with geometric frequency spacing, providing pitch-aligned
spectral analysis with configurable bins-per-octave and hop size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CQTParams:
    sr: float = 44100.0
    fmin: float = 32.7
    n_bins: int = 84
    bins_per_octave: int = 12
    hop_length: int = 512


def cqt_frequencies(fmin: float, n_bins: int, bins_per_octave: int = 12) -> np.ndarray:
    return fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))


def cqt_kernel(
    sr: float,
    fmin: float,
    n_bins: int,
    bins_per_octave: int = 12,
) -> list[np.ndarray]:
    freqs = cqt_frequencies(fmin, n_bins, bins_per_octave)
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    kernels: list[np.ndarray] = []
    for f_k in freqs:
        N_k = int(math.ceil(Q * sr / f_k))
        if N_k < 1:
            N_k = 1
        n = np.arange(N_k)
        window = np.hamming(N_k)
        kernel = (1.0 / N_k) * window * np.exp(2j * math.pi * f_k * n / sr)
        kernels.append(kernel)
    return kernels


def compute_cqt(
    signal: np.ndarray,
    params: CQTParams | None = None,
) -> np.ndarray:
    p = params or CQTParams()
    kernels = cqt_kernel(p.sr, p.fmin, p.n_bins, p.bins_per_octave)
    n_frames = max(1, (len(signal) - max(len(k) for k in kernels)) // p.hop_length + 1)
    cqt_matrix = np.zeros((p.n_bins, n_frames), dtype=complex)
    for b, kernel in enumerate(kernels):
        klen = len(kernel)
        for t in range(n_frames):
            start = t * p.hop_length
            end = start + klen
            if end > len(signal):
                break
            cqt_matrix[b, t] = np.sum(signal[start:end] * np.conj(kernel))
    return cqt_matrix


def cqt_to_chroma(cqt_matrix: np.ndarray, bins_per_octave: int = 12) -> np.ndarray:
    n_bins, n_frames = cqt_matrix.shape
    chroma = np.zeros((bins_per_octave, n_frames))
    mag = np.abs(cqt_matrix)
    for b in range(n_bins):
        chroma_bin = b % bins_per_octave
        chroma[chroma_bin] += mag[b]
    total = chroma.sum(axis=0, keepdims=True)
    total[total < 1e-12] = 1.0
    return chroma / total


def cqt_spectrogram(
    signal: np.ndarray,
    sr: float = 44100.0,
    fmin: float = 32.7,
    n_bins: int = 84,
    hop: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = CQTParams(sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop)
    cqt = compute_cqt(signal, params)
    freqs = cqt_frequencies(fmin, n_bins)
    n_frames = cqt.shape[1]
    times = np.arange(n_frames) * hop / sr
    return np.abs(cqt), freqs, times


def harmonic_cqt(
    cqt_matrix: np.ndarray,
    n_harmonics: int = 5,
    bins_per_octave: int = 12,
) -> np.ndarray:
    n_bins, n_frames = cqt_matrix.shape
    hcqt = np.zeros_like(cqt_matrix)
    for h in range(1, n_harmonics + 1):
        shift = int(round(bins_per_octave * math.log2(h)))
        for b in range(n_bins):
            src = b + shift
            if 0 <= src < n_bins:
                hcqt[b] += np.abs(cqt_matrix[src]) / h
    return hcqt


def pitch_salience(
    cqt_matrix: np.ndarray,
    bins_per_octave: int = 12,
    n_harmonics: int = 5,
) -> np.ndarray:
    hcqt = harmonic_cqt(cqt_matrix, n_harmonics, bins_per_octave)
    salience = np.sum(np.abs(hcqt), axis=0)
    peak = np.max(salience)
    if peak > 0:
        salience /= peak
    return salience
