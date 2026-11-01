"""Scattering transform for music signals.

First- and second-order scattering coefficients using Morlet wavelets.
Provides time-frequency representations invariant to local time-shifts
while preserving transient/texture information. Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def morlet_wavelet(
    N: int,
    scale: float,
    omega0: float = 5.0,
) -> np.ndarray:
    t = np.arange(N) - N / 2.0
    t = t / scale
    gauss = np.exp(-0.5 * t**2)
    oscillation = np.exp(1j * omega0 * t)
    correction = np.exp(-0.5 * omega0**2)
    psi = gauss * (oscillation - correction) / (np.pi**0.25 * np.sqrt(scale))
    return psi


def wavelet_convolution(signal: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    N = len(signal) + len(wavelet) - 1
    fft_size = 1
    while fft_size < N:
        fft_size <<= 1
    S = np.fft.fft(signal, fft_size)
    W = np.fft.fft(wavelet, fft_size)
    result = np.fft.ifft(S * W)[:len(signal)]
    return result


def lowpass_filter(signal: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return signal.copy()
    kernel = np.ones(scale) / scale
    return np.convolve(np.abs(signal) if np.iscomplexobj(signal) else signal, kernel, mode='same')


def compute_scales(
    n_octaves: int = 8,
    n_per_octave: int = 1,
    base_scale: float = 2.0,
) -> np.ndarray:
    scales = []
    for j in range(n_octaves):
        for q in range(n_per_octave):
            s = base_scale * (2.0 ** (j + q / n_per_octave))
            scales.append(s)
    return np.array(scales)


@dataclass
class ScatteringCoefficients:
    order0: np.ndarray
    order1: np.ndarray
    order2: np.ndarray
    scales_1: np.ndarray
    scales_2: np.ndarray


def scattering_transform(
    signal: np.ndarray,
    n_octaves: int = 8,
    n_per_octave: int = 1,
    averaging_scale: int = 256,
    omega0: float = 5.0,
    max_order: int = 2,
) -> ScatteringCoefficients:
    N = len(signal)
    scales_1 = compute_scales(n_octaves, n_per_octave)

    # Order 0: low-pass of signal
    order0 = lowpass_filter(signal.astype(float), averaging_scale)

    # Order 1: |signal * psi_j1| * phi
    order1_list = []
    u1_list = []
    for s in scales_1:
        psi = morlet_wavelet(min(int(6 * s), N), s, omega0)
        conv = wavelet_convolution(signal.astype(complex), psi)
        u1 = np.abs(conv)
        s1 = lowpass_filter(u1, averaging_scale)
        order1_list.append(s1)
        u1_list.append(u1)
    order1 = np.array(order1_list)

    # Order 2: ||signal * psi_j1| * psi_j2| * phi  (j2 > j1)
    order2_list = []
    scales_2_pairs = []
    if max_order >= 2:
        scales_2 = compute_scales(max(1, n_octaves - 2), n_per_octave)
        for i, s1 in enumerate(scales_1):
            for s2 in scales_2:
                if s2 <= s1:
                    continue
                psi2 = morlet_wavelet(min(int(6 * s2), N), s2, omega0)
                conv2 = wavelet_convolution(u1_list[i].astype(complex), psi2)
                u2 = np.abs(conv2)
                s2_coeff = lowpass_filter(u2, averaging_scale)
                order2_list.append(s2_coeff)
                scales_2_pairs.append((s1, s2))

    order2 = np.array(order2_list) if order2_list else np.empty((0, N))
    scales_2_arr = np.array(scales_2_pairs) if scales_2_pairs else np.empty((0, 2))

    return ScatteringCoefficients(
        order0=order0,
        order1=order1,
        order2=order2,
        scales_1=scales_1,
        scales_2=scales_2_arr,
    )


def scattering_distance(a: ScatteringCoefficients, b: ScatteringCoefficients) -> float:
    d0 = np.mean((a.order0 - b.order0) ** 2)
    n1 = min(a.order1.shape[0], b.order1.shape[0])
    d1 = np.mean((a.order1[:n1] - b.order1[:n1]) ** 2) if n1 > 0 else 0.0
    n2 = min(a.order2.shape[0], b.order2.shape[0])
    d2 = np.mean((a.order2[:n2] - b.order2[:n2]) ** 2) if n2 > 0 else 0.0
    return float(np.sqrt(d0 + d1 + d2))


def scattering_features(coeffs: ScatteringCoefficients) -> np.ndarray:
    feats = []
    feats.append(np.mean(coeffs.order0))
    feats.append(np.std(coeffs.order0))
    for row in coeffs.order1:
        feats.append(np.mean(row))
        feats.append(np.std(row))
    for row in coeffs.order2:
        feats.append(np.mean(row))
    return np.array(feats)


def temporal_scattering_modulation(
    signal: np.ndarray,
    frame_size: int = 2048,
    hop: int = 512,
    n_octaves: int = 6,
) -> np.ndarray:
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    results = []
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        sc = scattering_transform(frame, n_octaves=n_octaves, max_order=1, averaging_scale=frame_size // 4)
        results.append(np.mean(sc.order1, axis=1))
    return np.array(results)
