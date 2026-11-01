"""Wavelet transforms for music: CWT, DWT, multi-resolution, scalogram."""
from __future__ import annotations

import numpy as np


def morlet_wavelet(n: int, omega0: float = 5.0) -> np.ndarray:
    t = np.linspace(-4, 4, n)
    return np.pi ** (-0.25) * np.exp(1j * omega0 * t) * np.exp(-t ** 2 / 2)


def cwt_morlet(signal: np.ndarray, scales: np.ndarray | None = None,
               omega0: float = 5.0) -> np.ndarray:
    n = len(signal)
    if scales is None:
        scales = np.arange(1, min(n // 2, 128) + 1, dtype=np.float64)
    coeffs = np.zeros((len(scales), n), dtype=np.complex128)
    for i, s in enumerate(scales):
        wavelet_len = min(int(10 * s), n)
        if wavelet_len < 2:
            wavelet_len = 2
        wav = morlet_wavelet(wavelet_len, omega0)
        wav = wav / np.sqrt(s)
        pad_left = wavelet_len // 2
        padded = np.pad(signal, (pad_left, wavelet_len - pad_left), mode='reflect')
        for j in range(n):
            segment = padded[j:j + wavelet_len]
            coeffs[i, j] = np.sum(segment * np.conj(wav))
    return coeffs


def scalogram(signal: np.ndarray, scales: np.ndarray | None = None,
              omega0: float = 5.0) -> np.ndarray:
    coeffs = cwt_morlet(signal, scales, omega0)
    return np.abs(coeffs) ** 2


def scales_to_frequencies(scales: np.ndarray, sr: float = 44100.0,
                          omega0: float = 5.0) -> np.ndarray:
    return omega0 * sr / (2 * np.pi * scales)


def haar_wavelet_step(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(data) // 2
    if n == 0:
        return data.copy(), np.array([])
    approx = np.zeros(n)
    detail = np.zeros(n)
    for i in range(n):
        approx[i] = (data[2 * i] + data[2 * i + 1]) / np.sqrt(2)
        detail[i] = (data[2 * i] - data[2 * i + 1]) / np.sqrt(2)
    return approx, detail


def dwt_haar(signal: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    coefficients: list[np.ndarray] = []
    current = signal.astype(np.float64).copy()
    for _ in range(levels):
        if len(current) < 2:
            break
        approx, detail = haar_wavelet_step(current)
        coefficients.append(detail)
        current = approx
    coefficients.append(current)
    coefficients.reverse()
    return coefficients


def idwt_haar(coefficients: list[np.ndarray]) -> np.ndarray:
    if not coefficients:
        return np.array([])
    current = coefficients[0].copy()
    for detail in coefficients[1:]:
        n = len(detail)
        reconstructed = np.zeros(2 * n)
        for i in range(n):
            reconstructed[2 * i] = (current[i] + detail[i]) / np.sqrt(2)
            reconstructed[2 * i + 1] = (current[i] - detail[i]) / np.sqrt(2)
        current = reconstructed
    return current


def multi_resolution_analysis(signal: np.ndarray, levels: int = 4) -> dict[str, np.ndarray]:
    coeffs = dwt_haar(signal, levels)
    result: dict[str, np.ndarray] = {}
    result["approximation"] = coeffs[0]
    for i, detail in enumerate(coeffs[1:], 1):
        result[f"detail_level_{i}"] = detail
    result["n_levels"] = np.array([len(coeffs) - 1])
    return result


def wavelet_packet_decomposition(signal: np.ndarray, max_level: int = 3) -> dict[str, np.ndarray]:
    packets: dict[str, np.ndarray] = {"0_0": signal.astype(np.float64)}
    for level in range(1, max_level + 1):
        parent_keys = [k for k in packets if k.startswith(f"{level - 1}_")]
        for pk in parent_keys:
            parent = packets[pk]
            if len(parent) < 2:
                continue
            approx, detail = haar_wavelet_step(parent)
            idx = int(pk.split("_")[1])
            packets[f"{level}_{2 * idx}"] = approx
            packets[f"{level}_{2 * idx + 1}"] = detail
    return packets


def wavelet_energy_bands(signal: np.ndarray, levels: int = 6) -> np.ndarray:
    coeffs = dwt_haar(signal, levels)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = np.sum(energies) + 1e-10
    return energies / total


def wavelet_entropy(signal: np.ndarray, levels: int = 6) -> float:
    energy_dist = wavelet_energy_bands(signal, levels)
    energy_dist = energy_dist[energy_dist > 1e-15]
    return -float(np.sum(energy_dist * np.log2(energy_dist)))
