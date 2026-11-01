"""Spectral source separation: NMF, harmonic-percussive, vocal isolation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stft(signal: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    window = np.hanning(n_fft)
    n = len(signal)
    n_frames = max(1, 1 + (n - n_fft) // hop)
    S = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = np.zeros(n_fft)
        end = min(start + n_fft, n)
        frame[: end - start] = signal[start:end] * window[: end - start]
        S[:, i] = np.fft.rfft(frame)
    return S


def istft(S: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    window = np.hanning(n_fft)
    n_bins, n_frames = S.shape
    length = n_fft + (n_frames - 1) * hop
    output = np.zeros(length, dtype=np.float64)
    norm = np.zeros(length, dtype=np.float64)
    for i in range(n_frames):
        frame = np.fft.irfft(S[:, i], n=n_fft)
        start = i * hop
        output[start : start + n_fft] += frame * window
        norm[start : start + n_fft] += window ** 2
    norm = np.maximum(norm, 1e-12)
    return output / norm


def harmonic_percussive_separation(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    kernel_h: int = 31,
    kernel_p: int = 31,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    S = stft(signal, n_fft=n_fft, hop=hop)
    mag = np.abs(S)
    from scipy.ndimage import median_filter
    H = median_filter(mag, size=(1, kernel_h))
    P = median_filter(mag, size=(kernel_p, 1))
    Hm = np.maximum(H, 1e-12)
    Pm = np.maximum(P, 1e-12)
    mask_h = (Hm / (Hm + Pm + 1e-12)) ** margin
    mask_p = (Pm / (Hm + Pm + 1e-12)) ** margin
    h_raw = istft(S * mask_h, n_fft=n_fft, hop=hop)
    p_raw = istft(S * mask_p, n_fft=n_fft, hop=hop)
    n = len(signal)
    harmonic = np.zeros(n, dtype=np.float64)
    percussive = np.zeros(n, dtype=np.float64)
    harmonic[: min(n, len(h_raw))] = h_raw[: min(n, len(h_raw))]
    percussive[: min(n, len(p_raw))] = p_raw[: min(n, len(p_raw))]
    return harmonic, percussive


@dataclass
class NMFResult:
    W: np.ndarray
    H: np.ndarray
    cost: list[float]
    n_iter: int


def nmf_separate(
    mag: np.ndarray,
    n_components: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> NMFResult:
    V = np.maximum(mag, 1e-12)
    n_freq, n_frames = V.shape
    rng = np.random.RandomState(42)
    W = rng.rand(n_freq, n_components).astype(np.float64) + 0.1
    H = rng.rand(n_components, n_frames).astype(np.float64) + 0.1
    costs = []
    for it in range(max_iter):
        WH = W @ H + 1e-12
        cost = float(np.sum(V * np.log(V / WH) - V + WH))
        costs.append(cost)
        if it > 0 and abs(costs[-1] - costs[-2]) < tol:
            break
        H *= (W.T @ (V / WH)) / (W.T @ np.ones_like(V) + 1e-12)
        WH = W @ H + 1e-12
        W *= ((V / WH) @ H.T) / (np.ones_like(V) @ H.T + 1e-12)
    return NMFResult(W=W, H=H, cost=costs, n_iter=len(costs))


def nmf_reconstruct(
    S_complex: np.ndarray, W: np.ndarray, H: np.ndarray, component: int,
) -> np.ndarray:
    mag = np.abs(S_complex)
    total = W @ H + 1e-12
    source_mag = np.outer(W[:, component], H[component, :])
    mask = source_mag / total
    return S_complex * mask


def spectral_centroid_separation(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop: int = 512,
    cutoff_hz: float = 4000.0,
) -> tuple[np.ndarray, np.ndarray]:
    S = stft(signal, n_fft=n_fft, hop=hop)
    freqs = np.linspace(0, sr / 2, S.shape[0])
    low_mask = (freqs <= cutoff_hz).astype(float)[:, None]
    high_mask = 1.0 - low_mask
    lo_raw = istft(S * low_mask, n_fft=n_fft, hop=hop)
    hi_raw = istft(S * high_mask, n_fft=n_fft, hop=hop)
    n = len(signal)
    low = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low[: min(n, len(lo_raw))] = lo_raw[: min(n, len(lo_raw))]
    high[: min(n, len(hi_raw))] = hi_raw[: min(n, len(hi_raw))]
    return low, high


def wiener_filter(
    mag_sources: list[np.ndarray], S_mix: np.ndarray,
) -> list[np.ndarray]:
    total = sum(m ** 2 for m in mag_sources)
    total = np.maximum(total, 1e-12)
    separated = []
    for m in mag_sources:
        mask = m ** 2 / total
        separated.append(S_mix * mask)
    return separated
