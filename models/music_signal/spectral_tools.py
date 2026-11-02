"""Spectral analysis utilities for musical signals."""
from __future__ import annotations

import math

import numpy as np

from .windowing import WindowType, window_function


def stft(
    signal: np.ndarray, n_fft: int = 2048, hop: int = 512,
    wtype: WindowType = WindowType.HANN,
) -> np.ndarray:
    w = window_function(wtype, n_fft)
    n_frames = 1 + (len(signal) - n_fft) // hop
    if n_frames <= 0:
        padded = np.pad(signal, (0, n_fft - len(signal)))
        return np.fft.rfft(padded * w).reshape(1, -1)
    S = np.zeros((n_frames, n_fft // 2 + 1), dtype=complex)
    for i in range(n_frames):
        frame = signal[i * hop: i * hop + n_fft] * w
        S[i] = np.fft.rfft(frame)
    return S


def istft(
    S: np.ndarray, hop: int = 512,
    wtype: WindowType = WindowType.HANN,
) -> np.ndarray:
    n_frames, n_bins = S.shape
    n_fft = (n_bins - 1) * 2
    w = window_function(wtype, n_fft)
    length = n_fft + (n_frames - 1) * hop
    out = np.zeros(length)
    win_sum = np.zeros(length)
    for i in range(n_frames):
        frame = np.fft.irfft(S[i], n_fft) * w
        start = i * hop
        out[start: start + n_fft] += frame
        win_sum[start: start + n_fft] += w ** 2
    nonzero = win_sum > 1e-10
    out[nonzero] /= win_sum[nonzero]
    return out


def magnitude_to_db(S: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(S), amin) / ref)


def db_to_magnitude(S_db: np.ndarray, ref: float = 1.0) -> np.ndarray:
    return ref * 10 ** (S_db / 20)


def hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sr: float, fmin: float = 0.0, fmax: float | None = None,
) -> np.ndarray:
    fmax = fmax or sr / 2
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fbank = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fbank[i, j] = (right - j) / (right - center)
    return fbank


def cqt_frequencies(n_bins: int, fmin: float, bins_per_octave: int = 12) -> np.ndarray:
    return fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)


def spectral_centroid(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    mag = np.abs(S)
    if mag.ndim == 1:
        mag = mag.reshape(1, -1)
    total = np.sum(mag, axis=1)
    total = np.where(total == 0, 1, total)
    return np.sum(mag * freqs[np.newaxis, :], axis=1) / total


def spectral_bandwidth(
    S: np.ndarray, freqs: np.ndarray, centroid: np.ndarray | None = None,
) -> np.ndarray:
    mag = np.abs(S)
    if mag.ndim == 1:
        mag = mag.reshape(1, -1)
    if centroid is None:
        centroid = spectral_centroid(S, freqs)
    total = np.sum(mag, axis=1)
    total = np.where(total == 0, 1, total)
    deviation = freqs[np.newaxis, :] - centroid[:, np.newaxis]
    return np.sqrt(np.sum(mag * deviation ** 2, axis=1) / total)


def spectral_contrast(
    S: np.ndarray, freqs: np.ndarray, n_bands: int = 6,
) -> np.ndarray:
    mag = np.abs(S)
    if mag.ndim == 1:
        mag = mag.reshape(1, -1)
    fmin = 200.0
    octa = np.zeros(n_bands + 1)
    for i in range(n_bands + 1):
        octa[i] = fmin * (2 ** i)
    contrast = np.zeros((mag.shape[0], n_bands))
    for b in range(n_bands):
        lo = octa[b]
        hi = octa[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        band = mag[:, mask]
        band_sorted = np.sort(band, axis=1)
        n_cols = band_sorted.shape[1]
        alpha = max(1, int(0.2 * n_cols))
        peaks = np.mean(band_sorted[:, -alpha:], axis=1)
        valleys = np.mean(band_sorted[:, :alpha], axis=1)
        contrast[:, b] = np.log1p(peaks) - np.log1p(valleys)
    return contrast


def spectral_rolloff(
    S: np.ndarray, freqs: np.ndarray, percentile: float = 0.85,
) -> np.ndarray:
    mag = np.abs(S)
    if mag.ndim == 1:
        mag = mag.reshape(1, -1)
    total = np.sum(mag, axis=1, keepdims=True)
    cumsum = np.cumsum(mag, axis=1)
    threshold = percentile * total
    rolloff = np.zeros(mag.shape[0])
    for i in range(mag.shape[0]):
        idx = np.searchsorted(cumsum[i], threshold[i, 0])
        idx = min(idx, len(freqs) - 1)
        rolloff[i] = freqs[idx]
    return rolloff


def chromagram(
    S: np.ndarray, freqs: np.ndarray, n_chroma: int = 12,
) -> np.ndarray:
    mag = np.abs(S)
    if mag.ndim == 1:
        mag = mag.reshape(1, -1)
    chroma = np.zeros((mag.shape[0], n_chroma))
    for j, f in enumerate(freqs):
        if f <= 0:
            continue
        pitch_class = int(round(12 * math.log2(f / 440.0))) % n_chroma
        chroma[:, pitch_class] += mag[:, j]
    norm = np.max(chroma, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return chroma / norm
