"""Reassigned spectrogram for sharper time-frequency representation."""
from __future__ import annotations

import numpy as np


def _stft_frame(signal: np.ndarray, center: int, window: np.ndarray) -> np.ndarray:
    n_fft = len(window)
    half = n_fft // 2
    start = center - half
    frame = np.zeros(n_fft)
    for i in range(n_fft):
        idx = start + i
        if 0 <= idx < len(signal):
            frame[i] = signal[idx]
    return frame * window


def compute_stft_derivatives(signal: np.ndarray, n_fft: int = 1024,
                              hop: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(signal)
    n_frames = max(1, (n - n_fft) // hop + 1)
    n_bins = n_fft // 2 + 1
    window = np.hanning(n_fft)
    t_ramp = np.arange(n_fft) - n_fft / 2.0
    d_window = -t_ramp / (n_fft / 2.0) ** 2 * window * n_fft
    t_window = t_ramp * window
    S = np.zeros((n_bins, n_frames), dtype=complex)
    S_d = np.zeros_like(S)
    S_t = np.zeros_like(S)
    for t in range(n_frames):
        center = t * hop + n_fft // 2
        S[:, t] = np.fft.rfft(_stft_frame(signal, center, window))
        S_d[:, t] = np.fft.rfft(_stft_frame(signal, center, d_window))
        S_t[:, t] = np.fft.rfft(_stft_frame(signal, center, t_window))
    return S, S_d, S_t


def reassign_time_frequency(S: np.ndarray, S_d: np.ndarray, S_t: np.ndarray,
                             sr: float, hop: int, n_fft: int
                             ) -> tuple[np.ndarray, np.ndarray]:
    n_bins, n_frames = S.shape
    mag = np.abs(S)
    threshold = 1e-10 * np.max(mag)
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    time_centers = np.arange(n_frames) * hop / sr
    f_reassigned = np.tile(freq_bins[:, None], (1, n_frames))
    t_reassigned = np.tile(time_centers[None, :], (n_bins, 1))
    mask = mag > threshold
    f_reassigned[mask] = freq_bins[np.where(mask)[0]] - np.imag(S_d[mask] / S[mask]) * sr / (2 * np.pi)
    t_reassigned[mask] = time_centers[np.where(mask)[1]] + np.real(S_t[mask] / S[mask]) / sr
    return f_reassigned, t_reassigned


def reassigned_spectrogram(signal: np.ndarray, sr: float, n_fft: int = 1024,
                            hop: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S, S_d, S_t = compute_stft_derivatives(signal, n_fft, hop)
    f_r, t_r = reassign_time_frequency(S, S_d, S_t, sr, hop, n_fft)
    return np.abs(S) ** 2, f_r, t_r


def synchrosqueezed_transform(signal: np.ndarray, sr: float, n_fft: int = 1024,
                               hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    S, S_d, S_t = compute_stft_derivatives(signal, n_fft, hop)
    n_bins, n_frames = S.shape
    mag = np.abs(S)
    threshold = 1e-10 * np.max(mag)
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    inst_freq = np.zeros_like(mag)
    mask = mag > threshold
    inst_freq[mask] = freq_bins[np.where(mask)[0]] - np.imag(S_d[mask] / S[mask]) * sr / (2 * np.pi)
    sst = np.zeros_like(S)
    df = freq_bins[1] - freq_bins[0] if len(freq_bins) > 1 else 1.0
    for t in range(n_frames):
        for k in range(n_bins):
            if mag[k, t] > threshold:
                target_bin = int(round(inst_freq[k, t] / df))
                target_bin = max(0, min(n_bins - 1, target_bin))
                sst[target_bin, t] += S[k, t]
    return np.abs(sst) ** 2, freq_bins


def extract_ridges(power: np.ndarray, n_ridges: int = 3,
                   min_length: int = 10) -> list[np.ndarray]:
    n_bins, n_frames = power.shape
    ridges = []
    remaining = power.copy()
    for _ in range(n_ridges):
        ridge = np.zeros(n_frames, dtype=int)
        ridge[0] = np.argmax(remaining[:, 0])
        for t in range(1, n_frames):
            low = max(0, ridge[t - 1] - 2)
            high = min(n_bins, ridge[t - 1] + 3)
            ridge[t] = low + np.argmax(remaining[low:high, t])
        if n_frames >= min_length:
            ridges.append(ridge)
        for t in range(n_frames):
            lo = max(0, ridge[t] - 1)
            hi = min(n_bins, ridge[t] + 2)
            remaining[lo:hi, t] = 0
    return ridges


def ridge_to_frequency(ridge: np.ndarray, sr: float, n_fft: int) -> np.ndarray:
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return freq_bins[np.clip(ridge, 0, len(freq_bins) - 1)]
