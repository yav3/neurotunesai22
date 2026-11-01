"""Harmonic analysis transforms: HPSS, HNR, pitch estimation, subharmonics."""
from __future__ import annotations

import numpy as np


def _stft_magnitude(signal: np.ndarray, frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_size) / frame_size))
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    mag = np.zeros((frame_size // 2 + 1, n_frames))
    for i in range(n_frames):
        chunk = signal[i * hop:i * hop + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        mag[:, i] = np.abs(np.fft.rfft(chunk * window))
    return mag


def _median_filter_rows(mat: np.ndarray, kernel: int) -> np.ndarray:
    half = kernel // 2
    out = np.empty_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            lo = max(0, j - half)
            hi = min(mat.shape[1], j + half + 1)
            out[i, j] = np.median(mat[i, lo:hi])
    return out


def _median_filter_cols(mat: np.ndarray, kernel: int) -> np.ndarray:
    half = kernel // 2
    out = np.empty_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            lo = max(0, i - half)
            hi = min(mat.shape[0], i + half + 1)
            out[i, j] = np.median(mat[lo:hi, j])
    return out


def hpss(signal: np.ndarray, frame_size: int = 2048, hop: int = 512,
         kernel_size: int = 31) -> tuple[np.ndarray, np.ndarray]:
    mag = _stft_magnitude(signal, frame_size, hop)
    harmonic_enhanced = _median_filter_rows(mag, kernel_size)
    percussive_enhanced = _median_filter_cols(mag, kernel_size)
    total = harmonic_enhanced + percussive_enhanced + 1e-10
    h_mask = harmonic_enhanced / total
    p_mask = percussive_enhanced / total
    h_energy = np.sum(mag * h_mask, axis=0)
    p_energy = np.sum(mag * p_mask, axis=0)
    return h_energy, p_energy


def harmonic_product_spectrum(signal: np.ndarray, sr: float = 44100.0,
                              n_harmonics: int = 5,
                              n_fft: int = 4096) -> tuple[float, np.ndarray]:
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
    spectrum = np.abs(np.fft.rfft(signal[:n_fft] * window))
    hps = spectrum.copy()
    for h in range(2, n_harmonics + 1):
        decimated = spectrum[::h]
        hps[:len(decimated)] *= decimated
    peak_bin = np.argmax(hps[1:]) + 1
    f0 = peak_bin * sr / n_fft
    return f0, hps


def harmonic_to_noise_ratio(signal: np.ndarray, sr: float = 44100.0,
                             frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    hnr = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * hop:i * hop + frame_size].astype(np.float64)
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        ac = np.correlate(frame, frame, mode='full')
        ac = ac[len(ac) // 2:]
        if ac[0] < 1e-10:
            hnr[i] = -100.0
            continue
        ac_norm = ac / ac[0]
        min_lag = int(sr / 1000.0)
        max_lag = int(sr / 50.0)
        search = ac_norm[min_lag:min(max_lag, len(ac_norm))]
        if len(search) == 0:
            hnr[i] = -100.0
            continue
        r_max = np.max(search)
        r_max = max(r_max, 1e-10)
        if r_max >= 1.0:
            hnr[i] = 100.0
        else:
            hnr[i] = 10 * np.log10(r_max / (1 - r_max + 1e-10))
    return hnr


def subharmonic_summation(spectrum: np.ndarray, sr: float = 44100.0,
                          n_subharmonics: int = 5,
                          n_fft: int = 4096) -> tuple[float, np.ndarray]:
    freqs = np.arange(len(spectrum)) * sr / n_fft
    shs = np.zeros_like(spectrum, dtype=np.float64)
    for k in range(1, n_subharmonics + 1):
        weight = 1.0 / k
        indices = np.arange(len(spectrum)) * k
        valid = indices < len(spectrum)
        shs[valid] += weight * spectrum[indices[valid]]
    min_bin = max(1, int(50.0 * n_fft / sr))
    max_bin = min(len(shs), int(2000.0 * n_fft / sr))
    peak = np.argmax(shs[min_bin:max_bin]) + min_bin
    return freqs[peak], shs


def autocorrelation_pitch(signal: np.ndarray, sr: float = 44100.0,
                          min_freq: float = 50.0,
                          max_freq: float = 2000.0) -> float:
    n = len(signal)
    if n < 2:
        return 0.0
    sig = signal.astype(np.float64)
    sig -= np.mean(sig)
    ac = np.correlate(sig, sig, mode='full')
    ac = ac[n - 1:]
    if ac[0] < 1e-10:
        return 0.0
    ac = ac / ac[0]
    min_lag = max(1, int(sr / max_freq))
    max_lag = min(n - 1, int(sr / min_freq))
    search = ac[min_lag:max_lag + 1]
    if len(search) == 0:
        return 0.0
    peak_offset = np.argmax(search)
    lag = min_lag + peak_offset
    if lag == 0:
        return 0.0
    return sr / lag


def zero_crossing_rate(signal: np.ndarray, frame_size: int = 2048,
                       hop: int = 512) -> np.ndarray:
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * hop:i * hop + frame_size]
        if len(frame) < 2:
            continue
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        zcr[i] = crossings / (2 * len(frame))
    return zcr
