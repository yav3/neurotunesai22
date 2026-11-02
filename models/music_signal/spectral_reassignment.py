"""Spectral reassignment: sharpen time-frequency representations.

Implements the reassignment method of Auger & Flandrin (1995) for STFT,
synchrosqueezed transforms, and instantaneous frequency/group delay
estimation with sub-bin precision.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ReassignmentResult:
    magnitudes: np.ndarray
    inst_freq_hz: np.ndarray
    group_delay_s: np.ndarray
    reassigned_spectrogram: np.ndarray


def _analytic_window(N: int, kind: str = "hann") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(N)
    if kind == "hann":
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif kind == "gaussian":
        sigma = N / 6.0
        w = np.exp(-0.5 * ((n - (N - 1) / 2) / sigma) ** 2)
    else:
        w = np.ones(N)
    dw = np.zeros(N)
    dw[1:-1] = (w[2:] - w[:-2]) / 2.0
    tw = n * w
    return w, dw, tw


def instantaneous_frequency(stft_h: np.ndarray, stft_dh: np.ndarray,
                            sr: int, hop: int) -> np.ndarray:
    n_freq = stft_h.shape[0]
    omega_k = np.arange(n_freq) * sr / (2.0 * (n_freq - 1))
    omega_k = omega_k[:, np.newaxis]
    eps = 1e-14
    magnitude = np.abs(stft_h)
    mask = magnitude > eps
    inst_f = np.full_like(magnitude, 0.0)
    ratio = np.zeros_like(stft_h)
    ratio[mask] = stft_dh[mask] / stft_h[mask]
    inst_f = omega_k - np.imag(ratio) * sr / (2 * np.pi)
    inst_f[~mask] = omega_k[~mask] if omega_k.shape == inst_f[~mask].shape else 0.0
    return inst_f


def group_delay(stft_h: np.ndarray, stft_th: np.ndarray,
                hop: int, sr: int) -> np.ndarray:
    eps = 1e-14
    magnitude = np.abs(stft_h)
    mask = magnitude > eps
    gd = np.zeros_like(magnitude)
    gd[mask] = np.real(stft_th[mask] / stft_h[mask]) / sr
    return gd


def reassigned_stft(audio: np.ndarray, sr: int = 44100,
                    n_fft: int = 2048, hop: int = 512,
                    window: str = "hann") -> ReassignmentResult:
    w, dw, tw = _analytic_window(n_fft, window)
    n_frames = max(1, (len(audio) - n_fft) // hop + 1)
    n_freq = n_fft // 2 + 1
    stft_h = np.zeros((n_freq, n_frames), dtype=complex)
    stft_dh = np.zeros((n_freq, n_frames), dtype=complex)
    stft_th = np.zeros((n_freq, n_frames), dtype=complex)
    for t in range(n_frames):
        start = t * hop
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        stft_h[:, t] = np.fft.rfft(frame * w)
        stft_dh[:, t] = np.fft.rfft(frame * dw)
        stft_th[:, t] = np.fft.rfft(frame * tw)
    magnitudes = np.abs(stft_h)
    inst_f = instantaneous_frequency(stft_h, stft_dh, sr, hop)
    gd = group_delay(stft_h, stft_th, hop, sr)
    freq_bins = np.linspace(0, sr / 2, n_freq)
    time_bins = np.arange(n_frames) * hop / sr
    reassigned = np.zeros((n_freq, n_frames))
    for t in range(n_frames):
        for k in range(n_freq):
            mag = magnitudes[k, t]
            if mag < 1e-10:
                continue
            new_k = int(np.round(inst_f[k, t] / (sr / 2) * (n_freq - 1)))
            new_t = int(np.round(gd[k, t] / (hop / sr)))
            new_k = max(0, min(new_k, n_freq - 1))
            new_t = max(0, min(new_t, n_frames - 1))
            reassigned[new_k, new_t] += mag ** 2
    return ReassignmentResult(
        magnitudes=magnitudes,
        inst_freq_hz=inst_f,
        group_delay_s=gd,
        reassigned_spectrogram=reassigned,
    )


def synchrosqueeze(stft_h: np.ndarray, inst_freq_hz: np.ndarray,
                   sr: int, n_fft: int) -> np.ndarray:
    n_freq, n_frames = stft_h.shape
    squeezed = np.zeros((n_freq, n_frames))
    freq_res = sr / n_fft
    for t in range(n_frames):
        for k in range(n_freq):
            mag = np.abs(stft_h[k, t])
            if mag < 1e-10:
                continue
            target_bin = int(np.round(inst_freq_hz[k, t] / freq_res))
            target_bin = max(0, min(target_bin, n_freq - 1))
            squeezed[target_bin, t] += mag
    return squeezed


def spectral_flatness_profile(spectrogram: np.ndarray) -> np.ndarray:
    n_frames = spectrogram.shape[1]
    flatness = np.zeros(n_frames)
    for t in range(n_frames):
        col = spectrogram[:, t]
        col = col[col > 0]
        if len(col) < 2:
            flatness[t] = 0.0
            continue
        geo = np.exp(np.mean(np.log(col + 1e-20)))
        arith = np.mean(col)
        flatness[t] = geo / max(arith, 1e-20)
    return flatness


def chromagram_from_reassigned(reassigned: np.ndarray, sr: int,
                               n_fft: int, n_chroma: int = 12) -> np.ndarray:
    n_freq, n_frames = reassigned.shape
    chroma = np.zeros((n_chroma, n_frames))
    freqs = np.linspace(0, sr / 2, n_freq)
    for k in range(1, n_freq):
        if freqs[k] < 20:
            continue
        midi = 69 + 12 * np.log2(freqs[k] / 440.0)
        pc = int(np.round(midi)) % 12
        chroma[pc] += reassigned[k]
    col_max = chroma.max(axis=0, keepdims=True)
    col_max[col_max < 1e-10] = 1.0
    return chroma / col_max
