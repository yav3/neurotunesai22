"""Phase vocoder: time-stretching and pitch-shifting via STFT manipulation.

Implements the phase vocoder algorithm for time-scale modification without
changing pitch, pitch-shifting without changing duration, and phase-locking
for transient preservation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PhaseVocoderConfig:
    n_fft: int = 2048
    hop_length: int = 512
    window: str = "hann"


def _hann_window(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def stft(signal: np.ndarray, n_fft: int = 2048,
         hop: int = 512) -> np.ndarray:
    win = _hann_window(n_fft)
    n_frames = 1 + (len(signal) - n_fft) // hop
    frames = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    for i in range(n_frames):
        segment = signal[i * hop:i * hop + n_fft] * win
        frames[:, i] = np.fft.rfft(segment)
    return frames


def istft(stft_matrix: np.ndarray, hop: int = 512,
          n_fft: int = 2048) -> np.ndarray:
    win = _hann_window(n_fft)
    n_frames = stft_matrix.shape[1]
    length = n_fft + (n_frames - 1) * hop
    output = np.zeros(length)
    window_sum = np.zeros(length)
    for i in range(n_frames):
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        start = i * hop
        output[start:start + n_fft] += frame * win
        window_sum[start:start + n_fft] += win ** 2
    nonzero = window_sum > 1e-10
    output[nonzero] /= window_sum[nonzero]
    return output


def phase_vocoder_stretch(stft_in: np.ndarray, rate: float,
                          hop: int = 512) -> np.ndarray:
    n_bins, n_frames = stft_in.shape
    n_out = int(n_frames / rate)
    phase_advance = np.linspace(0, np.pi * hop, n_bins, endpoint=False)
    out = np.zeros((n_bins, n_out), dtype=complex)
    phase_acc = np.angle(stft_in[:, 0])
    for t in range(n_out):
        src = t * rate
        i0 = int(src)
        frac = src - i0
        i1 = min(i0 + 1, n_frames - 1)
        mag = (1 - frac) * np.abs(stft_in[:, i0]) + frac * np.abs(stft_in[:, i1])
        out[:, t] = mag * np.exp(1j * phase_acc)
        if t < n_out - 1:
            dphi = np.angle(stft_in[:, min(i0 + 1, n_frames - 1)]) - np.angle(stft_in[:, i0])
            dphi -= phase_advance
            dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
            phase_acc += phase_advance + dphi
    return out


def time_stretch(signal: np.ndarray, rate: float, n_fft: int = 2048,
                 hop: int = 512) -> np.ndarray:
    S = stft(signal, n_fft, hop)
    S_stretched = phase_vocoder_stretch(S, rate, hop)
    return istft(S_stretched, hop, n_fft)


def pitch_shift(signal: np.ndarray, sr: int, semitones: float,
                n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    rate = 2.0 ** (semitones / 12.0)
    stretched = time_stretch(signal, rate, n_fft, hop)
    indices = np.linspace(0, len(stretched) - 1, len(signal))
    return np.interp(np.arange(len(signal)),
                     np.arange(len(stretched)), stretched)


def phase_lock(stft_in: np.ndarray, hop: int = 512) -> np.ndarray:
    n_bins, n_frames = stft_in.shape
    locked = stft_in.copy()
    peaks = np.zeros(n_bins, dtype=bool)
    for t in range(n_frames):
        mag = np.abs(stft_in[:, t])
        for k in range(1, n_bins - 1):
            if mag[k] > mag[k - 1] and mag[k] > mag[k + 1]:
                peaks[k] = True
            else:
                peaks[k] = False
        peak_phase = np.angle(stft_in[:, t])
        for k in range(n_bins):
            if not peaks[k]:
                nearest = np.argmin(np.abs(np.where(peaks)[0] - k)) if np.any(peaks) else k
                if np.any(peaks):
                    pk = np.where(peaks)[0][nearest]
                    locked[k, t] = mag[k] * np.exp(1j * peak_phase[pk])
    return locked


def transient_preservation_mask(magnitude: np.ndarray,
                                threshold: float = 2.0) -> np.ndarray:
    flux = np.sum(np.maximum(np.diff(magnitude, axis=1), 0), axis=0)
    median_flux = np.median(flux)
    return flux > threshold * median_flux


def robotisation(stft_in: np.ndarray) -> np.ndarray:
    return np.abs(stft_in).astype(complex)


def whisperisation(stft_in: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, stft_in.shape)
    return np.abs(stft_in) * np.exp(1j * phases)
