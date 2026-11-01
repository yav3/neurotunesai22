"""Spectrogram computation: STFT, mel filterbank, CQT, chromagram, onset strength,
spectral flux, and spectral contrast.  Pure numpy/scipy — returns arrays, no plotting."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SpectrogramConfig:
    n_fft: int = 2048
    hop_length: int = 512
    window: str = "hann"
    fmin: float = 20.0
    fmax: float = 20000.0
    n_mels: int = 128
    ref_db: float = 1.0
    top_db: float = 80.0


def _window(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    return np.ones(n)


def stft_magnitude(signal: np.ndarray, config: Optional[SpectrogramConfig] = None) -> np.ndarray:
    cfg = config or SpectrogramConfig()
    win = _window(cfg.window, cfg.n_fft)
    n_frames = 1 + (len(signal) - cfg.n_fft) // cfg.hop_length
    if n_frames <= 0:
        return np.zeros((cfg.n_fft // 2 + 1, 0))
    out = np.zeros((cfg.n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * cfg.hop_length
        frame = signal[start : start + cfg.n_fft] * win
        spectrum = np.fft.rfft(frame)
        out[:, i] = np.abs(spectrum)
    return out


def amplitude_to_db(S: np.ndarray, ref: float = 1.0, top_db: float = 80.0) -> np.ndarray:
    magnitude = np.maximum(S, 1e-10)
    db = 20.0 * np.log10(magnitude / max(ref, 1e-10))
    db_max = db.max() if db.size > 0 else 0.0
    return np.maximum(db, db_max - top_db)


def hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: float, n_fft: int, n_mels: int = 128,
                   fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    n_bins = n_fft // 2 + 1
    mel_lo = hz_to_mel(fmin)
    mel_hi = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = np.array([mel_to_hz(m) for m in mel_pts])
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        lo, center, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        for k in range(lo, center):
            if center > lo:
                fb[m, k] = (k - lo) / (center - lo)
        for k in range(center, hi):
            if hi > center:
                fb[m, k] = (hi - k) / (hi - center)
    return fb


def mel_spectrogram(signal: np.ndarray, sr: float,
                    config: Optional[SpectrogramConfig] = None) -> np.ndarray:
    cfg = config or SpectrogramConfig()
    S = stft_magnitude(signal, cfg)
    fb = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, min(cfg.fmax, sr / 2))
    return fb @ S


def cqt_kernel(sr: float, fmin: float = 32.7, n_bins: int = 84,
               bins_per_octave: int = 12) -> np.ndarray:
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    max_len = int(np.ceil(Q * sr / fmin))
    kernel = np.zeros((n_bins, max_len), dtype=complex)
    for k in range(n_bins):
        N_k = int(np.ceil(Q * sr / freqs[k]))
        if N_k > max_len:
            N_k = max_len
        t = np.arange(N_k)
        win = np.hamming(N_k)
        kernel[k, :N_k] = (win / N_k) * np.exp(2j * np.pi * freqs[k] * t / sr)
    return kernel


def constant_q_transform(signal: np.ndarray, sr: float, fmin: float = 32.7,
                         n_bins: int = 84, bins_per_octave: int = 12) -> np.ndarray:
    K = cqt_kernel(sr, fmin, n_bins, bins_per_octave)
    klen = K.shape[1]
    hop = max(klen // 4, 1)
    n_frames = max(1, 1 + (len(signal) - klen) // hop)
    out = np.zeros((n_bins, n_frames))
    for i in range(n_frames):
        start = i * hop
        end = start + klen
        if end > len(signal):
            seg = np.zeros(klen)
            seg[: len(signal) - start] = signal[start:]
        else:
            seg = signal[start:end]
        out[:, i] = np.abs(K @ seg)
    return out


def chromagram(signal: np.ndarray, sr: float, n_fft: int = 4096,
               hop_length: int = 512) -> np.ndarray:
    cfg = SpectrogramConfig(n_fft=n_fft, hop_length=hop_length)
    S = stft_magnitude(signal, cfg)
    n_bins = S.shape[0]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    chroma = np.zeros((12, S.shape[1]))
    for i, f in enumerate(freqs):
        if f < 20:
            continue
        midi = 69 + 12 * np.log2(f / 440.0)
        pc = int(round(midi)) % 12
        chroma[pc] += S[i]
    return chroma


def spectral_flux(S: np.ndarray) -> np.ndarray:
    if S.shape[1] < 2:
        return np.zeros(max(S.shape[1], 1))
    diff = np.diff(S, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.concatenate([[0.0], flux])


def onset_strength(S_db: np.ndarray) -> np.ndarray:
    if S_db.shape[1] < 2:
        return np.zeros(max(S_db.shape[1], 1))
    diff = np.diff(S_db, axis=1)
    pos_diff = np.maximum(diff, 0)
    env = np.sum(pos_diff, axis=0)
    return np.concatenate([[0.0], env])


def spectral_contrast(S: np.ndarray, n_bands: int = 6) -> np.ndarray:
    n_freq, n_frames = S.shape
    band_edges = np.logspace(np.log10(1), np.log10(n_freq), n_bands + 1).astype(int)
    band_edges = np.clip(band_edges, 0, n_freq)
    contrast = np.zeros((n_bands, n_frames))
    for b in range(n_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        if hi <= lo:
            continue
        band = S[lo:hi, :]
        sorted_band = np.sort(band, axis=0)
        alpha = max(1, (hi - lo) // 5)
        peaks = np.mean(sorted_band[-alpha:, :], axis=0)
        valleys = np.mean(sorted_band[:alpha, :], axis=0)
        contrast[b] = peaks - valleys
    return contrast
