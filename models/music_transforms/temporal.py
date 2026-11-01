"""Temporal feature extraction: onset strength, spectral flux, ZCR, RMS, autocorrelation."""
from __future__ import annotations

import numpy as np


def onset_strength(
    signal: np.ndarray, sr: int = 22050, hop_length: int = 512, n_fft: int = 2048,
) -> np.ndarray:
    from models.music_transforms.spectral import stft, mel_filterbank, power_spectrogram
    S = stft(signal, n_fft=n_fft, hop_length=hop_length)
    power = power_spectrogram(S)
    fb = mel_filterbank(n_mels=40, n_fft=n_fft, sr=sr)
    mel_power = fb @ power
    log_mel = np.log1p(mel_power * 1000)
    diff = np.diff(log_mel, axis=1)
    diff = np.maximum(diff, 0)
    onset = diff.mean(axis=0)
    return onset / (onset.max() + 1e-12)


def spectral_flux(spectrogram: np.ndarray) -> np.ndarray:
    diff = np.diff(spectrogram, axis=1)
    positive_diff = np.maximum(diff, 0)
    flux = np.sum(positive_diff, axis=0)
    return flux


def zero_crossing_rate(signal: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length]
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        zcr[i] = crossings / (2 * frame_length)
    return zcr


def rms_energy(signal: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


def autocorrelation(signal: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    n = len(signal)
    if max_lag is None:
        max_lag = n
    max_lag = min(max_lag, n)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    F = np.fft.rfft(signal, n=fft_size)
    acf_full = np.fft.irfft(F * np.conj(F), n=fft_size)[:max_lag]
    if acf_full[0] != 0:
        acf_full = acf_full / acf_full[0]
    return acf_full


def temporal_centroid(signal: np.ndarray, sr: int = 22050) -> float:
    energy = signal ** 2
    total = energy.sum()
    if total < 1e-12:
        return 0.0
    times = np.arange(len(signal)) / sr
    return float(np.sum(times * energy) / total)
