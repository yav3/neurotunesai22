"""Fingerprint robustness: degradation functions and evaluation framework."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def add_white_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    sig_power = np.mean(signal ** 2)
    if sig_power < 1e-20:
        return signal.copy()
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.default_rng(0).normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def time_stretch(signal: np.ndarray, rate: float = 1.0) -> np.ndarray:
    if rate <= 0 or abs(rate - 1.0) < 1e-10:
        return signal.copy()
    n = len(signal)
    new_n = int(n / rate)
    if new_n < 2:
        return signal[:1].copy()
    old_indices = np.linspace(0, n - 1, new_n)
    return np.interp(old_indices, np.arange(n), signal)


def pitch_shift_semitones(
    signal: np.ndarray, semitones: float, sr: int = 22050,
) -> np.ndarray:
    rate = 2.0 ** (semitones / 12.0)
    stretched = time_stretch(signal, rate)
    n = len(signal)
    if len(stretched) >= n:
        return stretched[:n]
    padded = np.zeros(n)
    padded[: len(stretched)] = stretched
    return padded


def apply_lowpass(
    signal: np.ndarray, cutoff_hz: float = 4000.0, sr: int = 22050,
) -> np.ndarray:
    if len(signal) < 4:
        return signal.copy()
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    mask = np.where(freqs <= cutoff_hz, 1.0, 0.0)
    rolloff = np.where(
        (freqs > cutoff_hz) & (freqs < cutoff_hz * 1.5),
        np.exp(-((freqs - cutoff_hz) / (cutoff_hz * 0.1)) ** 2),
        mask,
    )
    return np.fft.irfft(spectrum * rolloff, n=len(signal))


def mp3_simulation(signal: np.ndarray, quality: float = 0.5) -> np.ndarray:
    bits = max(4, int(16 * quality))
    levels = 2 ** bits
    quantized = np.round(signal * levels) / levels
    cutoff = 4000.0 + 16000.0 * quality
    return apply_lowpass(quantized, cutoff)


@dataclass(frozen=True)
class DegradationResult:
    name: str
    score: float
    passed: bool


def evaluate_robustness(
    original: np.ndarray,
    fingerprint_fn: Callable[[np.ndarray], object],
    match_fn: Callable[[object, object], float],
    threshold: float = 0.5,
) -> list[DegradationResult]:
    original_fp = fingerprint_fn(original)
    degradations: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("noise_30dB", lambda s: add_white_noise(s, 30.0)),
        ("noise_20dB", lambda s: add_white_noise(s, 20.0)),
        ("noise_10dB", lambda s: add_white_noise(s, 10.0)),
        ("stretch_0.95", lambda s: time_stretch(s, 0.95)),
        ("stretch_1.05", lambda s: time_stretch(s, 1.05)),
        ("pitch_+1", lambda s: pitch_shift_semitones(s, 1.0)),
        ("pitch_-1", lambda s: pitch_shift_semitones(s, -1.0)),
        ("lowpass_8k", lambda s: apply_lowpass(s, 8000.0)),
        ("lowpass_4k", lambda s: apply_lowpass(s, 4000.0)),
        ("mp3_high", lambda s: mp3_simulation(s, 0.8)),
        ("mp3_low", lambda s: mp3_simulation(s, 0.3)),
    ]
    results = []
    for name, degrade in degradations:
        degraded = degrade(original)
        degraded_fp = fingerprint_fn(degraded)
        score = match_fn(original_fp, degraded_fp)
        results.append(DegradationResult(name, score, score >= threshold))
    return results
