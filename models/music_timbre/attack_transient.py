"""Onset transient analysis — attack characterization and classification."""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum


class AttackType(Enum):
    PERCUSSIVE = "percussive"
    PLUCKED = "plucked"
    BOWED = "bowed"
    BLOWN = "blown"
    SOFT = "soft"


@dataclass
class TransientDescriptor:
    attack_time_ms: float
    peak_amplitude: float
    rise_slope: float  # amplitude / ms
    spectral_centroid_hz: float
    attack_type: AttackType
    noise_ratio: float  # ratio of noise energy in attack


def compute_attack_time(envelope: np.ndarray, sr: float, threshold: float = 0.1) -> float:
    """Time (ms) from onset threshold crossing to peak amplitude."""
    peak_idx = np.argmax(envelope)
    peak_val = envelope[peak_idx]
    if peak_val <= 0:
        return 0.0
    onset_level = peak_val * threshold
    onset_idx = 0
    for i in range(peak_idx):
        if envelope[i] >= onset_level:
            onset_idx = i
            break
    return (peak_idx - onset_idx) / sr * 1000.0


def rise_slope(envelope: np.ndarray, sr: float, threshold: float = 0.1) -> float:
    """Slope of attack rise (amplitude / ms)."""
    attack_ms = compute_attack_time(envelope, sr, threshold)
    if attack_ms <= 0:
        return 0.0
    peak = float(np.max(envelope))
    onset_level = peak * threshold
    return (peak - onset_level) / attack_ms


def spectral_centroid_at_onset(signal: np.ndarray, sr: float,
                                onset_samples: int = 512) -> float:
    """Spectral centroid of the first onset_samples of the signal."""
    seg = signal[:onset_samples]
    if len(seg) < 4:
        return 0.0
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(seg)) / len(seg)))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    total = np.sum(spectrum)
    if total < 1e-30:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def noise_ratio_at_onset(signal: np.ndarray, sr: float,
                          onset_samples: int = 512, n_harmonics: int = 10) -> float:
    """Ratio of non-harmonic to total energy in the attack segment."""
    seg = signal[:onset_samples]
    if len(seg) < 16:
        return 0.0
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    total_energy = np.sum(spectrum ** 2)
    if total_energy < 1e-30:
        return 0.0
    peak_idx = np.argmax(spectrum[1:]) + 1
    f0 = freqs[peak_idx]
    if f0 < 20:
        return 1.0
    harmonic_energy = 0.0
    for h in range(1, n_harmonics + 1):
        target = h * f0
        closest = np.argmin(np.abs(freqs - target))
        low = max(0, closest - 1)
        high = min(len(spectrum), closest + 2)
        harmonic_energy += np.sum(spectrum[low:high] ** 2)
    return float(1.0 - min(1.0, harmonic_energy / total_energy))


def classify_attack(attack_ms: float, noise: float) -> AttackType:
    """Heuristic classification based on attack time and noise content."""
    if attack_ms < 5.0:
        return AttackType.PERCUSSIVE
    elif attack_ms < 20.0:
        if noise > 0.5:
            return AttackType.PERCUSSIVE
        return AttackType.PLUCKED
    elif attack_ms < 80.0:
        if noise > 0.4:
            return AttackType.BLOWN
        return AttackType.BOWED
    else:
        return AttackType.SOFT


def analyze_transient(signal: np.ndarray, sr: float,
                       onset_samples: int = 512) -> TransientDescriptor:
    """Full transient analysis of a signal's attack."""
    env = np.abs(signal)
    kernel_size = max(1, int(sr * 0.002))
    if kernel_size > 1 and len(env) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        env = np.convolve(env, kernel, mode='same')
    atk_ms = compute_attack_time(env, sr)
    slope = rise_slope(env, sr)
    centroid = spectral_centroid_at_onset(signal, sr, onset_samples)
    noise = noise_ratio_at_onset(signal, sr, onset_samples)
    atype = classify_attack(atk_ms, noise)
    return TransientDescriptor(
        attack_time_ms=atk_ms,
        peak_amplitude=float(np.max(np.abs(signal))),
        rise_slope=slope,
        spectral_centroid_hz=centroid,
        attack_type=atype,
        noise_ratio=noise,
    )
