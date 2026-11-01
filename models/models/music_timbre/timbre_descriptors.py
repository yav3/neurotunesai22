"""Timbre descriptors: spectral shape, temporal envelope, and perceptual timbre features.

Implements MPEG-7 inspired audio descriptors and perceptual timbre attributes
for instrument identification, sound design, and music information retrieval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimbreDescriptors:
    brightness: float
    warmth: float
    roughness: float
    sharpness: float
    fullness: float
    attack_time_ms: float
    spectral_centroid_hz: float
    spectral_spread_hz: float
    spectral_irregularity: float
    odd_even_ratio: float
    inharmonicity: float
    temporal_centroid: float


@dataclass(frozen=True)
class AttackDescriptor:
    onset_time: float
    attack_time: float
    peak_time: float
    peak_amplitude: float
    log_attack_time: float
    attack_slope: float


@dataclass(frozen=True)
class SpectralEnvelope:
    frequencies: np.ndarray
    magnitudes: np.ndarray
    formant_freqs: np.ndarray
    formant_bw: np.ndarray
    spectral_tilt_db_per_oct: float


def brightness(spectrum: np.ndarray, freqs: np.ndarray, cutoff_hz: float = 1500.0) -> float:
    """Ratio of energy above cutoff to total energy — perceptual brightness."""
    total = np.sum(spectrum ** 2)
    if total < 1e-12:
        return 0.0
    above = np.sum(spectrum[freqs >= cutoff_hz] ** 2)
    return float(above / total)


def warmth(spectrum: np.ndarray, freqs: np.ndarray,
           low: float = 100.0, high: float = 500.0) -> float:
    """Energy concentration in the low-mid band (100–500 Hz)."""
    total = np.sum(spectrum ** 2)
    if total < 1e-12:
        return 0.0
    band = np.sum(spectrum[(freqs >= low) & (freqs <= high)] ** 2)
    return float(band / total)


def spectral_irregularity(magnitudes: np.ndarray) -> float:
    """Jensen spectral irregularity: sum of squared differences between adjacent partials."""
    if len(magnitudes) < 2:
        return 0.0
    diffs = np.diff(magnitudes)
    return float(np.sum(diffs ** 2) / np.sum(magnitudes ** 2 + 1e-12))


def odd_even_ratio(magnitudes: np.ndarray) -> float:
    """Ratio of odd to even harmonic energy (fundamental = harmonic 1 = odd)."""
    if len(magnitudes) < 2:
        return 1.0
    odd_energy = np.sum(magnitudes[0::2] ** 2)
    even_energy = np.sum(magnitudes[1::2] ** 2)
    if even_energy < 1e-12:
        return float('inf') if odd_energy > 1e-12 else 1.0
    return float(odd_energy / even_energy)


def inharmonicity(partials_hz: np.ndarray, fundamental_hz: float) -> float:
    """Average deviation of partials from perfect harmonic series.

    Returns mean |f_n - n*f0| / f0 over all partials.
    """
    if fundamental_hz < 1e-6 or len(partials_hz) == 0:
        return 0.0
    harmonics = np.arange(1, len(partials_hz) + 1) * fundamental_hz
    deviations = np.abs(partials_hz - harmonics) / fundamental_hz
    return float(np.mean(deviations))


def log_attack_time(signal: np.ndarray, sr: float, threshold: float = 0.1) -> AttackDescriptor:
    """Compute log-attack-time and attack characteristics from a signal onset."""
    env = np.abs(signal)
    if len(env) == 0 or np.max(env) < 1e-12:
        return AttackDescriptor(0.0, 0.0, 0.0, 0.0, float('-inf'), 0.0)
    peak_idx = int(np.argmax(env))
    peak_amp = float(env[peak_idx])
    onset_level = threshold * peak_amp
    onset_idx = 0
    for i in range(peak_idx):
        if env[i] >= onset_level:
            onset_idx = i
            break
    onset_t = onset_idx / sr
    peak_t = peak_idx / sr
    attack_t = peak_t - onset_t
    lat = math.log10(max(attack_t, 1e-6))
    slope = peak_amp / max(attack_t, 1e-6)
    return AttackDescriptor(
        onset_time=onset_t, attack_time=attack_t, peak_time=peak_t,
        peak_amplitude=peak_amp, log_attack_time=lat, attack_slope=slope,
    )


def temporal_centroid(signal: np.ndarray, sr: float) -> float:
    """Temporal centroid (center of gravity) of the energy envelope, in seconds."""
    env = signal ** 2
    total = np.sum(env)
    if total < 1e-12:
        return 0.0
    times = np.arange(len(signal)) / sr
    return float(np.sum(times * env) / total)


def spectral_tilt(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """Spectral tilt in dB/octave via log-frequency linear regression."""
    mask = (spectrum > 1e-12) & (freqs > 0)
    if np.sum(mask) < 2:
        return 0.0
    log_f = np.log2(freqs[mask])
    log_mag = 20.0 * np.log10(spectrum[mask])
    n = len(log_f)
    sx = np.sum(log_f)
    sy = np.sum(log_mag)
    sxx = np.sum(log_f ** 2)
    sxy = np.sum(log_f * log_mag)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return float((n * sxy - sx * sy) / denom)


def roughness_sethares(partials_hz: np.ndarray, amplitudes: np.ndarray) -> float:
    """Sensory roughness via Sethares (1993) dissonance model.

    Sum of pairwise roughness r(f_i, f_j) = a_i*a_j*(e^{-b1*s} - e^{-b2*s})
    where s = |f_i - f_j| / min(f_i, f_j).
    """
    b1, b2 = 3.5, 5.75
    n = len(partials_hz)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            f_min = min(partials_hz[i], partials_hz[j])
            if f_min < 1.0:
                continue
            s = abs(partials_hz[i] - partials_hz[j]) / f_min
            total += amplitudes[i] * amplitudes[j] * (math.exp(-b1 * s) - math.exp(-b2 * s))
    return float(total)


def sharpness_zwicker(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """Simplified Zwicker sharpness: weighted spectral centroid emphasizing high frequencies.

    Uses a weighting function g(z) that increases above 16 Bark.
    """
    bark = 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)
    weights = np.where(bark <= 16.0, 1.0, 0.066 * np.exp(0.171 * bark))
    loudness = spectrum ** 2
    total = np.sum(loudness)
    if total < 1e-12:
        return 0.0
    return float(np.sum(weights * loudness * bark) / total)


def estimate_spectral_envelope(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    n_formants: int = 4,
) -> SpectralEnvelope:
    """Estimate spectral envelope with formant frequencies via peak picking."""
    log_spec = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    formant_freqs = []
    formant_bw = []
    if len(log_spec) > 2:
        for i in range(1, len(log_spec) - 1):
            if log_spec[i] > log_spec[i - 1] and log_spec[i] > log_spec[i + 1]:
                formant_freqs.append(freqs[i])
                left = right = i
                threshold = log_spec[i] - 3.0
                while left > 0 and log_spec[left] > threshold:
                    left -= 1
                while right < len(log_spec) - 1 and log_spec[right] > threshold:
                    right += 1
                formant_bw.append(freqs[min(right, len(freqs) - 1)] - freqs[max(left, 0)])
                if len(formant_freqs) >= n_formants:
                    break
    tilt = spectral_tilt(spectrum, freqs)
    return SpectralEnvelope(
        frequencies=freqs,
        magnitudes=spectrum,
        formant_freqs=np.array(formant_freqs, dtype=np.float64),
        formant_bw=np.array(formant_bw, dtype=np.float64),
        spectral_tilt_db_per_oct=tilt,
    )


def compute_timbre_descriptors(
    signal: np.ndarray,
    sr: float,
    spectrum: np.ndarray,
    freqs: np.ndarray,
    partials_hz: np.ndarray | None = None,
    partial_amps: np.ndarray | None = None,
    fundamental_hz: float | None = None,
) -> TimbreDescriptors:
    """Compute a full set of timbre descriptors for a sound."""
    b = brightness(spectrum, freqs)
    w = warmth(spectrum, freqs)
    sc = float(np.sum(freqs * spectrum ** 2) / max(np.sum(spectrum ** 2), 1e-12))
    ss = float(np.sqrt(np.sum((freqs - sc) ** 2 * spectrum ** 2) / max(np.sum(spectrum ** 2), 1e-12)))
    if partials_hz is not None and partial_amps is not None:
        r = roughness_sethares(partials_hz, partial_amps)
        si = spectral_irregularity(partial_amps)
        oer = odd_even_ratio(partial_amps)
        inh = inharmonicity(partials_hz, fundamental_hz or 0.0)
    else:
        r = 0.0
        si = spectral_irregularity(spectrum[:64])
        oer = 1.0
        inh = 0.0
    s = sharpness_zwicker(spectrum, freqs)
    f = 1.0 - b
    attack = log_attack_time(signal, sr)
    tc = temporal_centroid(signal, sr)
    return TimbreDescriptors(
        brightness=b, warmth=w, roughness=r, sharpness=s, fullness=f,
        attack_time_ms=attack.attack_time * 1000.0,
        spectral_centroid_hz=sc, spectral_spread_hz=ss,
        spectral_irregularity=si, odd_even_ratio=oer,
        inharmonicity=inh, temporal_centroid=tc,
    )
