"""Perceptual features: psychoacoustically-motivated audio descriptors.

Extract features grounded in human hearing: loudness (ISO 532-1 approximation),
sharpness (Zwicker/Fastl), roughness (amplitude modulation depth), fluctuation
strength, and sensory dissonance (Plomp-Levelt).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PerceptualSnapshot:
    loudness_sone: float
    sharpness_acum: float
    roughness_asper: float
    fluctuation_vacil: float
    dissonance: float


def bark_scale(f_hz: np.ndarray) -> np.ndarray:
    return 13.0 * np.arctan(0.00076 * f_hz) + 3.5 * np.arctan((f_hz / 7500.0) ** 2)


def critical_bandwidth(fc_hz: float) -> float:
    return 25.0 + 75.0 * (1.0 + 1.4 * (fc_hz / 1000.0) ** 2) ** 0.69


def specific_loudness(spectrum_db: np.ndarray, freqs_hz: np.ndarray,
                      n_bands: int = 24) -> np.ndarray:
    bark_edges = np.linspace(0, 24, n_bands + 1)
    bark_vals = bark_scale(freqs_hz)
    specific = np.zeros(n_bands)
    for i in range(n_bands):
        mask = (bark_vals >= bark_edges[i]) & (bark_vals < bark_edges[i + 1])
        if np.any(mask):
            band_db = spectrum_db[mask]
            excitation = 10.0 ** (np.max(band_db) / 10.0)
            specific[i] = excitation ** 0.23
    return specific


def total_loudness(specific: np.ndarray) -> float:
    return float(np.sum(specific))


def sharpness_zwicker(specific: np.ndarray) -> float:
    n = len(specific)
    z = np.linspace(0.5, n - 0.5, n)
    g = np.where(z <= 16, 1.0, 0.066 * np.exp(0.171 * z))
    N_total = np.sum(specific)
    if N_total < 1e-10:
        return 0.0
    return 0.11 * float(np.sum(specific * g * z)) / N_total


def roughness_from_modulation(signal: np.ndarray, sr: int,
                              frame_size: int = 2048) -> float:
    n_frames = max(1, len(signal) // frame_size)
    roughness_vals = []
    for i in range(n_frames):
        frame = signal[i * frame_size:(i + 1) * frame_size]
        if len(frame) < frame_size:
            break
        analytic = np.abs(_hilbert_envelope(frame))
        env_spectrum = np.abs(np.fft.rfft(analytic - np.mean(analytic)))
        freqs = np.fft.rfftfreq(len(analytic), 1.0 / sr)
        mod_band = (freqs >= 15) & (freqs <= 300)
        if np.any(mod_band):
            mod_depth = np.max(env_spectrum[mod_band]) / max(np.mean(analytic), 1e-10)
            fc_weight = np.where((freqs >= 60) & (freqs <= 80), 1.0, 0.5)
            roughness_vals.append(float(mod_depth * np.mean(fc_weight[mod_band])))
    return float(np.mean(roughness_vals)) if roughness_vals else 0.0


def _hilbert_envelope(x: np.ndarray) -> np.ndarray:
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    h[0] = 1
    if N % 2 == 0:
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[1:(N + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h))


def fluctuation_strength(signal: np.ndarray, sr: int,
                         frame_size: int = 8192) -> float:
    n_frames = max(1, len(signal) // frame_size)
    fluct_vals = []
    for i in range(n_frames):
        frame = signal[i * frame_size:(i + 1) * frame_size]
        if len(frame) < frame_size:
            break
        env = _hilbert_envelope(frame)
        env_spectrum = np.abs(np.fft.rfft(env - np.mean(env)))
        freqs = np.fft.rfftfreq(len(env), 1.0 / sr)
        mod_band = (freqs >= 0.5) & (freqs <= 20)
        if np.any(mod_band):
            peak_mod = np.max(env_spectrum[mod_band])
            peak_freq = freqs[mod_band][np.argmax(env_spectrum[mod_band])]
            weighting = 1.0 / (peak_freq / 4.0 + 4.0 / max(peak_freq, 0.1))
            fluct_vals.append(float(peak_mod * weighting))
    return float(np.mean(fluct_vals)) if fluct_vals else 0.0


def plomp_levelt_dissonance(freqs_hz: np.ndarray,
                            amplitudes: np.ndarray) -> float:
    n = len(freqs_hz)
    if n < 2:
        return 0.0
    total_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            f_low = min(freqs_hz[i], freqs_hz[j])
            f_high = max(freqs_hz[i], freqs_hz[j])
            s = 0.24 / (0.0207 * f_low + 18.96)
            x = s * (f_high - f_low)
            d = np.exp(-3.5 * x) - np.exp(-5.75 * x)
            total_d += d * amplitudes[i] * amplitudes[j]
    return float(total_d)


def spectral_dissonance_curve(spectrum: np.ndarray, freqs_hz: np.ndarray,
                              top_k: int = 20) -> np.ndarray:
    peak_idx = np.argsort(spectrum)[-top_k:]
    peak_freqs = freqs_hz[peak_idx]
    peak_amps = spectrum[peak_idx]
    peak_amps = peak_amps / max(peak_amps.max(), 1e-10)
    ratios = np.linspace(1.0, 2.0, 100)
    curve = np.zeros(len(ratios))
    for ri, r in enumerate(ratios):
        shifted = peak_freqs * r
        all_f = np.concatenate([peak_freqs, shifted])
        all_a = np.concatenate([peak_amps, peak_amps * 0.8])
        curve[ri] = plomp_levelt_dissonance(all_f, all_a)
    return curve


def perceptual_snapshot(signal: np.ndarray, sr: int,
                        n_fft: int = 4096) -> PerceptualSnapshot:
    spectrum = np.abs(np.fft.rfft(signal[:n_fft])) if len(signal) >= n_fft else np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(min(len(signal), n_fft), 1.0 / sr)
    spectrum_db = 20 * np.log10(spectrum + 1e-10)
    spec_loudness = specific_loudness(spectrum_db, freqs)
    loud = total_loudness(spec_loudness)
    sharp = sharpness_zwicker(spec_loudness)
    rough = roughness_from_modulation(signal, sr)
    fluct = fluctuation_strength(signal, sr)
    peak_idx = np.argsort(spectrum)[-15:]
    diss = plomp_levelt_dissonance(freqs[peak_idx], spectrum[peak_idx] / max(spectrum.max(), 1e-10))
    return PerceptualSnapshot(
        loudness_sone=loud,
        sharpness_acum=sharp,
        roughness_asper=rough,
        fluctuation_vacil=fluct,
        dissonance=diss,
    )
