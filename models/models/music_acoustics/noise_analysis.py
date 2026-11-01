"""Noise analysis: spectral noise profiling, SNR estimation, noise gating.

Implements noise floor estimation, signal-to-noise ratio measurement,
spectral subtraction, and noise gate design for audio signals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NoiseProfile:
    mean_spectrum: np.ndarray
    std_spectrum: np.ndarray
    floor_db: float
    peak_frequency_hz: float
    spectral_flatness: float


def estimate_noise_floor(signal: np.ndarray, sr: float = 44100.0,
                         frame_size: int = 2048, hop: int = 512,
                         percentile: float = 10.0) -> float:
    if len(signal) < frame_size:
        rms = np.sqrt(np.mean(signal ** 2))
        return 20.0 * np.log10(rms + 1e-12)
    n_frames = (len(signal) - frame_size) // hop + 1
    rms_values = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))
    floor_rms = float(np.percentile(rms_values, percentile))
    return 20.0 * np.log10(floor_rms + 1e-12)


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power)


def compute_snr_from_floor(signal: np.ndarray, noise_floor_db: float) -> float:
    sig_rms_db = 20.0 * np.log10(np.sqrt(np.mean(signal ** 2)) + 1e-12)
    return sig_rms_db - noise_floor_db


def noise_profile(noise_segment: np.ndarray, sr: float = 44100.0,
                  n_fft: int = 2048) -> NoiseProfile:
    hop = n_fft // 2
    n_frames = max(1, (len(noise_segment) - n_fft) // hop + 1)
    window = np.hanning(n_fft)
    spectra = []
    for i in range(n_frames):
        start = i * hop
        frame = noise_segment[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag = np.abs(np.fft.rfft(frame * window))
        spectra.append(mag)
    spectra_arr = np.array(spectra)
    mean_spec = np.mean(spectra_arr, axis=0)
    std_spec = np.std(spectra_arr, axis=0)
    floor_db = 20.0 * np.log10(np.mean(mean_spec) + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    peak_idx = int(np.argmax(mean_spec))
    peak_freq = float(freqs[peak_idx])
    geo_mean = np.exp(np.mean(np.log(mean_spec + 1e-12)))
    arith_mean = np.mean(mean_spec) + 1e-12
    flatness = float(geo_mean / arith_mean)
    return NoiseProfile(mean_spec, std_spec, floor_db, peak_freq, flatness)


def spectral_subtraction(signal: np.ndarray, profile: NoiseProfile,
                         n_fft: int = 2048, alpha: float = 2.0,
                         beta: float = 0.01) -> np.ndarray:
    hop = n_fft // 2
    window = np.hanning(n_fft)
    n_frames = (len(signal) - n_fft) // hop + 1
    output = np.zeros(len(signal))
    window_sum = np.zeros(len(signal))
    noise_power = (profile.mean_spectrum ** 2) * alpha
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)
        n_bins = len(mag)
        noise_est = noise_power[:n_bins] if len(noise_power) >= n_bins else np.pad(noise_power, (0, n_bins - len(noise_power)))
        cleaned_mag = np.maximum(mag ** 2 - noise_est, beta * mag ** 2)
        cleaned_mag = np.sqrt(cleaned_mag)
        cleaned_spectrum = cleaned_mag * np.exp(1j * phase)
        cleaned_frame = np.fft.irfft(cleaned_spectrum, n=n_fft)
        output[start:start + n_fft] += cleaned_frame * window
        window_sum[start:start + n_fft] += window ** 2
    mask = window_sum > 1e-8
    output[mask] /= window_sum[mask]
    return output


@dataclass
class NoiseGate:
    threshold_db: float = -40.0
    attack_ms: float = 1.0
    release_ms: float = 50.0
    hold_ms: float = 10.0
    range_db: float = -80.0

    def process(self, signal: np.ndarray, sr: float = 44100.0) -> np.ndarray:
        attack_samples = max(1, int(self.attack_ms * sr / 1000))
        release_samples = max(1, int(self.release_ms * sr / 1000))
        hold_samples = max(1, int(self.hold_ms * sr / 1000))
        threshold_lin = 10.0 ** (self.threshold_db / 20.0)
        range_lin = 10.0 ** (self.range_db / 20.0)
        env = np.abs(signal)
        gain = np.ones(len(signal))
        gate_open = False
        hold_counter = 0
        current_gain = range_lin
        for i in range(len(signal)):
            if env[i] >= threshold_lin:
                gate_open = True
                hold_counter = hold_samples
                target = 1.0
            elif hold_counter > 0:
                hold_counter -= 1
                target = 1.0
            else:
                gate_open = False
                target = range_lin
            if target > current_gain:
                coeff = 1.0 - math.exp(-1.0 / attack_samples)
            else:
                coeff = 1.0 - math.exp(-1.0 / release_samples)
            current_gain += coeff * (target - current_gain)
            gain[i] = current_gain
        return signal * gain


def a_weighting_snr(signal: np.ndarray, noise: np.ndarray,
                    sr: float = 44100.0, n_fft: int = 2048) -> float:
    def a_weight(f: float) -> float:
        if f < 1.0:
            return 0.0
        f2 = f * f
        num = 12194.0 ** 2 * f2 ** 2
        den = (f2 + 20.6 ** 2) * math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2)
        if den < 1e-30:
            return 0.0
        ra = num / den
        return 20.0 * math.log10(ra + 1e-12) + 2.0

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    weights = np.array([10.0 ** (a_weight(f) / 20.0) for f in freqs])
    sig_spec = np.abs(np.fft.rfft(signal[:n_fft] if len(signal) >= n_fft else np.pad(signal, (0, n_fft - len(signal)))))
    noise_spec = np.abs(np.fft.rfft(noise[:n_fft] if len(noise) >= n_fft else np.pad(noise, (0, n_fft - len(noise)))))
    n_bins = min(len(weights), len(sig_spec), len(noise_spec))
    sig_weighted = sig_spec[:n_bins] * weights[:n_bins]
    noise_weighted = noise_spec[:n_bins] * weights[:n_bins]
    sig_power = np.sum(sig_weighted ** 2)
    noise_power = np.sum(noise_weighted ** 2)
    if noise_power < 1e-20:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power)
