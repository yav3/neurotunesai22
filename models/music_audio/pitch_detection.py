"""Pitch detection: autocorrelation, YIN, parabolic interpolation, harmonic
product spectrum, pitch confidence, and voiced/unvoiced classification.  Pure numpy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PitchResult:
    frequency_hz: float
    confidence: float
    midi_note: float
    is_voiced: bool


def hz_to_midi(f: float) -> float:
    if f <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(f / 440.0)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def autocorrelation(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    fft_signal = np.fft.rfft(signal, fft_size)
    acf = np.fft.irfft(fft_signal * np.conj(fft_signal))
    return acf[:n]


def autocorrelation_pitch(
    signal: np.ndarray, sr: float,
    fmin: float = 50.0, fmax: float = 2000.0,
) -> PitchResult:
    acf = autocorrelation(signal)
    min_lag = max(1, int(sr / fmax))
    max_lag = min(len(acf) - 1, int(sr / fmin))
    if min_lag >= max_lag:
        return PitchResult(0.0, 0.0, 0.0, False)
    segment = acf[min_lag:max_lag + 1]
    peak_idx = np.argmax(segment) + min_lag
    if acf[0] < 1e-12:
        return PitchResult(0.0, 0.0, 0.0, False)
    confidence = float(acf[peak_idx] / acf[0])
    freq = sr / peak_idx
    return PitchResult(freq, confidence, hz_to_midi(freq), confidence > 0.5)


def _parabolic_interpolation(values: np.ndarray, idx: int) -> Tuple[float, float]:
    if idx <= 0 or idx >= len(values) - 1:
        return float(idx), float(values[idx])
    a, b, c = values[idx - 1], values[idx], values[idx + 1]
    denom = 2.0 * (2.0 * b - a - c)
    if abs(denom) < 1e-12:
        return float(idx), float(b)
    shift = (a - c) / denom
    refined_idx = idx + shift
    refined_val = b - 0.25 * (a - c) * shift
    return refined_idx, refined_val


def _cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    n = len(d)
    cmnd = np.ones(n)
    running_sum = 0.0
    for tau in range(1, n):
        running_sum += d[tau]
        if running_sum < 1e-12:
            cmnd[tau] = 1.0
        else:
            cmnd[tau] = d[tau] * tau / running_sum
    return cmnd


def yin_pitch(
    signal: np.ndarray, sr: float,
    fmin: float = 50.0, fmax: float = 2000.0,
    threshold: float = 0.15,
) -> PitchResult:
    n = len(signal)
    max_lag = min(n // 2, int(sr / fmin))
    min_lag = max(1, int(sr / fmax))
    d = np.zeros(max_lag)
    for tau in range(1, max_lag):
        diff = signal[:n - tau] - signal[tau:n]
        d[tau] = np.sum(diff ** 2)
    cmnd = _cumulative_mean_normalized_difference(d)
    best_tau = -1
    for tau in range(min_lag, max_lag):
        if cmnd[tau] < threshold:
            if tau + 1 < max_lag and cmnd[tau + 1] < cmnd[tau]:
                continue
            best_tau = tau
            break
    if best_tau < 1:
        if min_lag < max_lag:
            best_tau = int(np.argmin(cmnd[min_lag:max_lag])) + min_lag
        else:
            return PitchResult(0.0, 0.0, 0.0, False)
    refined_tau, _ = _parabolic_interpolation(cmnd, best_tau)
    if refined_tau < 1e-6:
        return PitchResult(0.0, 0.0, 0.0, False)
    freq = sr / refined_tau
    confidence = float(1.0 - cmnd[best_tau])
    return PitchResult(freq, max(0.0, confidence), hz_to_midi(freq), confidence > 0.5)


def harmonic_product_spectrum(
    signal: np.ndarray, sr: float,
    n_harmonics: int = 5,
    fmin: float = 50.0, fmax: float = 2000.0,
) -> PitchResult:
    n_fft = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    hps = spectrum.copy()
    for h in range(2, n_harmonics + 1):
        decimated = spectrum[::h]
        hps[:len(decimated)] *= decimated
    min_bin = max(1, int(fmin * n_fft / sr))
    max_bin = min(len(hps) - 1, int(fmax * n_fft / sr))
    if min_bin >= max_bin:
        return PitchResult(0.0, 0.0, 0.0, False)
    segment = hps[min_bin:max_bin + 1]
    if segment.max() < 1e-12:
        return PitchResult(0.0, 0.0, 0.0, False)
    peak_bin = np.argmax(segment) + min_bin
    refined_bin, _ = _parabolic_interpolation(hps, peak_bin)
    freq = refined_bin * sr / n_fft
    peak_val = hps[peak_bin]
    median_val = np.median(hps[min_bin:max_bin + 1])
    confidence = float(peak_val / (median_val + 1e-12))
    confidence = min(1.0, confidence / 10.0)
    return PitchResult(freq, confidence, hz_to_midi(freq), confidence > 0.3)


def pitch_track(
    signal: np.ndarray, sr: float,
    frame_size: int = 2048, hop_size: int = 512,
    method: str = "yin",
    fmin: float = 50.0, fmax: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = max(1, 1 + (len(signal) - frame_size) // hop_size)
    frequencies = np.zeros(n_frames)
    confidences = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        if method == "yin":
            result = yin_pitch(frame, sr, fmin, fmax)
        elif method == "autocorrelation":
            result = autocorrelation_pitch(frame, sr, fmin, fmax)
        elif method == "hps":
            result = harmonic_product_spectrum(frame, sr, fmin=fmin, fmax=fmax)
        else:
            result = yin_pitch(frame, sr, fmin, fmax)
        frequencies[i] = result.frequency_hz
        confidences[i] = result.confidence
        voiced[i] = result.is_voiced
    return frequencies, confidences, voiced


def median_pitch_filter(frequencies: np.ndarray, window: int = 5) -> np.ndarray:
    if len(frequencies) < window:
        return frequencies.copy()
    result = frequencies.copy()
    half = window // 2
    for i in range(half, len(frequencies) - half):
        seg = frequencies[i - half:i + half + 1]
        nonzero = seg[seg > 0]
        if len(nonzero) > 0:
            result[i] = np.median(nonzero)
    return result


def voiced_unvoiced_decision(
    signal: np.ndarray, sr: float,
    zcr_threshold: float = 0.1, energy_threshold: float = 0.01,
) -> bool:
    energy = float(np.mean(signal ** 2))
    if energy < energy_threshold:
        return False
    signs = np.sign(signal)
    zcr = float(np.sum(np.abs(np.diff(signs)) > 0) / len(signal))
    return zcr < zcr_threshold


def pitch_stability(frequencies: np.ndarray) -> float:
    valid = frequencies[frequencies > 0]
    if len(valid) < 2:
        return 1.0
    cents_diff = 1200.0 * np.abs(np.diff(np.log2(valid + 1e-12)))
    return float(max(0.0, 1.0 - np.mean(cents_diff) / 100.0))


def vibrato_detection(
    frequencies: np.ndarray, sr_frames: float,
    min_rate: float = 4.0, max_rate: float = 8.0,
) -> Tuple[float, float]:
    valid = frequencies[frequencies > 0]
    if len(valid) < 8:
        return 0.0, 0.0
    mean_f = np.mean(valid)
    centered = valid - mean_f
    fft_vals = np.abs(np.fft.rfft(centered))
    fft_freqs = np.fft.rfftfreq(len(centered), 1.0 / sr_frames)
    mask = (fft_freqs >= min_rate) & (fft_freqs <= max_rate)
    if not np.any(mask):
        return 0.0, 0.0
    peak_idx = np.argmax(fft_vals[mask])
    vibrato_rate = float(fft_freqs[mask][peak_idx])
    vibrato_depth_hz = float(fft_vals[mask][peak_idx]) * 2.0 / len(centered)
    vibrato_depth_cents = 1200.0 * np.log2((mean_f + vibrato_depth_hz) / mean_f) if mean_f > 0 else 0.0
    return vibrato_rate, float(vibrato_depth_cents)


def pitch_histogram(
    frequencies: np.ndarray, n_bins: int = 128,
    fmin: float = 50.0, fmax: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = frequencies[(frequencies >= fmin) & (frequencies <= fmax)]
    if len(valid) == 0:
        return np.zeros(n_bins), np.linspace(fmin, fmax, n_bins + 1)
    bins = np.linspace(fmin, fmax, n_bins + 1)
    hist, _ = np.histogram(valid, bins=bins)
    return hist.astype(float), bins
