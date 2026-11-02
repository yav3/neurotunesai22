"""Audio quality metrics: SNR, THD, clipping, silence, dynamic range.

Provides signal-level quality assessment: signal-to-noise ratio, total harmonic
distortion, clipping detection, silence detection, and loudness normalization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioQualityMetrics:
    snr_db: float
    thd_percent: float
    dynamic_range_db: float
    crest_factor_db: float
    dc_offset: float
    clipping_ratio: float
    silence_ratio: float


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB given separate signal and noise arrays."""
    sig_power = float(np.mean(signal ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power < 1e-12:
        return 120.0
    return 10.0 * math.log10(sig_power / noise_power)


def estimate_snr(
    signal: np.ndarray, sr: float,
    noise_floor_percentile: float = 5.0,
) -> float:
    """Estimate SNR from signal alone using percentile-based noise floor."""
    frame_len = int(sr * 0.02)
    if frame_len < 1 or len(signal) < frame_len:
        return 0.0
    n_frames = len(signal) // frame_len
    frame_powers = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * frame_len:(i + 1) * frame_len]
        frame_powers[i] = float(np.mean(frame ** 2))
    noise_floor = float(np.percentile(frame_powers, noise_floor_percentile))
    signal_power = float(np.mean(frame_powers))
    if noise_floor < 1e-12:
        return 120.0
    return 10.0 * math.log10(max(signal_power, 1e-12) / noise_floor)


def total_harmonic_distortion(
    signal: np.ndarray, fundamental_hz: float, sr: float,
) -> float:
    """THD as percentage: ratio of harmonic power to fundamental power."""
    n = len(signal)
    if n < 2 or fundamental_hz <= 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    f0_idx = int(np.argmin(np.abs(freqs - fundamental_hz)))
    bw = max(1, int(fundamental_hz * 0.05 / (sr / n)))
    fund_power = float(np.sum(spectrum[max(0, f0_idx - bw):f0_idx + bw + 1] ** 2))
    harm_power = 0.0
    for h in range(2, 11):
        h_freq = fundamental_hz * h
        if h_freq > sr / 2:
            break
        h_idx = int(np.argmin(np.abs(freqs - h_freq)))
        harm_power += float(np.sum(spectrum[max(0, h_idx - bw):h_idx + bw + 1] ** 2))
    if fund_power < 1e-12:
        return 0.0
    return 100.0 * math.sqrt(harm_power / fund_power)


def detect_clipping(
    signal: np.ndarray, threshold: float = 0.99,
) -> tuple[float, list[int]]:
    """Detect clipping: returns (clipping_ratio, list of clipped sample indices)."""
    abs_signal = np.abs(signal)
    clipped_mask = abs_signal >= threshold
    indices = list(np.where(clipped_mask)[0])
    ratio = len(indices) / max(len(signal), 1)
    return ratio, indices


def detect_silence(
    signal: np.ndarray,
    threshold_db: float = -60.0,
    min_duration_s: float = 0.1,
    sr: float = 44100.0,
) -> list[tuple[int, int]]:
    """Find silent regions: returns list of (start_sample, end_sample)."""
    threshold_amp = 10.0 ** (threshold_db / 20.0)
    min_samples = int(min_duration_s * sr)
    abs_signal = np.abs(signal)
    silent = abs_signal < threshold_amp
    regions = []
    start = None
    for i in range(len(silent)):
        if silent[i] and start is None:
            start = i
        elif not silent[i] and start is not None:
            if i - start >= min_samples:
                regions.append((start, i))
            start = None
    if start is not None and len(signal) - start >= min_samples:
        regions.append((start, len(signal)))
    return regions


def compute_audio_quality(
    signal: np.ndarray, sr: float,
) -> AudioQualityMetrics:
    """Full audio quality assessment."""
    snr = estimate_snr(signal, sr)
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak_db = 20.0 * math.log10(max(peak, 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    crest = peak_db - rms_db
    frame_len = int(sr * 0.05)
    if frame_len < 1:
        frame_len = 1
    n_frames = max(1, len(signal) // frame_len)
    frame_rms = np.zeros(n_frames)
    for i in range(n_frames):
        f = signal[i * frame_len:(i + 1) * frame_len]
        frame_rms[i] = 20.0 * math.log10(max(float(np.sqrt(np.mean(f ** 2))), 1e-12))
    dyn_range = float(np.percentile(frame_rms, 95) - np.percentile(frame_rms, 5))
    dc = float(np.mean(signal))
    clip_ratio, _ = detect_clipping(signal)
    silence_regions = detect_silence(signal, sr=sr)
    silence_samples = sum(e - s for s, e in silence_regions)
    silence_ratio = silence_samples / max(len(signal), 1)
    return AudioQualityMetrics(
        snr_db=snr, thd_percent=0.0, dynamic_range_db=dyn_range,
        crest_factor_db=crest, dc_offset=dc,
        clipping_ratio=clip_ratio, silence_ratio=silence_ratio,
    )


def loudness_normalize(
    signal: np.ndarray, target_lufs: float = -14.0, sr: float = 44100.0,
) -> np.ndarray:
    """Normalize signal to target integrated loudness (simplified LUFS)."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    current_db = 20.0 * math.log10(max(rms, 1e-12))
    current_lufs = current_db - 0.691
    gain_db = target_lufs - current_lufs
    gain_linear = 10.0 ** (gain_db / 20.0)
    return signal * gain_linear
