"""Dynamic range analysis and compression curve modeling."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DynamicRangeStats:
    """Dynamic range statistics for an audio signal."""
    peak_db: float
    rms_db: float
    crest_factor_db: float
    dynamic_range_db: float
    loudness_range_db: float

    @property
    def is_heavily_compressed(self) -> bool:
        return self.crest_factor_db < 6.0


def amplitude_to_db(amplitude: float) -> float:
    """Convert linear amplitude to dB (ref=1.0)."""
    if amplitude <= 0:
        return -120.0
    return 20.0 * math.log10(amplitude)


def db_to_amplitude(db: float) -> float:
    """Convert dB to linear amplitude."""
    return 10.0 ** (db / 20.0)


def rms_level(signal: np.ndarray) -> float:
    """RMS level in linear amplitude."""
    return float(np.sqrt(np.mean(signal ** 2)))


def peak_level(signal: np.ndarray) -> float:
    """Peak level in linear amplitude."""
    return float(np.max(np.abs(signal)))


def analyze_dynamic_range(signal: np.ndarray, frame_size: int = 4096) -> DynamicRangeStats:
    """Compute dynamic range statistics for a signal."""
    pk = peak_level(signal)
    rms = rms_level(signal)

    peak_db = amplitude_to_db(pk)
    rms_db = amplitude_to_db(rms)
    crest_db = peak_db - rms_db

    # Frame-wise RMS for loudness range
    n_frames = max(1, len(signal) // frame_size)
    frame_rms = []
    for i in range(n_frames):
        frame = signal[i * frame_size:(i + 1) * frame_size]
        fr = rms_level(frame)
        if fr > 0:
            frame_rms.append(amplitude_to_db(fr))

    if len(frame_rms) >= 2:
        sorted_rms = sorted(frame_rms)
        n = len(sorted_rms)
        # Loudness range: 10th to 95th percentile
        lo = sorted_rms[max(0, int(n * 0.10))]
        hi = sorted_rms[min(n - 1, int(n * 0.95))]
        loudness_range = hi - lo
        dynamic_range = sorted_rms[-1] - sorted_rms[0]
    else:
        loudness_range = 0.0
        dynamic_range = 0.0

    return DynamicRangeStats(
        peak_db=peak_db,
        rms_db=rms_db,
        crest_factor_db=crest_db,
        dynamic_range_db=dynamic_range,
        loudness_range_db=loudness_range,
    )


@dataclass
class CompressorSettings:
    """Parameters for a dynamic range compressor."""
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    knee_db: float = 0.0
    makeup_gain_db: float = 0.0


def compression_curve(input_db: float, settings: CompressorSettings) -> float:
    """Static compression curve: input dB -> output dB."""
    t = settings.threshold_db
    r = max(settings.ratio, 1.0)
    k = settings.knee_db

    if k > 0 and abs(input_db - t) < k / 2:
        # Soft knee region
        x = input_db - t + k / 2
        output = input_db + (1 / r - 1) * (x ** 2) / (2 * k)
    elif input_db > t:
        output = t + (input_db - t) / r
    else:
        output = input_db

    return output + settings.makeup_gain_db


def gain_reduction(input_db: float, settings: CompressorSettings) -> float:
    """Gain reduction in dB (always <= 0)."""
    return compression_curve(input_db, settings) - input_db - settings.makeup_gain_db


def apply_compression(
    signal: np.ndarray,
    settings: CompressorSettings,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Apply simple static compression to a signal (frame-based)."""
    frame_size = max(1, int(sample_rate * settings.attack_ms / 1000))
    output = np.copy(signal).astype(np.float64)
    n_frames = max(1, len(signal) // frame_size)

    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(signal))
        frame = signal[start:end]
        pk = peak_level(frame)
        if pk <= 0:
            continue
        input_db = amplitude_to_db(pk)
        output_db = compression_curve(input_db, settings)
        gain_linear = db_to_amplitude(output_db - input_db)
        output[start:end] = frame * gain_linear

    return output


def limiter_ceiling(signal: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Hard-limit a signal to a ceiling level."""
    ceiling_linear = db_to_amplitude(ceiling_db)
    pk = peak_level(signal)
    if pk <= ceiling_linear or pk <= 0:
        return signal.copy()
    return signal * (ceiling_linear / pk)


def loudness_normalize(signal: np.ndarray, target_rms_db: float = -14.0) -> np.ndarray:
    """Normalize signal to a target RMS level."""
    current_rms = rms_level(signal)
    if current_rms <= 0:
        return signal.copy()
    current_db = amplitude_to_db(current_rms)
    gain_db = target_rms_db - current_db
    return signal * db_to_amplitude(gain_db)


def multiband_split_points(n_bands: int = 3) -> list[float]:
    """Suggested crossover frequencies for multiband compression."""
    if n_bands <= 1:
        return []
    if n_bands == 2:
        return [2000.0]
    if n_bands == 3:
        return [200.0, 2000.0]
    if n_bands == 4:
        return [100.0, 1000.0, 8000.0]
    # Log-spaced for higher band counts
    return list(np.logspace(np.log10(80), np.log10(16000), n_bands - 1))
