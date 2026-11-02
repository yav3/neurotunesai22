"""Sidechain compression and ducking models.

RMS envelope followers, sidechain gain curves, pump-effect modelling,
and frequency-selective ducking for mix engineering.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DetectorType(Enum):
    PEAK = "peak"
    RMS = "rms"
    RMS_WEIGHTED = "rms_weighted"


@dataclass
class CompressorParams:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 5.0
    release_ms: float = 50.0
    knee_db: float = 6.0
    makeup_gain_db: float = 0.0


def rms_envelope(signal: np.ndarray, window_ms: float = 10.0, sr: int = 44100) -> np.ndarray:
    """Compute RMS envelope with sliding window."""
    win_samples = max(1, int(sr * window_ms / 1000.0))
    squared = signal ** 2
    kernel = np.ones(win_samples) / win_samples
    env = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return env


def amplitude_to_db(amplitude: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """Convert amplitude to decibels, floor at -120 dB."""
    safe = np.maximum(np.abs(amplitude), 1e-6 * ref)
    return 20.0 * np.log10(safe / ref)


def db_to_amplitude(db: np.ndarray) -> np.ndarray:
    return 10.0 ** (db / 20.0)


def gain_computer(
    input_db: np.ndarray,
    params: CompressorParams,
) -> np.ndarray:
    """Static gain curve computation with soft knee."""
    thresh = params.threshold_db
    ratio = params.ratio
    knee = params.knee_db
    half_knee = knee / 2.0

    output_db = np.copy(input_db)
    for i in range(len(input_db)):
        x = input_db[i]
        if x < thresh - half_knee:
            output_db[i] = x
        elif x > thresh + half_knee:
            output_db[i] = thresh + (x - thresh) / ratio
        else:
            # soft knee region
            delta = x - thresh + half_knee
            output_db[i] = x + (1.0 / ratio - 1.0) * delta ** 2 / (2.0 * knee)

    return output_db - input_db  # gain reduction in dB


def sidechain_duck(
    signal: np.ndarray,
    sidechain: np.ndarray,
    params: Optional[CompressorParams] = None,
    sr: int = 44100,
) -> np.ndarray:
    """Apply sidechain ducking: reduce *signal* when *sidechain* is loud."""
    if params is None:
        params = CompressorParams()

    n = min(len(signal), len(sidechain))
    signal = signal[:n]
    sidechain = sidechain[:n]

    env = rms_envelope(sidechain, window_ms=10.0, sr=sr)
    env_db = amplitude_to_db(env)

    gain_reduction_db = gain_computer(env_db, params)

    # ballistics (attack/release smoothing)
    attack_coeff = np.exp(-1.0 / (sr * params.attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * params.release_ms / 1000.0))

    smoothed = np.zeros_like(gain_reduction_db)
    smoothed[0] = gain_reduction_db[0]
    for i in range(1, len(smoothed)):
        if gain_reduction_db[i] < smoothed[i - 1]:
            smoothed[i] = attack_coeff * smoothed[i - 1] + (1 - attack_coeff) * gain_reduction_db[i]
        else:
            smoothed[i] = release_coeff * smoothed[i - 1] + (1 - release_coeff) * gain_reduction_db[i]

    gain_linear = db_to_amplitude(smoothed + params.makeup_gain_db)
    return signal * gain_linear


def pump_curve(
    bpm: float,
    duration_s: float = 1.0,
    sr: int = 44100,
    depth_db: float = -12.0,
    attack_fraction: float = 0.05,
    release_fraction: float = 0.7,
) -> np.ndarray:
    """Generate a rhythmic pump (sidechain ducking) envelope at given BPM."""
    beat_s = 60.0 / bpm
    n_total = int(duration_s * sr)
    n_beat = int(beat_s * sr)
    if n_beat < 2:
        return np.ones(n_total)

    n_attack = max(1, int(n_beat * attack_fraction))
    n_release = max(1, int(n_beat * release_fraction))
    n_hold = n_beat - n_attack - n_release
    if n_hold < 0:
        n_release = n_beat - n_attack
        n_hold = 0

    depth_linear = db_to_amplitude(np.array([depth_db]))[0]
    one_beat = np.concatenate([
        np.linspace(1.0, depth_linear, n_attack),
        np.full(max(0, n_hold), depth_linear),
        np.linspace(depth_linear, 1.0, n_release),
    ])

    full = np.tile(one_beat, n_total // len(one_beat) + 1)
    return full[:n_total]


def frequency_selective_duck(
    signal: np.ndarray,
    sidechain: np.ndarray,
    sr: int = 44100,
    low_hz: float = 20.0,
    high_hz: float = 200.0,
    params: Optional[CompressorParams] = None,
) -> np.ndarray:
    """Sidechain only a frequency band (e.g., bass ducking)."""
    from scipy.signal import butter, sosfilt

    if params is None:
        params = CompressorParams()

    n = min(len(signal), len(sidechain))
    signal = signal[:n].copy()
    sidechain = sidechain[:n]

    nyq = sr / 2.0
    lo = max(low_hz / nyq, 0.001)
    hi = min(high_hz / nyq, 0.999)
    sos_bp = butter(4, [lo, hi], btype="band", output="sos")
    sos_other_lo = butter(4, lo, btype="low", output="sos") if lo > 0.01 else None
    sos_other_hi = butter(4, hi, btype="high", output="sos")

    band = sosfilt(sos_bp, signal)
    ducked_band = sidechain_duck(band, sidechain, params, sr)

    high_part = sosfilt(sos_other_hi, signal)
    if sos_other_lo is not None:
        low_part = sosfilt(sos_other_lo, signal)
        return low_part + ducked_band + high_part
    return ducked_band + high_part
