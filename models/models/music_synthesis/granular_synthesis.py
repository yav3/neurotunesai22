"""Granular synthesis engine for time-stretching, pitch-shifting, and sound design."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class GrainParams:
    duration_ms: float = 50.0
    overlap: float = 0.5
    position_jitter_ms: float = 5.0
    pitch_jitter_cents: float = 0.0
    amplitude_jitter: float = 0.0
    window_type: str = "hann"


def _make_window(n: int, window_type: str = "hann") -> np.ndarray:
    if window_type == "hamming":
        return np.hamming(n)
    elif window_type == "gaussian":
        t = np.linspace(-2, 2, n)
        return np.exp(-0.5 * t ** 2)
    elif window_type == "tukey":
        alpha = 0.5
        w = np.ones(n)
        ramp = int(alpha * n / 2)
        if ramp > 0:
            w[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
            w[-ramp:] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))[::-1]
        return w
    return np.hanning(n)


def generate_grain(source: np.ndarray, sr: float, position: float,
                   params: GrainParams, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    grain_samples = int(params.duration_ms * sr / 1000.0)
    if grain_samples < 2:
        grain_samples = 2
    jitter = rng.normal(0, params.position_jitter_ms * sr / 1000.0)
    pos = int(position * sr + jitter)
    pos = max(0, min(pos - grain_samples // 2, len(source) - grain_samples))
    grain = source[pos:pos + grain_samples].copy().astype(float)
    if len(grain) < grain_samples:
        grain = np.pad(grain, (0, grain_samples - len(grain)))
    window = _make_window(grain_samples, params.window_type)
    grain *= window
    if params.pitch_jitter_cents != 0:
        shift = rng.normal(0, params.pitch_jitter_cents)
        ratio = 2.0 ** (shift / 1200.0)
        new_len = int(len(grain) / ratio)
        if new_len > 1:
            grain = np.interp(np.linspace(0, len(grain) - 1, new_len),
                              np.arange(len(grain)), grain)
            grain = np.pad(grain, (0, max(0, grain_samples - len(grain))))[:grain_samples]
    if params.amplitude_jitter > 0:
        grain *= 1.0 + rng.normal(0, params.amplitude_jitter)
    return grain


def grain_cloud(source: np.ndarray, sr: float, duration: float,
                params: GrainParams, density: float = 1.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    out_samples = int(duration * sr)
    output = np.zeros(out_samples)
    grain_samples = int(params.duration_ms * sr / 1000.0)
    hop = max(1, int(grain_samples * (1.0 - params.overlap) / density))
    src_duration = len(source) / sr
    t = 0
    while t < out_samples:
        position = rng.uniform(0, src_duration)
        grain = generate_grain(source, sr, position, params, rng)
        end = min(t + len(grain), out_samples)
        output[t:end] += grain[:end - t]
        t += hop
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output /= peak
    return output


def time_stretch_granular(source: np.ndarray, sr: float, stretch_factor: float,
                          params: GrainParams | None = None) -> np.ndarray:
    if params is None:
        params = GrainParams()
    duration = len(source) / sr * stretch_factor
    grain_samples = int(params.duration_ms * sr / 1000.0)
    hop = max(1, int(grain_samples * (1.0 - params.overlap)))
    out_samples = int(duration * sr)
    output = np.zeros(out_samples)
    src_pos = 0.0
    t = 0
    while t < out_samples:
        idx = int(src_pos)
        idx = max(0, min(idx, len(source) - grain_samples))
        grain = source[idx:idx + grain_samples].copy().astype(float)
        if len(grain) < grain_samples:
            grain = np.pad(grain, (0, grain_samples - len(grain)))
        grain *= _make_window(grain_samples, params.window_type)
        end = min(t + grain_samples, out_samples)
        output[t:end] += grain[:end - t]
        t += hop
        src_pos += hop / stretch_factor
    return output


def pitch_shift_granular(source: np.ndarray, sr: float, semitones: float,
                         params: GrainParams | None = None) -> np.ndarray:
    if params is None:
        params = GrainParams()
    ratio = 2.0 ** (semitones / 12.0)
    stretched = time_stretch_granular(source, sr, ratio, params)
    n_out = len(source)
    indices = np.linspace(0, len(stretched) - 1, n_out)
    return np.interp(indices, np.arange(len(stretched)), stretched)


def granular_freeze(source: np.ndarray, sr: float, position: float,
                    duration: float, params: GrainParams | None = None,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    if params is None:
        params = GrainParams(position_jitter_ms=2.0)
    if rng is None:
        rng = np.random.default_rng(42)
    out_samples = int(duration * sr)
    grain_samples = int(params.duration_ms * sr / 1000.0)
    hop = max(1, int(grain_samples * (1.0 - params.overlap)))
    output = np.zeros(out_samples)
    t = 0
    while t < out_samples:
        grain = generate_grain(source, sr, position, params, rng)
        end = min(t + len(grain), out_samples)
        output[t:end] += grain[:end - t]
        t += hop
    return output
