"""Texture generation: algorithmic ambient/drone/granular texture synthesis parameters."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class TextureType(str, Enum):
    DRONE = "drone"
    GRANULAR = "granular"
    ADDITIVE = "additive"
    NOISE_SCULPTED = "noise_sculpted"
    SPECTRAL_FREEZE = "spectral_freeze"


@dataclass
class GrainParams:
    duration_ms: float = 50.0
    density_hz: float = 20.0
    pitch_spread_cents: float = 0.0
    amplitude_jitter: float = 0.0
    pan_spread: float = 0.0
    window_type: str = "hann"


@dataclass
class DroneParams:
    fundamental_hz: float = 60.0
    n_partials: int = 12
    partial_decay: float = 0.85
    beating_rate_hz: float = 0.5
    detune_cents: float = 3.0
    amplitude: float = 0.5


@dataclass
class TextureLayer:
    texture_type: TextureType
    params: Dict[str, float] = field(default_factory=dict)
    gain_db: float = 0.0
    pan: float = 0.0


@dataclass
class TextureScore:
    layers: List[TextureLayer] = field(default_factory=list)
    duration_s: float = 60.0
    master_gain_db: float = -6.0


def generate_drone_partials(params: DroneParams, sr: int = 44100, duration_s: float = 1.0) -> np.ndarray:
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples)
    for k in range(1, params.n_partials + 1):
        amp = params.amplitude * (params.partial_decay ** (k - 1))
        freq = params.fundamental_hz * k
        detune = params.detune_cents * (k - 1) / 1200.0
        freq_detuned = freq * (2.0 ** detune)
        beating = 1.0 + 0.1 * np.sin(2.0 * np.pi * params.beating_rate_hz * t * k * 0.1)
        signal += amp * beating * np.sin(2.0 * np.pi * freq_detuned * t)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return signal


def grain_envelope(n_samples: int, window_type: str = "hann") -> np.ndarray:
    if window_type == "hann":
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_samples) / max(n_samples - 1, 1)))
    if window_type == "triangle":
        mid = n_samples // 2
        env = np.concatenate([np.linspace(0, 1, mid + 1), np.linspace(1, 0, n_samples - mid - 1)])
        return env[:n_samples]
    return np.ones(n_samples)


def generate_granular_texture(
    source: np.ndarray, params: GrainParams, sr: int = 44100, duration_s: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_out = int(sr * duration_s)
    output = np.zeros(n_out)
    grain_samples = max(1, int(params.duration_ms * sr / 1000.0))
    interval = max(1, int(sr / max(params.density_hz, 0.1)))
    envelope = grain_envelope(grain_samples, params.window_type)
    n_grains = n_out // interval + 1
    for i in range(n_grains):
        start_out = i * interval
        if start_out >= n_out:
            break
        src_start = rng.integers(0, max(1, len(source) - grain_samples))
        grain = source[src_start:src_start + grain_samples].copy()
        if len(grain) < grain_samples:
            grain = np.pad(grain, (0, grain_samples - len(grain)))
        if params.pitch_spread_cents > 0:
            shift = rng.normal(0, params.pitch_spread_cents / 100.0)
            ratio = 2.0 ** (shift / 12.0)
            indices = np.linspace(0, len(grain) - 1, int(len(grain) / max(ratio, 0.1)))
            indices = np.clip(indices.astype(int), 0, len(grain) - 1)
            grain = grain[indices]
        grain = grain[:grain_samples]
        if len(grain) < grain_samples:
            grain = np.pad(grain, (0, grain_samples - len(grain)))
        amp = 1.0 + rng.uniform(-params.amplitude_jitter, params.amplitude_jitter)
        grain *= envelope * amp
        end_out = min(start_out + grain_samples, n_out)
        output[start_out:end_out] += grain[:end_out - start_out]
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak
    return output


def noise_sculpt(
    n_samples: int, sr: int = 44100,
    center_hz: float = 1000.0, bandwidth_hz: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    spectrum = np.fft.rfft(noise)
    mask = np.exp(-0.5 * ((freqs - center_hz) / max(bandwidth_hz, 1.0)) ** 2)
    spectrum *= mask
    result = np.fft.irfft(spectrum, n=n_samples)
    peak = np.max(np.abs(result))
    if peak > 0:
        result /= peak
    return result


def spectral_freeze(signal: np.ndarray, frame_idx: int = 0, n_fft: int = 2048) -> np.ndarray:
    if len(signal) < n_fft:
        padded = np.pad(signal, (0, n_fft - len(signal)))
    else:
        start = frame_idx * (n_fft // 2)
        start = min(start, len(signal) - n_fft)
        start = max(start, 0)
        padded = signal[start:start + n_fft]
    spectrum = np.fft.rfft(padded)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    frozen = magnitude * np.exp(1j * phase)
    return np.fft.irfft(frozen, n=n_fft)


def additive_texture(
    partials: List[Tuple[float, float]],
    sr: int = 44100, duration_s: float = 1.0,
    vibrato_hz: float = 0.0, vibrato_depth_cents: float = 0.0,
) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    signal = np.zeros(n)
    for freq, amp in partials:
        if vibrato_hz > 0 and vibrato_depth_cents > 0:
            mod = vibrato_depth_cents / 1200.0 * np.sin(2.0 * np.pi * vibrato_hz * t)
            inst_freq = freq * (2.0 ** mod)
            phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
        else:
            phase = 2.0 * np.pi * freq * t
        signal += amp * np.sin(phase)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return signal


def layer_mix(layers: List[Tuple[np.ndarray, float]], target_length: int) -> np.ndarray:
    output = np.zeros(target_length)
    for signal, gain_db in layers:
        gain = 10.0 ** (gain_db / 20.0)
        n = min(len(signal), target_length)
        output[:n] += signal[:n] * gain
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak
    return output


def texture_density(layers: List[TextureLayer]) -> float:
    if not layers:
        return 0.0
    total_gain = sum(10.0 ** (l.gain_db / 20.0) for l in layers)
    return min(1.0, total_gain / len(layers))


def evolve_parameter(
    start: float, end: float, n_steps: int, curve: str = "linear",
) -> np.ndarray:
    t = np.linspace(0, 1, n_steps)
    if curve == "exponential":
        t = t ** 2
    elif curve == "logarithmic":
        t = np.sqrt(t)
    elif curve == "sigmoid":
        t = 1.0 / (1.0 + np.exp(-10.0 * (t - 0.5)))
    return start + (end - start) * t
