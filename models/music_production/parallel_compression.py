"""Parallel compression: New York compression and multiband dynamics.

Implements parallel (New York) compression, multiband dynamics processing,
transient-sustain separation for mix bus processing, and loudness-aware
makeup gain.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CompressorParams:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    knee_db: float = 6.0
    makeup_db: float = 0.0


@dataclass
class MultibandConfig:
    crossover_hz: List[float]
    params_per_band: List[CompressorParams]


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(lin: float) -> float:
    return 20.0 * np.log10(max(abs(lin), 1e-10))


def envelope_follower(signal: np.ndarray, attack_samples: int,
                      release_samples: int) -> np.ndarray:
    env = np.zeros(len(signal))
    rect = np.abs(signal)
    for i in range(1, len(signal)):
        if rect[i] > env[i - 1]:
            coeff = 1.0 - np.exp(-1.0 / max(attack_samples, 1))
            env[i] = env[i - 1] + coeff * (rect[i] - env[i - 1])
        else:
            coeff = 1.0 - np.exp(-1.0 / max(release_samples, 1))
            env[i] = env[i - 1] + coeff * (rect[i] - env[i - 1])
    return env


def soft_knee_gain(input_db: float, threshold_db: float,
                   ratio: float, knee_db: float) -> float:
    half_knee = knee_db / 2.0
    if input_db < threshold_db - half_knee:
        return 0.0
    elif input_db > threshold_db + half_knee:
        return (threshold_db + (input_db - threshold_db) / ratio) - input_db
    else:
        x = input_db - threshold_db + half_knee
        return (1.0 / ratio - 1.0) * (x ** 2) / (2.0 * knee_db)


def compress(signal: np.ndarray, sr: int,
             params: CompressorParams) -> np.ndarray:
    attack_s = params.attack_ms / 1000.0 * sr
    release_s = params.release_ms / 1000.0 * sr
    env = envelope_follower(signal, int(attack_s), int(release_s))
    env_db = 20.0 * np.log10(env + 1e-10)
    gain_db = np.array([soft_knee_gain(e, params.threshold_db, params.ratio, params.knee_db)
                        for e in env_db])
    gain_linear = 10.0 ** (gain_db / 20.0)
    makeup = db_to_linear(params.makeup_db)
    return signal * gain_linear * makeup


def parallel_compress(dry: np.ndarray, sr: int,
                      params: CompressorParams,
                      wet_mix: float = 0.5) -> np.ndarray:
    wet = compress(dry, sr, params)
    return dry * (1.0 - wet_mix) + wet * wet_mix


def linkwitz_riley_split(signal: np.ndarray, sr: int,
                         crossover_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    wc = 2.0 * np.pi * crossover_hz / sr
    alpha = np.sin(wc) / (2.0 * 0.707)
    b0 = (1.0 - np.cos(wc)) / 2.0
    b1 = 1.0 - np.cos(wc)
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(wc)
    a2 = 1.0 - alpha
    low = _biquad(signal, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    low = _biquad(low, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    high = signal - low
    return low, high


def _biquad(x: np.ndarray, b0: float, b1: float, b2: float,
            a1: float, a2: float) -> np.ndarray:
    y = np.zeros_like(x)
    for i in range(2, len(x)):
        y[i] = b0 * x[i] + b1 * x[i - 1] + b2 * x[i - 2] - a1 * y[i - 1] - a2 * y[i - 2]
    return y


def multiband_compress(signal: np.ndarray, sr: int,
                       config: MultibandConfig) -> np.ndarray:
    bands = []
    remaining = signal.copy()
    for xover in config.crossover_hz:
        low, remaining = linkwitz_riley_split(remaining, sr, xover)
        bands.append(low)
    bands.append(remaining)
    n_params = len(config.params_per_band)
    result = np.zeros_like(signal)
    for i, band in enumerate(bands):
        p = config.params_per_band[min(i, n_params - 1)]
        result += compress(band, sr, p)
    return result


def transient_sustain_split(signal: np.ndarray, sr: int,
                            attack_ms: float = 5.0,
                            release_ms: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    fast_env = envelope_follower(signal, int(attack_ms / 1000 * sr),
                                 int(release_ms / 1000 * sr))
    slow_env = envelope_follower(signal, int(50.0 / 1000 * sr),
                                 int(300.0 / 1000 * sr))
    transient_mask = np.clip(fast_env / (slow_env + 1e-10) - 1.0, 0, 1)
    transient = signal * transient_mask
    sustain = signal * (1.0 - transient_mask)
    return transient, sustain


def loudness_aware_makeup(signal: np.ndarray, target_lufs: float = -14.0,
                          sr: int = 44100, block_s: float = 0.4) -> Tuple[np.ndarray, float]:
    block_len = int(block_s * sr)
    n_blocks = max(1, len(signal) // block_len)
    block_loudness = []
    for i in range(n_blocks):
        block = signal[i * block_len:(i + 1) * block_len]
        rms = np.sqrt(np.mean(block ** 2))
        lufs_approx = -0.691 + 20 * np.log10(max(rms, 1e-10))
        block_loudness.append(lufs_approx)
    gated = [l for l in block_loudness if l > -70]
    if not gated:
        return signal, 0.0
    relative_threshold = np.mean(gated) - 10
    final_blocks = [l for l in gated if l > relative_threshold]
    integrated = float(np.mean(final_blocks)) if final_blocks else -70.0
    gain_db = target_lufs - integrated
    gain_linear = db_to_linear(gain_db)
    return signal * gain_linear, gain_db


def gain_reduction_meter(signal: np.ndarray, sr: int,
                         params: CompressorParams,
                         hop_ms: float = 10.0) -> np.ndarray:
    hop = int(hop_ms / 1000 * sr)
    n_frames = max(1, len(signal) // hop)
    gr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + hop]
        rms_db = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-10)
        gr[i] = soft_knee_gain(rms_db, params.threshold_db, params.ratio, params.knee_db)
    return gr
