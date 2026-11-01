"""Dynamics processing for musical audio: compression, limiting, gating, loudness."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CompressorParams:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    knee_db: float = 6.0
    makeup_db: float = 0.0


def compute_gain_reduction(level_db: float, params: CompressorParams) -> float:
    t = params.threshold_db
    r = params.ratio
    k = params.knee_db
    if r <= 1.0:
        return 0.0
    if k <= 0:
        if level_db <= t:
            return 0.0
        return -(level_db - t) * (1 - 1 / r)
    low = t - k / 2
    high = t + k / 2
    if level_db <= low:
        return 0.0
    if level_db >= high:
        return -(level_db - t) * (1 - 1 / r)
    x = level_db - low
    return -(1 - 1 / r) * x ** 2 / (2 * k)


def envelope_follower(
    signal: np.ndarray, attack_ms: float, release_ms: float, sr: float,
) -> np.ndarray:
    att = math.exp(-1.0 / (attack_ms * sr / 1000)) if attack_ms > 0 else 0.0
    rel = math.exp(-1.0 / (release_ms * sr / 1000)) if release_ms > 0 else 0.0
    env = np.zeros(len(signal))
    level = 0.0
    for i in range(len(signal)):
        inp = abs(signal[i])
        if inp > level:
            level = att * level + (1 - att) * inp
        else:
            level = rel * level + (1 - rel) * inp
        env[i] = level
    return env


def compressor(
    signal: np.ndarray, params: CompressorParams, sr: float,
) -> np.ndarray:
    env = envelope_follower(signal, params.attack_ms, params.release_ms, sr)
    env_db = 20 * np.log10(np.maximum(env, 1e-10))
    gain_db = np.array([compute_gain_reduction(l, params) + params.makeup_db for l in env_db])
    gain_linear = 10 ** (gain_db / 20)
    return signal * gain_linear


def limiter(
    signal: np.ndarray, threshold_db: float, sr: float, lookahead_ms: float = 5.0,
) -> np.ndarray:
    params = CompressorParams(
        threshold_db=threshold_db, ratio=100.0,
        attack_ms=0.1, release_ms=50.0, knee_db=0.0,
    )
    lookahead_samples = int(lookahead_ms * sr / 1000)
    if lookahead_samples > 0:
        padded = np.pad(signal, (lookahead_samples, 0))
        result = compressor(padded, params, sr)
        return result[lookahead_samples:]
    return compressor(signal, params, sr)


def gate(
    signal: np.ndarray, threshold_db: float, sr: float,
    attack_ms: float = 1.0, release_ms: float = 50.0,
) -> np.ndarray:
    env = envelope_follower(signal, attack_ms, release_ms, sr)
    env_db = 20 * np.log10(np.maximum(env, 1e-10))
    gain = np.where(env_db >= threshold_db, 1.0, 0.0)
    return signal * gain


def rms_level(signal: np.ndarray, frame_size: int = 1024) -> np.ndarray:
    n_frames = max(1, len(signal) // frame_size)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * frame_size: (i + 1) * frame_size]
        rms[i] = math.sqrt(np.mean(frame ** 2))
    return rms


def peak_level(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return -math.inf
    return 20 * math.log10(max(np.max(np.abs(signal)), 1e-10))


def lufs_momentary(signal: np.ndarray, sr: float) -> float:
    if len(signal) == 0:
        return -70.0
    from .filters import biquad_design, apply_biquad, FilterType
    pre = biquad_design(FilterType.HIGHSHELF, 1500.0, sr, q=0.7, gain_db=4.0)
    hp = biquad_design(FilterType.HIGHPASS, 38.0, sr, q=0.5)
    filtered = apply_biquad(signal, pre)
    filtered = apply_biquad(filtered, hp)
    mean_sq = float(np.mean(filtered ** 2))
    if mean_sq <= 0:
        return -70.0
    return -0.691 + 10 * math.log10(mean_sq)


def crest_factor(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return 0.0
    rms = math.sqrt(np.mean(signal ** 2))
    if rms < 1e-10:
        return 0.0
    return float(np.max(np.abs(signal)) / rms)


def dynamic_range(signal: np.ndarray, sr: float, frame_ms: float = 50.0) -> float:
    frame_size = max(1, int(frame_ms * sr / 1000))
    levels = rms_level(signal, frame_size)
    levels_db = 20 * np.log10(np.maximum(levels, 1e-10))
    active = levels_db[levels_db > -60]
    if len(active) < 2:
        return 0.0
    return float(np.percentile(active, 95) - np.percentile(active, 5))
