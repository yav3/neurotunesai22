"""Attack-decay model: characterise temporal envelope shape of timbres.

Extract and parameterise ADSR (attack, decay, sustain, release) envelopes,
classify percussive vs sustained timbres, and compute envelope similarity
for instrument clustering and synthesis control.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ADSRParams:
    attack_ms: float
    decay_ms: float
    sustain_level: float
    release_ms: float


@dataclass
class EnvelopeFeatures:
    attack_time_ms: float
    decay_time_ms: float
    sustain_level: float
    release_time_ms: float
    log_attack_time: float
    temporal_centroid: float
    is_percussive: bool


def amplitude_envelope(signal: np.ndarray, hop: int = 256) -> np.ndarray:
    n_frames = len(signal) // hop
    env = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * hop:(i + 1) * hop]
        env[i] = np.sqrt(np.mean(frame ** 2))
    return env


def detect_attack(envelope: np.ndarray, threshold: float = 0.9) -> int:
    peak = np.max(envelope)
    if peak < 1e-10:
        return 0
    target = threshold * peak
    for i, v in enumerate(envelope):
        if v >= target:
            return i
    return len(envelope) - 1


def detect_decay_end(envelope: np.ndarray, attack_idx: int,
                     sustain_threshold: float = 0.1) -> int:
    peak = np.max(envelope)
    if peak < 1e-10:
        return attack_idx
    target = sustain_threshold * peak
    for i in range(attack_idx, len(envelope)):
        if envelope[i] <= target:
            return i
    return len(envelope) - 1


def estimate_sustain(envelope: np.ndarray, decay_end: int,
                     release_start: Optional[int] = None) -> float:
    end = release_start or int(len(envelope) * 0.8)
    if decay_end >= end:
        return 0.0
    segment = envelope[decay_end:end]
    peak = np.max(envelope)
    if peak < 1e-10:
        return 0.0
    return float(np.mean(segment) / peak)


def extract_adsr(signal: np.ndarray, sr: int,
                 hop: int = 256) -> ADSRParams:
    env = amplitude_envelope(signal, hop)
    hop_ms = hop / sr * 1000.0
    attack_idx = detect_attack(env)
    decay_end = detect_decay_end(env, attack_idx)
    sustain = estimate_sustain(env, decay_end)
    release_start = int(len(env) * 0.8)
    release_idx = len(env) - 1
    return ADSRParams(
        attack_ms=attack_idx * hop_ms,
        decay_ms=(decay_end - attack_idx) * hop_ms,
        sustain_level=sustain,
        release_ms=(release_idx - release_start) * hop_ms,
    )


def synthesise_adsr(params: ADSRParams, duration_ms: float,
                    sr: int = 44100) -> np.ndarray:
    n = int(duration_ms / 1000.0 * sr)
    env = np.zeros(n)
    a = int(params.attack_ms / 1000.0 * sr)
    d = int(params.decay_ms / 1000.0 * sr)
    r = int(params.release_ms / 1000.0 * sr)
    s_end = max(n - r, a + d)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0 and a + d <= n:
        env[a:a + d] = np.linspace(1, params.sustain_level, d)
    env[a + d:s_end] = params.sustain_level
    if r > 0 and s_end < n:
        env[s_end:n] = np.linspace(params.sustain_level, 0, n - s_end)
    return env


def temporal_centroid(envelope: np.ndarray) -> float:
    t = np.arange(len(envelope))
    total = np.sum(envelope)
    if total < 1e-10:
        return 0.0
    return float(np.sum(t * envelope) / total) / len(envelope)


def is_percussive(adsr: ADSRParams, threshold_ms: float = 50.0) -> bool:
    return adsr.attack_ms < threshold_ms and adsr.sustain_level < 0.2


def envelope_similarity(env_a: np.ndarray, env_b: np.ndarray) -> float:
    n = min(len(env_a), len(env_b))
    a = env_a[:n] / (np.max(env_a[:n]) + 1e-10)
    b = env_b[:n] / (np.max(env_b[:n]) + 1e-10)
    return float(1.0 - np.mean(np.abs(a - b)))


def classify_instrument_family(adsr: ADSRParams) -> str:
    if adsr.attack_ms < 10 and adsr.sustain_level < 0.1:
        return "percussion"
    if adsr.attack_ms < 30 and adsr.sustain_level > 0.5:
        return "plucked_string"
    if adsr.attack_ms > 100 and adsr.sustain_level > 0.6:
        return "bowed_string"
    if 20 < adsr.attack_ms < 80 and adsr.sustain_level > 0.4:
        return "wind"
    if adsr.attack_ms < 5:
        return "keyboard"
    return "unknown"
