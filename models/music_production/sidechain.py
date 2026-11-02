"""Sidechain compression and ducking for music production."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SidechainParams:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 1.0
    release_ms: float = 100.0
    makeup_gain_db: float = 0.0


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    if linear <= 0:
        return -120.0
    return 20.0 * np.log10(linear)


def envelope_follower(signal: np.ndarray, sr: float,
                       attack_ms: float, release_ms: float) -> np.ndarray:
    """Peak envelope follower with separate attack/release time constants."""
    attack_coeff = 1.0 - np.exp(-2.2 / (attack_ms * sr / 1000.0)) if attack_ms > 0 else 1.0
    release_coeff = 1.0 - np.exp(-2.2 / (release_ms * sr / 1000.0)) if release_ms > 0 else 1.0
    env = np.zeros(len(signal))
    level = 0.0
    for i in range(len(signal)):
        inp = abs(signal[i])
        if inp > level:
            level += attack_coeff * (inp - level)
        else:
            level += release_coeff * (inp - level)
        env[i] = level
    return env


def compute_gain_reduction(envelope_db: np.ndarray, threshold_db: float,
                            ratio: float) -> np.ndarray:
    """Compute gain reduction curve from envelope in dB."""
    gr = np.zeros_like(envelope_db)
    for i in range(len(envelope_db)):
        if envelope_db[i] > threshold_db:
            overshoot = envelope_db[i] - threshold_db
            gr[i] = overshoot * (1.0 - 1.0 / ratio)
    return gr


def apply_sidechain(target: np.ndarray, sidechain_signal: np.ndarray,
                     sr: float, params: SidechainParams) -> np.ndarray:
    """Apply sidechain compression to target using sidechain_signal as key input."""
    n = min(len(target), len(sidechain_signal))
    env = envelope_follower(sidechain_signal[:n], sr, params.attack_ms, params.release_ms)
    env_db = np.array([linear_to_db(float(e)) for e in env])
    gr = compute_gain_reduction(env_db, params.threshold_db, params.ratio)
    gain_linear = np.array([db_to_linear(-float(g)) for g in gr])
    makeup = db_to_linear(params.makeup_gain_db)
    output = target[:n].copy().astype(float)
    output *= gain_linear * makeup
    return output


def ducking_curve(sidechain_signal: np.ndarray, sr: float,
                   depth_db: float = -12.0, attack_ms: float = 5.0,
                   release_ms: float = 200.0, threshold_db: float = -30.0) -> np.ndarray:
    """Generate a ducking gain curve (linear) from a sidechain signal."""
    env = envelope_follower(sidechain_signal, sr, attack_ms, release_ms)
    threshold_lin = db_to_linear(threshold_db)
    depth_lin = db_to_linear(depth_db)
    curve = np.ones(len(env))
    for i in range(len(env)):
        if env[i] > threshold_lin:
            curve[i] = depth_lin
    return curve


def pumping_rate(sidechain_signal: np.ndarray, sr: float,
                  threshold_db: float = -20.0) -> float:
    """Estimate the pumping rate (Hz) — how often the sidechain triggers."""
    env = envelope_follower(sidechain_signal, sr, attack_ms=1.0, release_ms=50.0)
    threshold_lin = db_to_linear(threshold_db)
    crossings = 0
    above = False
    for e in env:
        if e > threshold_lin and not above:
            crossings += 1
            above = True
        elif e <= threshold_lin:
            above = False
    duration_s = len(sidechain_signal) / sr
    return crossings / max(duration_s, 0.001)
