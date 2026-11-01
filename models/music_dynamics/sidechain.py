"""Sidechain processing: ducking, pumping, frequency-dependent sidechaining.

Common production technique where one signal's amplitude modulates another.
All operations are pure numpy with sample-accurate processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SidechainMode(str, Enum):
    FULL = "full"
    HIGHPASS = "highpass"
    LOWPASS = "lowpass"
    BANDPASS = "bandpass"


@dataclass
class SidechainParams:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 5.0
    release_ms: float = 100.0
    depth_db: float = -12.0
    mode: SidechainMode = SidechainMode.FULL


def rms_envelope(signal: np.ndarray, window: int = 1024) -> np.ndarray:
    sq = signal ** 2
    kernel = np.ones(window) / window
    env = np.sqrt(np.convolve(sq, kernel, mode="same"))
    return env


def peak_envelope(signal: np.ndarray, attack_samples: int = 10,
                  release_samples: int = 500) -> np.ndarray:
    env = np.zeros(len(signal))
    env[0] = abs(signal[0])
    att = 1.0 - np.exp(-1.0 / max(attack_samples, 1))
    rel = 1.0 - np.exp(-1.0 / max(release_samples, 1))
    for i in range(1, len(signal)):
        inp = abs(signal[i])
        if inp > env[i - 1]:
            env[i] = att * inp + (1 - att) * env[i - 1]
        else:
            env[i] = rel * inp + (1 - rel) * env[i - 1]
    return env


def compute_gain_curve(envelope_db: np.ndarray, threshold_db: float,
                       ratio: float, depth_db: float) -> np.ndarray:
    excess = envelope_db - threshold_db
    excess = np.maximum(excess, 0.0)
    gain_reduction = excess * (1.0 - 1.0 / max(ratio, 1.001))
    gain_reduction = np.minimum(gain_reduction, -depth_db)
    return -gain_reduction


def apply_sidechain(target: np.ndarray, sidechain: np.ndarray,
                    params: SidechainParams,
                    sr: int = 44100) -> np.ndarray:
    n = min(len(target), len(sidechain))
    target = target[:n].copy()
    sc = sidechain[:n]

    attack_samples = max(1, int(params.attack_ms * sr / 1000))
    release_samples = max(1, int(params.release_ms * sr / 1000))

    env = peak_envelope(sc, attack_samples, release_samples)
    env_db = 20.0 * np.log10(env + 1e-10)

    gain_db = compute_gain_curve(env_db, params.threshold_db,
                                  params.ratio, params.depth_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return target * gain_linear


def rhythmic_ducking(target: np.ndarray, bpm: float, sr: int = 44100,
                     depth_db: float = -12.0,
                     duck_fraction: float = 0.25,
                     shape: str = "sine") -> np.ndarray:
    beat_samples = int(60.0 / bpm * sr)
    n = len(target)
    gain = np.ones(n)
    duck_len = int(beat_samples * duck_fraction)

    depth_linear = 10.0 ** (depth_db / 20.0)

    for start in range(0, n, beat_samples):
        end = min(start + duck_len, n)
        seg_len = end - start
        if seg_len <= 0:
            continue
        t = np.linspace(0, np.pi, seg_len)
        if shape == "sine":
            curve = 1.0 - (1.0 - depth_linear) * np.sin(t)
        elif shape == "exponential":
            curve = depth_linear + (1.0 - depth_linear) * (1.0 - np.exp(-3 * t / np.pi))
        else:
            curve = np.linspace(depth_linear, 1.0, seg_len)
        gain[start:end] = curve

    return target * gain


@dataclass
class PumpingProfile:
    depth_db: float
    attack_ms: float
    release_ms: float
    beats_per_pump: float = 1.0

    def generate_envelope(self, n_samples: int, sr: int = 44100,
                          bpm: float = 120.0) -> np.ndarray:
        beat_len = int(60.0 / bpm * sr * self.beats_per_pump)
        attack_len = int(self.attack_ms * sr / 1000)
        release_len = int(self.release_ms * sr / 1000)
        depth = 10.0 ** (self.depth_db / 20.0)

        one_cycle = np.ones(beat_len)
        if attack_len > 0 and attack_len < beat_len:
            one_cycle[:attack_len] = np.linspace(depth, 1.0, attack_len)
        remain = beat_len - attack_len
        if release_len > 0 and remain > 0:
            rel = min(release_len, remain)
            one_cycle[beat_len - rel:] = np.linspace(1.0, depth, rel)

        n_reps = (n_samples // beat_len) + 2
        full = np.tile(one_cycle, n_reps)
        return full[:n_samples]


def multiband_sidechain(target: np.ndarray, sidechain: np.ndarray,
                        crossover_hz: float = 200.0,
                        sr: int = 44100,
                        low_params: SidechainParams | None = None,
                        high_params: SidechainParams | None = None) -> np.ndarray:
    from numpy.fft import rfft, irfft

    n = min(len(target), len(sidechain))
    target = target[:n]
    sc = sidechain[:n]

    T = rfft(target)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    low_mask = (freqs <= crossover_hz).astype(float)
    high_mask = 1.0 - low_mask

    target_low = irfft(T * low_mask, n=n)
    target_high = irfft(T * high_mask, n=n)

    lp = low_params or SidechainParams(depth_db=-18.0)
    hp = high_params or SidechainParams(depth_db=-6.0)

    processed_low = apply_sidechain(target_low, sc, lp, sr)
    processed_high = apply_sidechain(target_high, sc, hp, sr)

    return processed_low + processed_high
