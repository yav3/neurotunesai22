"""Audio dynamics: RMS, crest factor, dynamic range, compression, clipping detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_rms(
    signal: np.ndarray, frame_length: int = 2048, hop_length: int = 512,
) -> np.ndarray:
    n = len(signal)
    n_frames = max(1, 1 + (n - frame_length) // hop_length)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, n)
        frame = signal[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


def compute_crest_factor(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-20:
        return 0.0
    peak = np.max(np.abs(signal))
    return float(20.0 * np.log10(peak / rms))


def compute_dynamic_range(
    signal: np.ndarray, percentile_low: float = 10.0, percentile_high: float = 90.0,
) -> float:
    abs_signal = np.abs(signal)
    abs_signal = abs_signal[abs_signal > 1e-20]
    if len(abs_signal) < 2:
        return 0.0
    db = 20.0 * np.log10(abs_signal)
    low = np.percentile(db, percentile_low)
    high = np.percentile(db, percentile_high)
    return float(high - low)


def compute_loudness_envelope(
    signal: np.ndarray, sr: int, frame_ms: float = 50.0,
) -> np.ndarray:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    rms = compute_rms(signal, frame_length=frame_len, hop_length=frame_len // 2)
    safe_rms = np.maximum(rms, 1e-20)
    return 20.0 * np.log10(safe_rms)


def detect_clipping(
    signal: np.ndarray, threshold: float = 0.99,
) -> list[tuple[int, int]]:
    clipped = np.abs(signal) >= threshold
    regions: list[tuple[int, int]] = []
    in_clip = False
    start = 0
    for i, c in enumerate(clipped):
        if c and not in_clip:
            in_clip = True
            start = i
        elif not c and in_clip:
            in_clip = False
            regions.append((start, i))
    if in_clip:
        regions.append((start, len(signal)))
    return regions


@dataclass
class DynamicsCompressor:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 5.0
    release_ms: float = 50.0
    knee_db: float = 6.0
    makeup_gain_db: float = 0.0

    def process(self, signal: np.ndarray, sr: int) -> np.ndarray:
        attack_coeff = np.exp(-1.0 / (sr * self.attack_ms / 1000.0))
        release_coeff = np.exp(-1.0 / (sr * self.release_ms / 1000.0))
        output = np.zeros_like(signal)
        envelope = 0.0
        makeup = 10.0 ** (self.makeup_gain_db / 20.0)
        for i in range(len(signal)):
            abs_val = abs(signal[i])
            if abs_val > envelope:
                envelope = attack_coeff * envelope + (1 - attack_coeff) * abs_val
            else:
                envelope = release_coeff * envelope + (1 - release_coeff) * abs_val
            if envelope < 1e-20:
                output[i] = signal[i] * makeup
                continue
            level_db = 20.0 * np.log10(envelope)
            if level_db <= self.threshold_db - self.knee_db / 2:
                gain_db = 0.0
            elif level_db >= self.threshold_db + self.knee_db / 2:
                gain_db = (self.threshold_db - level_db) * (1.0 - 1.0 / self.ratio)
            else:
                knee_factor = (level_db - self.threshold_db + self.knee_db / 2) / max(self.knee_db, 1e-10)
                gain_db = (1.0 - 1.0 / self.ratio) * (level_db - self.threshold_db + self.knee_db / 2) ** 2 / (2.0 * max(self.knee_db, 1e-10))
                gain_db = -gain_db
            gain = 10.0 ** (gain_db / 20.0)
            output[i] = signal[i] * gain * makeup
        return output


class DynamicsLimiter(DynamicsCompressor):
    def __init__(
        self, threshold_db: float = -1.0, attack_ms: float = 0.1,
        release_ms: float = 50.0, makeup_gain_db: float = 0.0,
    ) -> None:
        super().__init__(
            threshold_db=threshold_db, ratio=100.0,
            attack_ms=attack_ms, release_ms=release_ms,
            knee_db=0.0, makeup_gain_db=makeup_gain_db,
        )
