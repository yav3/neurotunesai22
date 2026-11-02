"""Dynamic analysis: extract and characterise dynamics from audio or MIDI."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DynamicProfile:
    """Summary statistics of dynamic range and variation."""
    mean_velocity: float
    std_velocity: float
    min_velocity: int
    max_velocity: int
    dynamic_range: int
    loudness_curve: np.ndarray
    rms_db: np.ndarray


def rms_envelope(signal: np.ndarray, frame_size: int = 2048, hop_size: int = 512) -> np.ndarray:
    """Compute RMS envelope of an audio signal."""
    n_frames = 1 + max(0, (len(signal) - frame_size) // hop_size)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        frame = signal[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


def rms_to_db(rms: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """Convert RMS values to decibels."""
    return 20.0 * np.log10(np.maximum(rms, 1e-10) / ref)


def velocity_histogram(velocities: list[int], n_bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of MIDI velocities."""
    counts, edges = np.histogram(velocities, bins=n_bins, range=(0, 128))
    return counts, edges


def dynamic_range_midi(velocities: list[int]) -> int:
    """Dynamic range as max - min velocity."""
    if not velocities:
        return 0
    return max(velocities) - min(velocities)


def loudness_variation(rms_db: np.ndarray, window: int = 10) -> np.ndarray:
    """Local standard deviation of loudness over a sliding window."""
    n = len(rms_db)
    if n < window:
        return np.array([float(np.std(rms_db))])
    result = np.zeros(n - window + 1)
    for i in range(len(result)):
        result[i] = np.std(rms_db[i : i + window])
    return result


def detect_dynamic_changes(
    velocities: list[int],
    threshold: int = 15,
) -> list[tuple[int, int, int]]:
    """Find abrupt dynamic changes (subito) in a velocity sequence.

    Returns list of (index, velocity_before, velocity_after) for each change
    exceeding the threshold.
    """
    changes = []
    for i in range(1, len(velocities)):
        delta = abs(velocities[i] - velocities[i - 1])
        if delta >= threshold:
            changes.append((i, velocities[i - 1], velocities[i]))
    return changes


def classify_dynamic_level(velocity: int) -> str:
    """Classify a single MIDI velocity into a dynamic marking name."""
    if velocity <= 15:
        return "ppp"
    if velocity <= 33:
        return "pp"
    if velocity <= 49:
        return "p"
    if velocity <= 64:
        return "mp"
    if velocity <= 80:
        return "mf"
    if velocity <= 96:
        return "f"
    if velocity <= 112:
        return "ff"
    return "fff"


def analyse_dynamics(
    velocities: list[int],
    signal: np.ndarray | None = None,
    sr: int = 44100,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> DynamicProfile:
    """Full dynamic analysis from MIDI velocities and optionally audio."""
    vels = np.array(velocities, dtype=float)
    mean_v = float(np.mean(vels)) if len(vels) > 0 else 0.0
    std_v = float(np.std(vels)) if len(vels) > 0 else 0.0
    min_v = int(np.min(vels)) if len(vels) > 0 else 0
    max_v = int(np.max(vels)) if len(vels) > 0 else 0

    if signal is not None and len(signal) > 0:
        rms = rms_envelope(signal, frame_size, hop_size)
        rms_db = rms_to_db(rms)
    else:
        rms = np.array([mean_v / 127.0])
        rms_db = rms_to_db(rms)

    return DynamicProfile(
        mean_velocity=mean_v,
        std_velocity=std_v,
        min_velocity=min_v,
        max_velocity=max_v,
        dynamic_range=max_v - min_v,
        loudness_curve=rms,
        rms_db=rms_db,
    )


def compute_compression_ratio(
    velocities: list[int],
    target_range: int = 40,
) -> float:
    """How much the velocity range would need compressing to fit target_range.

    Returns ratio >= 1.0 (1.0 = no compression needed).
    """
    dr = dynamic_range_midi(velocities)
    if dr <= target_range or dr == 0:
        return 1.0
    return dr / target_range


def normalise_velocities(
    velocities: list[int],
    target_mean: int = 80,
    target_range: int = 60,
) -> list[int]:
    """Normalise velocity distribution to a target mean and range."""
    if not velocities:
        return []
    arr = np.array(velocities, dtype=float)
    current_mean = np.mean(arr)
    current_range = np.max(arr) - np.min(arr)
    if current_range == 0:
        return [target_mean] * len(velocities)
    scale = target_range / current_range
    shifted = (arr - current_mean) * scale + target_mean
    return [max(1, min(127, int(round(v)))) for v in shifted]
