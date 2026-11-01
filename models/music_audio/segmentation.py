"""Audio segmentation: silence detection, activity detection, boundaries."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start_sample: int
    end_sample: int
    label: str
    energy_db: float


def rms_energy(signal: np.ndarray, frame_size: int = 1024, hop: int = 512) -> np.ndarray:
    """Frame-wise RMS energy in dB."""
    n = len(signal)
    n_frames = max(1, (n - frame_size) // hop + 1)
    result = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size].astype(float)
        rms = math.sqrt(float(np.mean(frame ** 2)) + 1e-30)
        result[i] = 20.0 * math.log10(rms + 1e-30)
    return result


def detect_silence(
    signal: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_duration_sec: float = 0.5,
    frame_size: int = 1024,
    hop: int = 512,
) -> list[Segment]:
    """Find silent regions (energy below threshold for min_duration)."""
    energy = rms_energy(signal, frame_size, hop)
    min_frames = max(1, int(min_duration_sec * sr / hop))

    segments: list[Segment] = []
    in_silence = False
    start_frame = 0

    for i, e in enumerate(energy):
        if e < threshold_db:
            if not in_silence:
                in_silence = True
                start_frame = i
        else:
            if in_silence:
                duration = i - start_frame
                if duration >= min_frames:
                    segments.append(Segment(
                        start_sample=start_frame * hop,
                        end_sample=i * hop,
                        label="silence",
                        energy_db=float(np.mean(energy[start_frame:i])),
                    ))
                in_silence = False

    if in_silence:
        duration = len(energy) - start_frame
        if duration >= min_frames:
            segments.append(Segment(
                start_sample=start_frame * hop,
                end_sample=len(signal),
                label="silence",
                energy_db=float(np.mean(energy[start_frame:])),
            ))

    return segments


def detect_activity(
    signal: np.ndarray,
    sr: int,
    threshold_db: float = -30.0,
    frame_size: int = 1024,
    hop: int = 512,
) -> list[Segment]:
    """Find active (non-silent) regions."""
    energy = rms_energy(signal, frame_size, hop)
    segments: list[Segment] = []
    in_activity = False
    start_frame = 0

    for i, e in enumerate(energy):
        if e >= threshold_db:
            if not in_activity:
                in_activity = True
                start_frame = i
        else:
            if in_activity:
                segments.append(Segment(
                    start_sample=start_frame * hop,
                    end_sample=i * hop,
                    label="active",
                    energy_db=float(np.mean(energy[start_frame:i])),
                ))
                in_activity = False

    if in_activity:
        segments.append(Segment(
            start_sample=start_frame * hop,
            end_sample=len(signal),
            label="active",
            energy_db=float(np.mean(energy[start_frame:])),
        ))

    return segments


def energy_novelty(
    signal: np.ndarray,
    frame_size: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    """Detect boundaries via energy novelty (first-order difference of RMS)."""
    energy = rms_energy(signal, frame_size, hop)
    if len(energy) < 2:
        return np.zeros(max(len(energy), 0))
    novelty = np.zeros(len(energy))
    novelty[1:] = np.maximum(np.diff(energy), 0.0)
    return novelty


def find_boundaries(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop: int = 512,
    threshold: float = 3.0,
    min_gap_sec: float = 1.0,
) -> np.ndarray:
    """Find segment boundaries from energy novelty peaks.

    Returns array of boundary times in seconds.
    """
    novelty = energy_novelty(signal, frame_size, hop)
    if len(novelty) == 0:
        return np.array([])

    mean_n = float(np.mean(novelty))
    std_n = float(np.std(novelty))
    if std_n < 1e-12:
        return np.array([])

    peak_mask = novelty > mean_n + threshold * std_n
    peak_frames = np.where(peak_mask)[0]

    if len(peak_frames) == 0:
        return np.array([])

    min_gap_frames = max(1, int(min_gap_sec * sr / hop))
    filtered = [peak_frames[0]]
    for f in peak_frames[1:]:
        if f - filtered[-1] >= min_gap_frames:
            filtered.append(f)

    return np.array(filtered) * hop / sr
