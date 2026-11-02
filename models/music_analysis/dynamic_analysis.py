"""Musical dynamics analysis — loudness contours, crescendo/decrescendo, compression metrics."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np


class DynamicType(str, Enum):
    CRESCENDO = "crescendo"
    DECRESCENDO = "decrescendo"
    SFORZANDO = "sforzando"
    SUBITO_PIANO = "subito_piano"
    TERRACED = "terraced"
    STABLE = "stable"


@dataclass
class DynamicEvent:
    start_frame: int
    end_frame: int
    type: DynamicType
    start_db: float
    end_db: float
    slope_db_per_s: float


@dataclass
class DynamicProfile:
    rms_contour_db: np.ndarray
    peak_contour_db: np.ndarray
    dynamic_range_db: float
    events: List[DynamicEvent]
    loudness_war_score: float


def rms_contour(signal: np.ndarray, frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = (len(signal) - frame_size) // hop + 1
    if n_frames < 1:
        return np.array([0.0])
    contour = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        contour[i] = np.sqrt(np.mean(frame ** 2) + 1e-30)
    return contour


def rms_to_db(rms: np.ndarray, ref: float = 1.0) -> np.ndarray:
    return 20.0 * np.log10(rms / ref + 1e-30)


def peak_contour(signal: np.ndarray, frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = (len(signal) - frame_size) // hop + 1
    if n_frames < 1:
        return np.array([0.0])
    contour = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        contour[i] = np.max(np.abs(frame))
    return contour


def dynamic_range(rms_db: np.ndarray, percentile_lo: float = 5.0, percentile_hi: float = 95.0) -> float:
    lo = float(np.percentile(rms_db, percentile_lo))
    hi = float(np.percentile(rms_db, percentile_hi))
    return hi - lo


def sectional_dynamic_range(
    rms_db: np.ndarray, boundaries: List[int],
) -> List[float]:
    ranges = []
    pts = [0] + boundaries + [len(rms_db)]
    for i in range(len(pts) - 1):
        sec = rms_db[pts[i]:pts[i + 1]]
        if len(sec) > 1:
            ranges.append(dynamic_range(sec))
        else:
            ranges.append(0.0)
    return ranges


def detect_crescendo(
    rms_db: np.ndarray, sr: float, hop: int,
    min_duration_s: float = 0.5, min_change_db: float = 3.0,
) -> List[DynamicEvent]:
    events = []
    frame_duration = hop / sr
    smoothed = np.convolve(rms_db, np.ones(5) / 5.0, mode='same')
    diff = np.diff(smoothed)
    positive = diff > 0
    start = None
    for i in range(len(positive)):
        if positive[i] and start is None:
            start = i
        elif not positive[i] and start is not None:
            duration_s = (i - start) * frame_duration
            change_db = smoothed[i] - smoothed[start]
            if duration_s >= min_duration_s and change_db >= min_change_db:
                events.append(DynamicEvent(
                    start_frame=start, end_frame=i,
                    type=DynamicType.CRESCENDO,
                    start_db=float(smoothed[start]),
                    end_db=float(smoothed[i]),
                    slope_db_per_s=change_db / duration_s,
                ))
            start = None
    return events


def detect_decrescendo(
    rms_db: np.ndarray, sr: float, hop: int,
    min_duration_s: float = 0.5, min_change_db: float = 3.0,
) -> List[DynamicEvent]:
    events = []
    frame_duration = hop / sr
    smoothed = np.convolve(rms_db, np.ones(5) / 5.0, mode='same')
    diff = np.diff(smoothed)
    negative = diff < 0
    start = None
    for i in range(len(negative)):
        if negative[i] and start is None:
            start = i
        elif not negative[i] and start is not None:
            duration_s = (i - start) * frame_duration
            change_db = smoothed[start] - smoothed[i]
            if duration_s >= min_duration_s and change_db >= min_change_db:
                events.append(DynamicEvent(
                    start_frame=start, end_frame=i,
                    type=DynamicType.DECRESCENDO,
                    start_db=float(smoothed[start]),
                    end_db=float(smoothed[i]),
                    slope_db_per_s=-change_db / duration_s,
                ))
            start = None
    return events


def detect_sforzando(
    rms_db: np.ndarray, threshold_db: float = 10.0,
) -> List[int]:
    diff = np.diff(rms_db)
    return [int(i) for i in range(len(diff)) if diff[i] > threshold_db]


def is_terraced_dynamics(rms_db: np.ndarray, n_levels: int = 4) -> bool:
    from scipy.cluster.vq import kmeans
    data = rms_db.reshape(-1, 1).astype(float)
    centroids, distortion = kmeans(data, n_levels)
    return distortion < 2.0


def loudness_war_score(rms_db: np.ndarray) -> float:
    dr = dynamic_range(rms_db)
    return float(np.clip(1.0 - dr / 20.0, 0.0, 1.0))


def crest_factor(signal: np.ndarray, frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = (len(signal) - frame_size) // hop + 1
    cf = np.zeros(max(n_frames, 1))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-30)
        peak = np.max(np.abs(frame)) + 1e-30
        cf[i] = 20.0 * np.log10(peak / rms)
    return cf


def compression_ratio_estimate(
    input_db: np.ndarray, output_db: np.ndarray,
) -> float:
    if len(input_db) < 2:
        return 1.0
    slope, _ = np.polyfit(input_db, output_db, 1)
    if slope < 1e-6:
        return float("inf")
    return float(1.0 / slope)


def analyze_dynamics(
    signal: np.ndarray, sr: float = 44100.0,
    frame_size: int = 2048, hop: int = 512,
) -> DynamicProfile:
    rms = rms_contour(signal, frame_size, hop)
    rms_db = rms_to_db(rms)
    peaks = peak_contour(signal, frame_size, hop)
    peak_db = rms_to_db(peaks)
    dr = dynamic_range(rms_db)
    events = detect_crescendo(rms_db, sr, hop) + detect_decrescendo(rms_db, sr, hop)
    lws = loudness_war_score(rms_db)
    return DynamicProfile(
        rms_contour_db=rms_db, peak_contour_db=peak_db,
        dynamic_range_db=dr, events=events, loudness_war_score=lws,
    )
