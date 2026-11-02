"""Tempo dynamics: rubato detection, tempo curves, agogic accents, phrase arching."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class TempoFeature(str, Enum):
    RUBATO = "rubato"
    AGOGIC_ACCENT = "agogic_accent"
    RITARDANDO = "ritardando"
    ACCELERANDO = "accelerando"
    FERMATA = "fermata"
    PHRASE_ARCH = "phrase_arch"


@dataclass
class TempoPoint:
    time: float
    bpm: float
    deviation_pct: float = 0.0


@dataclass
class TempoArc:
    start_time: float = 0.0
    end_time: float = 0.0
    peak_time: float = 0.0
    peak_ratio: float = 1.0
    symmetry: float = 0.5


@dataclass
class AgogicEvent:
    index: int
    duration_ratio: float
    feature: TempoFeature = TempoFeature.AGOGIC_ACCENT


def ioi_from_onsets(onsets: np.ndarray) -> np.ndarray:
    if len(onsets) < 2:
        return np.array([])
    return np.diff(onsets)


def onset_to_local_tempo(onsets: np.ndarray, nominal_bpm: float = 120.0
                          ) -> Tuple[np.ndarray, np.ndarray]:
    ioi = ioi_from_onsets(onsets)
    if len(ioi) == 0:
        return np.array([]), np.array([])
    nominal_ioi = 60.0 / nominal_bpm
    bpm = nominal_ioi / np.maximum(ioi, 1e-9) * nominal_bpm
    times = (onsets[:-1] + onsets[1:]) / 2.0
    return times, bpm


def smooth_tempo_curve(bpm: np.ndarray, window: int = 5) -> np.ndarray:
    if len(bpm) < window:
        return bpm.copy()
    kernel = np.ones(window) / window
    return np.convolve(bpm, kernel, mode='same')


def onset_deviations_ms(onsets: np.ndarray,
                        nominal_onsets: np.ndarray) -> np.ndarray:
    n = min(len(onsets), len(nominal_onsets))
    return (onsets[:n] - nominal_onsets[:n]) * 1000.0


def find_rubato_regions(deviations_ms: np.ndarray,
                        threshold_ms: float = 30.0) -> List[Tuple[int, int]]:
    regions = []
    in_region = False
    start = 0
    for i, d in enumerate(np.abs(deviations_ms)):
        if d > threshold_ms and not in_region:
            in_region = True
            start = i
        elif d <= threshold_ms and in_region:
            in_region = False
            regions.append((start, i))
    if in_region:
        regions.append((start, len(deviations_ms)))
    return regions


def find_agogic_accents(durations: np.ndarray,
                        nominal_durations: np.ndarray,
                        threshold: float = 1.3) -> List[AgogicEvent]:
    n = min(len(durations), len(nominal_durations))
    accents = []
    for i in range(n):
        if nominal_durations[i] > 1e-9:
            ratio = durations[i] / nominal_durations[i]
            if ratio > threshold:
                accents.append(AgogicEvent(index=i, duration_ratio=ratio))
    return accents


def find_ritardando(bpm: np.ndarray, min_length: int = 4,
                    min_slope: float = -0.5) -> List[Tuple[int, int]]:
    regions = []
    start = None
    for i in range(1, len(bpm)):
        if bpm[i] - bpm[i - 1] < min_slope:
            if start is None:
                start = i - 1
        else:
            if start is not None and (i - start) >= min_length:
                regions.append((start, i))
            start = None
    if start is not None and (len(bpm) - start) >= min_length:
        regions.append((start, len(bpm)))
    return regions


def find_accelerando(bpm: np.ndarray, min_length: int = 4,
                     min_slope: float = 0.5) -> List[Tuple[int, int]]:
    regions = []
    start = None
    for i in range(1, len(bpm)):
        if bpm[i] - bpm[i - 1] > min_slope:
            if start is None:
                start = i - 1
        else:
            if start is not None and (i - start) >= min_length:
                regions.append((start, i))
            start = None
    if start is not None and (len(bpm) - start) >= min_length:
        regions.append((start, len(bpm)))
    return regions


def detect_phrase_arcs(bpm: np.ndarray, nominal_bpm: float,
                       times: np.ndarray,
                       boundaries: List[Tuple[int, int]]
                       ) -> List[TempoArc]:
    arcs = []
    for start, end in boundaries:
        if end <= start or end > len(bpm):
            continue
        seg = bpm[start:end]
        peak_idx = int(np.argmax(seg))
        peak_ratio = float(seg[peak_idx] / nominal_bpm)
        dur = end - start
        sym = (peak_idx + 0.5) / dur if dur > 0 else 0.5
        arcs.append(TempoArc(
            start_time=float(times[start]) if start < len(times) else 0.0,
            end_time=float(times[min(end - 1, len(times) - 1)]),
            peak_time=float(times[min(start + peak_idx, len(times) - 1)]),
            peak_ratio=peak_ratio, symmetry=sym,
        ))
    return arcs


def tempo_cv(bpm: np.ndarray) -> float:
    if len(bpm) < 2:
        return 0.0
    return float(np.std(bpm) / (np.mean(bpm) + 1e-9))


def tempo_stability_score(bpm: np.ndarray) -> float:
    return float(max(0.0, 1.0 - tempo_cv(bpm)))


def expressiveness_index(deviations_ms: np.ndarray,
                         bpm: np.ndarray) -> float:
    if len(deviations_ms) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(deviations_ms ** 2)))
    stability = tempo_stability_score(bpm)
    return float(np.tanh(rms / 50.0) * stability)


def performance_consistency(takes: List[np.ndarray]) -> float:
    if len(takes) < 2:
        return 1.0
    min_len = min(len(t) for t in takes)
    aligned = np.array([t[:min_len] for t in takes])
    std_per = np.std(aligned, axis=0)
    return float(1.0 - np.tanh(np.mean(std_per) * 1000.0 / 50.0))


def final_rit_strength(bpm: np.ndarray, nominal_bpm: float,
                        tail_frac: float = 0.15) -> float:
    n = len(bpm)
    if n < 4:
        return 0.0
    tail_start = max(0, int(n * (1.0 - tail_frac)))
    tail = bpm[tail_start:]
    if len(tail) < 2:
        return 0.0
    slope = (tail[-1] - tail[0]) / (len(tail) * nominal_bpm + 1e-9)
    return float(max(0.0, -slope))
