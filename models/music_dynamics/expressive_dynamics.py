"""Expressive dynamics: crescendo/decrescendo detection, terraced vs. gradual dynamics, accent patterns."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class DynamicMarking(Enum):
    PPP = -3
    PP = -2
    P = -1
    MP = 0
    MF = 1
    F = 2
    FF = 3
    FFF = 4


class DynamicContourType(Enum):
    CRESCENDO = "crescendo"
    DECRESCENDO = "decrescendo"
    TERRACED_UP = "terraced_up"
    TERRACED_DOWN = "terraced_down"
    STABLE = "stable"
    SFORZANDO = "sforzando"
    SUBITO = "subito"


@dataclass(frozen=True)
class DynamicEvent:
    start_time: float
    end_time: float
    contour: DynamicContourType
    start_level_db: float
    end_level_db: float


@dataclass(frozen=True)
class DynamicProfile:
    events: List[DynamicEvent]
    mean_level_db: float
    dynamic_range_db: float
    loudness_curve: np.ndarray


def rms_envelope(signal: np.ndarray, hop: int = 512) -> np.ndarray:
    n = len(signal)
    n_frames = max(n // hop, 1)
    env = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, n)
        frame = signal[start:end]
        env[i] = np.sqrt(np.mean(frame ** 2))
    return env


def db_envelope(signal: np.ndarray, hop: int = 512) -> np.ndarray:
    env = rms_envelope(signal, hop)
    return 20.0 * np.log10(np.maximum(env, 1e-10))


def classify_dynamic_level(db: float) -> DynamicMarking:
    if db < -50:
        return DynamicMarking.PPP
    elif db < -40:
        return DynamicMarking.PP
    elif db < -30:
        return DynamicMarking.P
    elif db < -24:
        return DynamicMarking.MP
    elif db < -18:
        return DynamicMarking.MF
    elif db < -12:
        return DynamicMarking.F
    elif db < -6:
        return DynamicMarking.FF
    else:
        return DynamicMarking.FFF


def detect_crescendo(envelope_db: np.ndarray, min_frames: int = 5,
                     threshold_db: float = 3.0) -> List[Tuple[int, int, float]]:
    results = []
    n = len(envelope_db)
    i = 0
    while i < n - min_frames:
        run_start = i
        while i < n - 1 and envelope_db[i + 1] >= envelope_db[i] - 0.5:
            i += 1
        run_end = i
        if run_end - run_start >= min_frames:
            change = envelope_db[run_end] - envelope_db[run_start]
            if change >= threshold_db:
                results.append((run_start, run_end, float(change)))
        i += 1
    return results


def detect_decrescendo(envelope_db: np.ndarray, min_frames: int = 5,
                       threshold_db: float = 3.0) -> List[Tuple[int, int, float]]:
    results = []
    n = len(envelope_db)
    i = 0
    while i < n - min_frames:
        run_start = i
        while i < n - 1 and envelope_db[i + 1] <= envelope_db[i] + 0.5:
            i += 1
        run_end = i
        if run_end - run_start >= min_frames:
            change = envelope_db[run_start] - envelope_db[run_end]
            if change >= threshold_db:
                results.append((run_start, run_end, float(change)))
        i += 1
    return results


def detect_sforzando(envelope_db: np.ndarray,
                     threshold_db: float = 10.0) -> List[int]:
    results = []
    for i in range(1, len(envelope_db) - 1):
        rise = envelope_db[i] - envelope_db[i - 1]
        fall = envelope_db[i] - envelope_db[i + 1]
        if rise > threshold_db and fall > threshold_db * 0.5:
            results.append(i)
    return results


def terraced_dynamics(envelope_db: np.ndarray,
                      window: int = 10) -> List[Tuple[int, float]]:
    n = len(envelope_db)
    changes = []
    for i in range(window, n - window):
        left_mean = np.mean(envelope_db[i - window:i])
        right_mean = np.mean(envelope_db[i:i + window])
        left_std = np.std(envelope_db[i - window:i])
        right_std = np.std(envelope_db[i:i + window])
        if left_std < 2.0 and right_std < 2.0 and abs(right_mean - left_mean) > 4.0:
            changes.append((i, float(right_mean - left_mean)))
    return changes


def dynamic_range(envelope_db: np.ndarray) -> float:
    if len(envelope_db) == 0:
        return 0.0
    return float(np.percentile(envelope_db, 95) - np.percentile(envelope_db, 5))


def dynamic_variance(envelope_db: np.ndarray) -> float:
    if len(envelope_db) < 2:
        return 0.0
    return float(np.var(envelope_db))


def accent_pattern(envelope_db: np.ndarray,
                   beats_per_measure: int = 4) -> np.ndarray:
    n = len(envelope_db)
    if n < beats_per_measure:
        return envelope_db.copy()
    n_full = (n // beats_per_measure) * beats_per_measure
    reshaped = envelope_db[:n_full].reshape(-1, beats_per_measure)
    return np.mean(reshaped, axis=0)


def loudness_contour_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a_norm = (a[:n] - np.mean(a[:n]))
    b_norm = (b[:n] - np.mean(b[:n]))
    denom = np.sqrt(np.sum(a_norm ** 2) * np.sum(b_norm ** 2))
    if denom < 1e-10:
        return 1.0
    return float(np.sum(a_norm * b_norm) / denom)


def analyze_dynamics(signal: np.ndarray, sr: int,
                     hop: int = 512) -> DynamicProfile:
    env_db = db_envelope(signal, hop)
    cresc = detect_crescendo(env_db)
    decresc = detect_decrescendo(env_db)
    sfz = detect_sforzando(env_db)
    frame_dur = hop / sr
    events = []
    for s, e, change in cresc:
        events.append(DynamicEvent(
            s * frame_dur, e * frame_dur,
            DynamicContourType.CRESCENDO,
            float(env_db[s]), float(env_db[e])))
    for s, e, change in decresc:
        events.append(DynamicEvent(
            s * frame_dur, e * frame_dur,
            DynamicContourType.DECRESCENDO,
            float(env_db[s]), float(env_db[e])))
    for idx in sfz:
        t = idx * frame_dur
        events.append(DynamicEvent(
            t, t + frame_dur,
            DynamicContourType.SFORZANDO,
            float(env_db[idx - 1]) if idx > 0 else float(env_db[idx]),
            float(env_db[idx])))
    events.sort(key=lambda e: e.start_time)
    return DynamicProfile(
        events=events,
        mean_level_db=float(np.mean(env_db)),
        dynamic_range_db=dynamic_range(env_db),
        loudness_curve=env_db,
    )
