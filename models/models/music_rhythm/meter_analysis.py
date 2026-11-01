"""Computational meter analysis: pulse extraction, beat tracking, groove.

Implements autocorrelation-based pulse strength estimation, hierarchical
meter hypothesis testing, beat tracking from onsets, Lerdahl-Jackendoff
metrical well-formedness, groove/microtiming quantification, and polyrhythm
detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pulse_autocorrelation(onsets: np.ndarray, sr: int = 100,
                          max_bpm: float = 240.0,
                          min_bpm: float = 30.0) -> np.ndarray:
    """Compute pulse strength at each lag via onset autocorrelation."""
    min_lag = max(1, int(60.0 / max_bpm * sr))
    max_lag = min(len(onsets) - 1, int(60.0 / min_bpm * sr))
    if min_lag >= max_lag:
        return np.array([])
    acf = np.zeros(max_lag + 1)
    norm = np.sum(onsets ** 2)
    if norm < 1e-12:
        return acf
    for lag in range(min_lag, max_lag + 1):
        acf[lag] = np.sum(onsets[: len(onsets) - lag] * onsets[lag:]) / norm
    return acf


def estimate_tempo(onsets: np.ndarray, sr: int = 100,
                   min_bpm: float = 40.0, max_bpm: float = 220.0) -> float:
    acf = pulse_autocorrelation(onsets, sr, max_bpm, min_bpm)
    min_lag = max(1, int(60.0 / max_bpm * sr))
    max_lag = min(len(acf) - 1, int(60.0 / min_bpm * sr))
    if min_lag >= max_lag or len(acf) <= max_lag:
        return 120.0
    search = acf[min_lag : max_lag + 1]
    if len(search) == 0:
        return 120.0
    best_lag = min_lag + int(np.argmax(search))
    return 60.0 * sr / best_lag


@dataclass
class MeterHypothesis:
    """Hierarchical meter as nested pulse levels."""
    beats_per_bar: int
    subdivisions: int
    tempo_bpm: float
    strength: float

    @property
    def beat_period_sec(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def bar_period_sec(self) -> float:
        return self.beat_period_sec * self.beats_per_bar

    @property
    def subdivision_period_sec(self) -> float:
        return self.beat_period_sec / self.subdivisions


def test_meter_hypotheses(onsets: np.ndarray, sr: int = 100,
                          tempo_bpm: float | None = None) -> list[MeterHypothesis]:
    if tempo_bpm is None:
        tempo_bpm = estimate_tempo(onsets, sr)
    acf = pulse_autocorrelation(onsets, sr)
    beat_lag = int(60.0 / tempo_bpm * sr)
    hypotheses = []
    for bpb in (2, 3, 4, 6):
        for subdiv in (1, 2, 3):
            bar_lag = beat_lag * bpb
            sub_lag = max(1, beat_lag // subdiv)
            strength = 0.0
            count = 0
            for lag in (sub_lag, beat_lag, bar_lag):
                if 0 < lag < len(acf):
                    strength += acf[lag]
                    count += 1
            if count > 0:
                hypotheses.append(MeterHypothesis(bpb, subdiv, tempo_bpm, strength / count))
    hypotheses.sort(key=lambda h: h.strength, reverse=True)
    return hypotheses


@dataclass
class BeatTrack:
    """Beat positions and confidence."""
    times: np.ndarray
    tempo_bpm: float
    confidence: float


def track_beats(onset_times: np.ndarray, tempo_bpm: float | None = None) -> BeatTrack:
    if len(onset_times) < 2:
        return BeatTrack(onset_times, 120.0, 0.0)
    ioi = np.diff(onset_times)
    median_ioi = float(np.median(ioi))
    if tempo_bpm is None:
        tempo_bpm = 60.0 / median_ioi if median_ioi > 0 else 120.0
    beat_period = 60.0 / tempo_bpm
    beats = [onset_times[0]]
    for t in onset_times[1:]:
        expected = beats[-1] + beat_period
        if abs(t - expected) < beat_period * 0.35:
            beats.append(t)
        elif t > expected + beat_period * 0.5:
            while beats[-1] + beat_period < t - beat_period * 0.35:
                beats.append(beats[-1] + beat_period)
            beats.append(t)
    beat_arr = np.array(beats)
    if len(beat_arr) < 2:
        return BeatTrack(beat_arr, tempo_bpm, 0.0)
    ioi_beats = np.diff(beat_arr)
    regularity = 1.0 - float(np.std(ioi_beats) / (np.mean(ioi_beats) + 1e-12))
    return BeatTrack(beat_arr, tempo_bpm, max(0.0, regularity))


# ---------------------------------------------------------------------------
# Metrical complexity (Lerdahl-Jackendoff inspired)
# ---------------------------------------------------------------------------

def metrical_weight(position: int, beats_per_bar: int, subdivisions: int) -> float:
    """Assign metrical weight to a position in a bar.

    Downbeat = 1.0, other strong beats scaled by level in hierarchy.
    """
    total = beats_per_bar * subdivisions
    pos = position % total
    if pos == 0:
        return 1.0
    for level in range(1, 6):
        step = total // (2 ** level) if total % (2 ** level) == 0 else 0
        if step > 0 and pos % step == 0:
            return 1.0 / (level + 1)
    return 0.1


def syncopation_score(accents: list[float], meter: list[float]) -> float:
    """Measure syncopation as sum of accent on metrically weak positions."""
    if not accents or not meter:
        return 0.0
    n = min(len(accents), len(meter))
    score = 0.0
    for i in range(n):
        if accents[i] > 0.5 and meter[i] < 0.5:
            score += accents[i] * (1.0 - meter[i])
    return score / n


# ---------------------------------------------------------------------------
# Groove and microtiming
# ---------------------------------------------------------------------------

@dataclass
class GrooveAnalysis:
    mean_deviation_ms: float
    swing_ratio: float
    push_pull: float
    groove_consistency: float


def analyze_groove(onset_times: np.ndarray, grid_times: np.ndarray) -> GrooveAnalysis:
    n = min(len(onset_times), len(grid_times))
    if n < 2:
        return GrooveAnalysis(0.0, 1.0, 0.0, 1.0)
    deviations = (onset_times[:n] - grid_times[:n]) * 1000.0
    mean_dev = float(np.mean(np.abs(deviations)))
    push_pull = float(np.mean(deviations))
    consistency = max(0.0, 1.0 - float(np.std(deviations)) / 50.0)
    ioi = np.diff(onset_times[:n])
    if len(ioi) >= 2:
        long = ioi[::2]
        short = ioi[1::2]
        n_pairs = min(len(long), len(short))
        if n_pairs > 0:
            swing = float(np.mean(long[:n_pairs]) / (np.mean(short[:n_pairs]) + 1e-12))
        else:
            swing = 1.0
    else:
        swing = 1.0
    return GrooveAnalysis(mean_dev, swing, push_pull, consistency)


# ---------------------------------------------------------------------------
# Polyrhythm detection
# ---------------------------------------------------------------------------

def detect_polyrhythm(onsets_a: np.ndarray, onsets_b: np.ndarray,
                      tolerance: float = 0.05) -> dict[str, float]:
    if len(onsets_a) < 2 or len(onsets_b) < 2:
        return {"ratio": 1.0, "confidence": 0.0}
    ioi_a = float(np.median(np.diff(onsets_a)))
    ioi_b = float(np.median(np.diff(onsets_b)))
    if ioi_b < 1e-12:
        return {"ratio": 1.0, "confidence": 0.0}
    raw_ratio = ioi_a / ioi_b
    candidates = [(2, 3), (3, 2), (3, 4), (4, 3), (5, 4), (4, 5)]
    best_match = (1, 1)
    best_err = abs(raw_ratio - 1.0)
    for p, q in candidates:
        target = p / q
        err = abs(raw_ratio - target)
        if err < best_err:
            best_err = err
            best_match = (p, q)
    confidence = max(0.0, 1.0 - best_err / tolerance) if best_err < tolerance else 0.0
    return {"ratio": best_match[0] / best_match[1],
            "pattern": f"{best_match[0]}:{best_match[1]}",
            "confidence": confidence}
