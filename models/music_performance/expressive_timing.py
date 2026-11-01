"""Expressive timing: model micro-timing deviations in human performance.

Quantify and synthesise the subtle timing deviations that distinguish
human performance from rigid quantisation: swing ratio, anticipation/
delay profiles, phrase-arc timing, and groove template extraction.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TimingDeviation:
    onset_s: float
    grid_s: float
    deviation_ms: float
    position_in_beat: float


def swing_ratio(onsets_s: np.ndarray, bpm: float,
                subdivisions: int = 2) -> float:
    beat_dur = 60.0 / bpm
    sub_dur = beat_dur / subdivisions
    even_devs = []
    odd_devs = []
    for t in onsets_s:
        beat_phase = (t % beat_dur) / sub_dur
        sub_idx = int(round(beat_phase)) % subdivisions
        if sub_idx == 0:
            even_devs.append(t % sub_dur)
        else:
            odd_devs.append(t % sub_dur)
    if not even_devs or not odd_devs:
        return 0.5
    even_mean = np.mean(even_devs)
    odd_mean = np.mean(odd_devs)
    total = even_mean + odd_mean
    return float(odd_mean / total) if total > 1e-6 else 0.5


def timing_deviations(onsets_s: np.ndarray, bpm: float,
                      downbeat_s: float = 0.0) -> List[TimingDeviation]:
    beat_dur = 60.0 / bpm
    grid_points = np.arange(0, onsets_s[-1] + beat_dur, beat_dur / 4) + downbeat_s
    result = []
    for t in onsets_s:
        nearest_idx = np.argmin(np.abs(grid_points - t))
        grid_t = grid_points[nearest_idx]
        dev_ms = (t - grid_t) * 1000.0
        beat_phase = ((t - downbeat_s) % beat_dur) / beat_dur
        result.append(TimingDeviation(t, grid_t, dev_ms, beat_phase))
    return result


def phrase_arc_timing(deviations_ms: np.ndarray,
                      phrase_length: int) -> np.ndarray:
    if len(deviations_ms) < phrase_length:
        return deviations_ms
    n_phrases = len(deviations_ms) // phrase_length
    arcs = np.zeros(phrase_length)
    for p in range(n_phrases):
        segment = deviations_ms[p * phrase_length:(p + 1) * phrase_length]
        arcs += segment
    return arcs / max(n_phrases, 1)


def groove_template(onsets_s: np.ndarray, bpm: float,
                    steps: int = 16) -> np.ndarray:
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4
    step_dur = bar_dur / steps
    template = np.zeros(steps)
    counts = np.zeros(steps)
    for t in onsets_s:
        bar_phase = (t % bar_dur) / step_dur
        step = int(bar_phase) % steps
        deviation = (bar_phase - round(bar_phase)) * step_dur * 1000
        template[step] += deviation
        counts[step] += 1
    nonzero = counts > 0
    template[nonzero] /= counts[nonzero]
    return template


def apply_groove(quantised_s: np.ndarray, template_ms: np.ndarray,
                 bpm: float, strength: float = 1.0) -> np.ndarray:
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4
    steps = len(template_ms)
    step_dur = bar_dur / steps
    result = quantised_s.copy()
    for i, t in enumerate(quantised_s):
        step = int((t % bar_dur) / step_dur) % steps
        result[i] = t + strength * template_ms[step] / 1000.0
    return result


def humanise(onsets_s: np.ndarray, sigma_ms: float = 10.0,
             seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma_ms / 1000.0, len(onsets_s))
    return np.sort(onsets_s + noise)


def asynchrony_profile(onsets_by_part: Dict[str, np.ndarray],
                       tolerance_ms: float = 50.0) -> Dict[str, float]:
    parts = list(onsets_by_part.keys())
    result = {}
    for i, p1 in enumerate(parts):
        for p2 in parts[i + 1:]:
            paired = []
            for t1 in onsets_by_part[p1]:
                diffs = np.abs(onsets_by_part[p2] - t1) * 1000
                nearest = np.min(diffs)
                if nearest < tolerance_ms:
                    paired.append(nearest)
            key = f"{p1}_vs_{p2}"
            result[key] = float(np.mean(paired)) if paired else 0.0
    return result


def tempo_stability(onsets_s: np.ndarray) -> Dict[str, float]:
    if len(onsets_s) < 3:
        return {"mean_ioi": 0.0, "cv": 0.0, "drift": 0.0}
    iois = np.diff(onsets_s)
    mean_ioi = float(np.mean(iois))
    cv = float(np.std(iois) / mean_ioi) if mean_ioi > 1e-6 else 0.0
    half = len(iois) // 2
    drift = float(np.mean(iois[half:]) - np.mean(iois[:half]))
    return {"mean_ioi": mean_ioi, "cv": cv, "drift": drift}
