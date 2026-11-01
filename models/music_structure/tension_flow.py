"""Tension flow analysis: musical tension trajectories, climax detection, narrative arc."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TensionPoint:
    time: float
    tension: float
    components: dict = field(default_factory=dict)


@dataclass
class Climax:
    time: float
    tension: float
    prominence: float
    duration: float = 0.0


@dataclass
class TensionArc:
    points: List[TensionPoint]
    climaxes: List[Climax]
    mean_tension: float
    tension_range: float
    arc_type: str


def harmonic_tension(chord_tones: np.ndarray, key_center: int = 0) -> float:
    if len(chord_tones) == 0:
        return 0.0
    pcs = np.mod(chord_tones - key_center, 12)
    stability = np.array([1.0, 0.2, 0.4, 0.3, 0.5, 0.6, 0.1, 0.9, 0.3, 0.4, 0.2, 0.15])
    stab_sum = sum(stability[int(pc) % 12] for pc in pcs)
    return float(1.0 - stab_sum / (len(pcs) * 1.0))


def dissonance_tension(intervals: np.ndarray) -> float:
    if len(intervals) == 0:
        return 0.0
    dissonance = {0: 0.0, 1: 0.9, 2: 0.5, 3: 0.2, 4: 0.2, 5: 0.1, 6: 0.8,
                  7: 0.05, 8: 0.3, 9: 0.2, 10: 0.6, 11: 0.85}
    total = sum(dissonance.get(int(iv) % 12, 0.5) for iv in intervals)
    return float(np.clip(total / len(intervals), 0, 1))


def register_tension(pitches: np.ndarray) -> float:
    if len(pitches) == 0:
        return 0.0
    mean_pitch = np.mean(pitches)
    tension = np.clip((mean_pitch - 60) / 36, -1, 1)
    return float((tension + 1) / 2)


def rhythmic_tension(onsets: np.ndarray, tempo_period: float) -> float:
    if len(onsets) < 2:
        return 0.0
    ioi = np.diff(onsets)
    expected = tempo_period
    deviations = np.abs(ioi - expected) / expected
    syncopation = np.mean(deviations)
    density = len(onsets) / (onsets[-1] - onsets[0] + 1e-12) * tempo_period
    return float(np.clip(0.5 * syncopation + 0.3 * density + 0.2 * np.std(ioi) / (expected + 1e-12), 0, 1))


def dynamic_tension(amplitudes: np.ndarray) -> float:
    if len(amplitudes) == 0:
        return 0.0
    rms = np.sqrt(np.mean(amplitudes ** 2))
    return float(np.clip(rms, 0, 1))


def compute_tension_curve(
    chord_sequence: List[np.ndarray],
    times: np.ndarray,
    pitches: Optional[np.ndarray] = None,
    onsets: Optional[np.ndarray] = None,
    dynamics: Optional[np.ndarray] = None,
    tempo_period: float = 0.5,
    weights: Optional[dict] = None,
) -> List[TensionPoint]:
    w = weights or {"harmonic": 0.4, "dissonance": 0.2, "register": 0.15,
                    "rhythm": 0.15, "dynamic": 0.1}
    points = []
    for i, (chord, t) in enumerate(zip(chord_sequence, times)):
        components = {}
        components["harmonic"] = harmonic_tension(chord)
        if len(chord) >= 2:
            ivs = np.array([chord[j] - chord[0] for j in range(1, len(chord))])
            components["dissonance"] = dissonance_tension(ivs)
        else:
            components["dissonance"] = 0.0
        if pitches is not None and i < len(pitches):
            components["register"] = register_tension(np.array([pitches[i]]))
        else:
            components["register"] = 0.5
        if onsets is not None:
            mask = (onsets >= t - tempo_period) & (onsets < t + tempo_period)
            local_onsets = onsets[mask]
            components["rhythm"] = rhythmic_tension(local_onsets, tempo_period)
        else:
            components["rhythm"] = 0.3
        if dynamics is not None and i < len(dynamics):
            components["dynamic"] = dynamic_tension(np.array([dynamics[i]]))
        else:
            components["dynamic"] = 0.5
        tension = sum(w.get(k, 0) * v for k, v in components.items())
        points.append(TensionPoint(time=float(t), tension=float(np.clip(tension, 0, 1)),
                                   components=components))
    return points


def detect_climaxes(points: List[TensionPoint], min_prominence: float = 0.15) -> List[Climax]:
    if len(points) < 3:
        return []
    tensions = np.array([p.tension for p in points])
    climaxes = []
    for i in range(1, len(tensions) - 1):
        if tensions[i] > tensions[i - 1] and tensions[i] > tensions[i + 1]:
            left_min = np.min(tensions[:i]) if i > 0 else tensions[i]
            right_min = np.min(tensions[i + 1:]) if i < len(tensions) - 1 else tensions[i]
            prominence = tensions[i] - max(left_min, right_min)
            if prominence >= min_prominence:
                start = i
                while start > 0 and tensions[start - 1] < tensions[i] - prominence / 2:
                    start -= 1
                end = i
                while end < len(tensions) - 1 and tensions[end + 1] < tensions[i] - prominence / 2:
                    end += 1
                duration = points[end].time - points[start].time
                climaxes.append(Climax(
                    time=points[i].time, tension=float(tensions[i]),
                    prominence=float(prominence), duration=duration,
                ))
    climaxes.sort(key=lambda c: c.prominence, reverse=True)
    return climaxes


def classify_arc(points: List[TensionPoint]) -> str:
    if len(points) < 3:
        return "static"
    tensions = np.array([p.tension for p in points])
    n = len(tensions)
    thirds = [tensions[:n // 3], tensions[n // 3:2 * n // 3], tensions[2 * n // 3:]]
    means = [np.mean(t) for t in thirds]
    if means[0] < means[1] > means[2]:
        return "arch"
    if means[0] > means[1] and means[1] > means[2]:
        return "descending"
    if means[0] < means[1] and means[1] < means[2]:
        return "ascending"
    if means[0] > means[1] < means[2]:
        return "valley"
    slope = np.polyfit(np.arange(n), tensions, 1)[0]
    if abs(slope) < 0.001:
        return "static"
    return "complex"


def tension_arc(points: List[TensionPoint], min_prominence: float = 0.15) -> TensionArc:
    tensions = np.array([p.tension for p in points]) if points else np.array([0.0])
    climaxes = detect_climaxes(points, min_prominence)
    return TensionArc(
        points=points, climaxes=climaxes,
        mean_tension=float(np.mean(tensions)),
        tension_range=float(np.ptp(tensions)),
        arc_type=classify_arc(points),
    )


def tension_similarity(arc1: List[TensionPoint], arc2: List[TensionPoint]) -> float:
    t1 = np.array([p.tension for p in arc1])
    t2 = np.array([p.tension for p in arc2])
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
    target_len = max(len(t1), len(t2))
    t1_interp = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(t1)), t1)
    t2_interp = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(t2)), t2)
    corr = np.corrcoef(t1_interp, t2_interp)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float((corr + 1) / 2)


def tension_derivative(points: List[TensionPoint]) -> np.ndarray:
    tensions = np.array([p.tension for p in points])
    if len(tensions) < 2:
        return np.array([0.0])
    times = np.array([p.time for p in points])
    dt = np.diff(times)
    dt = np.where(dt == 0, 1e-12, dt)
    return np.diff(tensions) / dt


def cumulative_tension(points: List[TensionPoint]) -> float:
    if len(points) < 2:
        return 0.0
    tensions = np.array([p.tension for p in points])
    times = np.array([p.time for p in points])
    return float(np.trapezoid(tensions, times))


def tension_entropy(points: List[TensionPoint], n_bins: int = 10) -> float:
    tensions = np.array([p.tension for p in points])
    if len(tensions) == 0:
        return 0.0
    hist, _ = np.histogram(tensions, bins=n_bins, range=(0, 1), density=True)
    hist = hist / (np.sum(hist) + 1e-12)
    mask = hist > 0
    return float(-np.sum(hist[mask] * np.log2(hist[mask])))
