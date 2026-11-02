from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmotionArc:
    timestamps: list[float]
    valence_curve: list[float]
    arousal_curve: list[float]
    dominance_curve: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if not self.dominance_curve:
            self.dominance_curve = [0.5] * n


ARC_SHAPES = [
    "ramp_up",
    "ramp_down",
    "peak",
    "valley",
    "flat",
    "oscillating",
    "tension_release",
]


def _direction_changes(curve: list[float]) -> int:
    if len(curve) < 3:
        return 0
    diffs = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    count = 0
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i + 1] < 0:
            count += 1
    return count


def _curve_range(curve: list[float]) -> float:
    if not curve:
        return 0.0
    return max(curve) - min(curve)


class EmotionArcAnalyzer:
    @staticmethod
    def detect_arc_shape(arc: EmotionArc) -> tuple[str, float]:
        if len(arc.arousal_curve) < 3:
            return ("flat", 1.0)

        a = np.array(arc.arousal_curve, dtype=np.float64)
        n = len(a)
        t = np.linspace(0.0, 1.0, n)

        rng = float(a.max() - a.min())
        if rng < 0.05:
            return ("flat", 0.95)

        peak_idx = int(np.argmax(a))
        valley_idx = int(np.argmin(a))

        dc = _direction_changes(arc.arousal_curve)
        osc_ratio = dc / max(n - 2, 1)

        if osc_ratio > 0.03 and dc >= 4:
            return ("oscillating", min(1.0, osc_ratio * 10.0))

        slope = float(np.polyfit(t, a, 1)[0])

        if 0.2 < peak_idx / n < 0.8 and peak_idx != 0 and peak_idx != n - 1:
            left_mean = float(a[: peak_idx].mean())
            right_mean = float(a[peak_idx + 1 :].mean())
            peak_val = float(a[peak_idx])
            if peak_val > left_mean + rng * 0.25 and peak_val > right_mean + rng * 0.25:
                return ("peak", min(1.0, (peak_val - max(left_mean, right_mean)) / (rng + 1e-9)))

        if 0.2 < valley_idx / n < 0.8 and valley_idx != 0 and valley_idx != n - 1:
            left_mean = float(a[: valley_idx].mean())
            right_mean = float(a[valley_idx + 1 :].mean())
            valley_val = float(a[valley_idx])
            if valley_val < left_mean - rng * 0.25 and valley_val < right_mean - rng * 0.25:
                return ("valley", min(1.0, (min(left_mean, right_mean) - valley_val) / (rng + 1e-9)))

        # tension_release: long build then sharp drop in the last quarter
        three_quarter = int(n * 0.75)
        if three_quarter > 2:
            build_slope = float(np.polyfit(t[:three_quarter], a[:three_quarter], 1)[0])
            tail_drop = float(a[three_quarter:].mean()) - float(a[:three_quarter].mean())
            if build_slope > 0.3 and tail_drop < -rng * 0.15:
                return ("tension_release", min(1.0, abs(tail_drop) / (rng + 1e-9)))

        if slope > 0.3:
            return ("ramp_up", min(1.0, slope / (rng + 1e-9)))
        if slope < -0.3:
            return ("ramp_down", min(1.0, abs(slope) / (rng + 1e-9)))

        return ("flat", 0.5)

    @staticmethod
    def arc_complexity(arc: EmotionArc) -> float:
        dc_v = _direction_changes(arc.valence_curve)
        dc_a = _direction_changes(arc.arousal_curve)
        rng_v = _curve_range(arc.valence_curve)
        rng_a = _curve_range(arc.arousal_curve)
        n = max(len(arc.timestamps) - 2, 1)
        change_rate = (dc_v + dc_a) / (2.0 * n)
        range_factor = (rng_v + rng_a) / 2.0
        return float(np.clip(change_rate * 0.6 + range_factor * 0.4, 0.0, 1.0))

    @staticmethod
    def emotional_journey_distance(arc: EmotionArc) -> float:
        total = 0.0
        for i in range(len(arc.timestamps) - 1):
            dv = arc.valence_curve[i + 1] - arc.valence_curve[i]
            da = arc.arousal_curve[i + 1] - arc.arousal_curve[i]
            dd = arc.dominance_curve[i + 1] - arc.dominance_curve[i]
            total += math.sqrt(dv ** 2 + da ** 2 + dd ** 2)
        return total

    @staticmethod
    def generate_target_arc(
        shape: str,
        duration_s: float,
        n_points: int = 100,
    ) -> EmotionArc:
        t = np.linspace(0.0, duration_s, n_points)
        x = np.linspace(0.0, 1.0, n_points)

        if shape == "ramp_up":
            a = 0.2 + 0.6 * x
            v = 0.3 + 0.5 * x
        elif shape == "ramp_down":
            a = 0.8 - 0.6 * x
            v = 0.8 - 0.5 * x
        elif shape == "peak":
            a = 0.2 + 0.7 * np.sin(np.pi * x)
            v = 0.3 + 0.5 * np.sin(np.pi * x)
        elif shape == "valley":
            a = 0.8 - 0.6 * np.sin(np.pi * x)
            v = 0.7 - 0.4 * np.sin(np.pi * x)
        elif shape == "oscillating":
            a = 0.5 + 0.3 * np.sin(4 * np.pi * x)
            v = 0.5 + 0.2 * np.cos(4 * np.pi * x)
        elif shape == "tension_release":
            a = np.where(x < 0.75, 0.2 + 0.8 * (x / 0.75), 1.0 - 2.5 * (x - 0.75))
            v = np.where(x < 0.75, 0.3 + 0.2 * (x / 0.75), 0.5 + 1.5 * (x - 0.75))
        else:  # flat
            a = np.full(n_points, 0.5)
            v = np.full(n_points, 0.5)

        a_clipped: np.ndarray = np.clip(a, 0.0, 1.0)
        v_clipped: np.ndarray = np.clip(v, 0.0, 1.0)
        d = np.full(n_points, 0.5)

        return EmotionArc(
            timestamps=t.tolist(),
            valence_curve=v_clipped.tolist(),
            arousal_curve=a_clipped.tolist(),
            dominance_curve=d.tolist(),
        )
