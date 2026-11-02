from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RussellEmotion:
    name: str
    valence: float  # -1 to 1
    arousal: float  # -1 to 1
    angle_degrees: float


_STANDARD_EMOTIONS: list[RussellEmotion] = [
    RussellEmotion("happy",    0.8,  0.6,  math.degrees(math.atan2(0.6, 0.8))),
    RussellEmotion("excited",  0.5,  0.9,  math.degrees(math.atan2(0.9, 0.5))),
    RussellEmotion("tense",   -0.3,  0.8,  math.degrees(math.atan2(0.8, -0.3))),
    RussellEmotion("angry",   -0.7,  0.6,  math.degrees(math.atan2(0.6, -0.7))),
    RussellEmotion("sad",     -0.7, -0.5,  math.degrees(math.atan2(-0.5, -0.7))),
    RussellEmotion("bored",   -0.3, -0.7,  math.degrees(math.atan2(-0.7, -0.3))),
    RussellEmotion("calm",     0.5, -0.6,  math.degrees(math.atan2(-0.6, 0.5))),
    RussellEmotion("relaxed",  0.7, -0.3,  math.degrees(math.atan2(-0.3, 0.7))),
    RussellEmotion("content",  0.6,  0.0,  math.degrees(math.atan2(0.0, 0.6))),
    RussellEmotion("serene",   0.4, -0.5,  math.degrees(math.atan2(-0.5, 0.4))),
    RussellEmotion("alert",    0.0,  0.8,  math.degrees(math.atan2(0.8, 0.0))),
    RussellEmotion("fatigued",-0.5, -0.8,  math.degrees(math.atan2(-0.8, -0.5))),
]


class RussellCircumplex:
    def __init__(self) -> None:
        self.emotions = list(_STANDARD_EMOTIONS)

    def nearest_emotion(self, valence: float, arousal: float) -> RussellEmotion:
        best: RussellEmotion | None = None
        best_dist = float("inf")
        for e in self.emotions:
            d = math.hypot(valence - e.valence, arousal - e.arousal)
            if d < best_dist:
                best_dist = d
                best = e
        assert best is not None
        return best

    @staticmethod
    def emotion_vector(valence: float, arousal: float) -> tuple[float, float]:
        angle = math.degrees(math.atan2(arousal, valence))
        magnitude = math.hypot(valence, arousal)
        return (angle, magnitude)

    @staticmethod
    def emotional_distance(e1: RussellEmotion, e2: RussellEmotion) -> float:
        return math.hypot(e1.valence - e2.valence, e1.arousal - e2.arousal)

    def emotional_trajectory(
        self, va_sequence: list[tuple[float, float]],
    ) -> dict:
        if not va_sequence:
            return {
                "total_distance": 0.0,
                "mean_valence": 0.0,
                "mean_arousal": 0.0,
                "valence_range": 0.0,
                "arousal_range": 0.0,
                "dominant_quadrant": "low_arousal_low_valence",
            }
        vs = [p[0] for p in va_sequence]
        ars = [p[1] for p in va_sequence]
        total_dist = sum(
            math.hypot(
                va_sequence[i + 1][0] - va_sequence[i][0],
                va_sequence[i + 1][1] - va_sequence[i][1],
            )
            for i in range(len(va_sequence) - 1)
        )
        mean_v = sum(vs) / len(vs)
        mean_a = sum(ars) / len(ars)
        a_label = "high_arousal" if mean_a >= 0.0 else "low_arousal"
        v_label = "high_valence" if mean_v >= 0.0 else "low_valence"
        return {
            "total_distance": total_dist,
            "mean_valence": mean_v,
            "mean_arousal": mean_a,
            "valence_range": max(vs) - min(vs),
            "arousal_range": max(ars) - min(ars),
            "dominant_quadrant": f"{a_label}_{v_label}",
        }
