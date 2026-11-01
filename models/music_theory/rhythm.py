from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeSignature:
    numerator: int
    denominator: int
    feel: str


COMMON_METERS: dict[str, TimeSignature] = {
    "4/4":  TimeSignature(4, 4, "simple"),
    "3/4":  TimeSignature(3, 4, "simple"),
    "2/4":  TimeSignature(2, 4, "simple"),
    "6/8":  TimeSignature(6, 8, "compound"),
    "9/8":  TimeSignature(9, 8, "compound"),
    "12/8": TimeSignature(12, 8, "compound"),
    "5/4":  TimeSignature(5, 4, "irregular"),
    "7/8":  TimeSignature(7, 8, "irregular"),
    "5/8":  TimeSignature(5, 8, "irregular"),
}


@dataclass
class RhythmPattern:
    beats: list[float] = field(default_factory=list)
    time_signature: TimeSignature = field(
        default_factory=lambda: COMMON_METERS["4/4"]
    )
    syncopation_score: float = 0.0
    groove_template: str = ""


GROOVE_TEMPLATES: dict[str, RhythmPattern] = {
    "straight": RhythmPattern(
        beats=[0.0, 0.25, 0.5, 0.75],
        time_signature=COMMON_METERS["4/4"],
        syncopation_score=0.0,
        groove_template="straight",
    ),
    "swing": RhythmPattern(
        beats=[0.0, 0.33, 0.5, 0.83],
        time_signature=COMMON_METERS["4/4"],
        syncopation_score=0.3,
        groove_template="swing",
    ),
    "shuffle": RhythmPattern(
        beats=[0.0, 0.33, 0.5, 0.83],
        time_signature=COMMON_METERS["12/8"],
        syncopation_score=0.25,
        groove_template="shuffle",
    ),
    "bossa_nova": RhythmPattern(
        beats=[0.0, 0.1875, 0.375, 0.5, 0.6875],
        time_signature=COMMON_METERS["4/4"],
        syncopation_score=0.5,
        groove_template="bossa_nova",
    ),
    "reggae": RhythmPattern(
        beats=[0.25, 0.75],
        time_signature=COMMON_METERS["4/4"],
        syncopation_score=0.7,
        groove_template="reggae",
    ),
    "waltz": RhythmPattern(
        beats=[0.0, 0.333, 0.667],
        time_signature=COMMON_METERS["3/4"],
        syncopation_score=0.0,
        groove_template="waltz",
    ),
    "march": RhythmPattern(
        beats=[0.0, 0.5],
        time_signature=COMMON_METERS["2/4"],
        syncopation_score=0.0,
        groove_template="march",
    ),
    "tresillo": RhythmPattern(
        beats=[0.0, 0.375, 0.75],
        time_signature=COMMON_METERS["4/4"],
        syncopation_score=0.6,
        groove_template="tresillo",
    ),
    "clave_3_2": RhythmPattern(
        beats=[0.0, 0.1875, 0.375, 0.625, 0.75],
        time_signature=COMMON_METERS["4/4"],
        syncopation_score=0.65,
        groove_template="clave_3_2",
    ),
    "clave_2_3": RhythmPattern(
        beats=[0.0, 0.25, 0.5, 0.6875, 0.875],
        time_signature=COMMON_METERS["4/4"],
        syncopation_score=0.6,
        groove_template="clave_2_3",
    ),
}


class RhythmAnalyzer:

    @staticmethod
    def detect_meter(onset_times: list[float]) -> TimeSignature:
        if len(onset_times) < 2:
            return COMMON_METERS["4/4"]
        intervals = [
            onset_times[i + 1] - onset_times[i]
            for i in range(len(onset_times) - 1)
        ]
        if not intervals:
            return COMMON_METERS["4/4"]
        mean_ioi = sum(intervals) / len(intervals)
        if mean_ioi <= 0:
            return COMMON_METERS["4/4"]
        ratios = [ioi / mean_ioi for ioi in intervals]
        triple_score = sum(1 for r in ratios if 0.6 < r < 0.8 or 1.3 < r < 1.7)
        duple_score = sum(1 for r in ratios if 0.85 < r < 1.15)
        if triple_score > duple_score:
            return COMMON_METERS["6/8"]
        return COMMON_METERS["4/4"]

    @staticmethod
    def syncopation_score(pattern: RhythmPattern) -> float:
        if not pattern.beats:
            return 0.0
        strong_positions = [0.0, 0.5]
        weak_positions = [0.25, 0.75]
        on_strong = sum(1 for b in pattern.beats if any(abs(b - s) < 0.05 for s in strong_positions))
        on_weak = sum(1 for b in pattern.beats if any(abs(b - w) < 0.05 for w in weak_positions))
        total = len(pattern.beats)
        if total == 0:
            return 0.0
        return on_weak / total

    @staticmethod
    def swing_ratio(pattern: RhythmPattern) -> float:
        if len(pattern.beats) < 3:
            return 1.0
        pairs: list[float] = []
        for i in range(0, len(pattern.beats) - 1, 2):
            gap1 = pattern.beats[i + 1] - pattern.beats[i]
            if i + 2 < len(pattern.beats):
                gap2 = pattern.beats[i + 2] - pattern.beats[i + 1]
                if gap2 > 0:
                    pairs.append(gap1 / gap2)
        if not pairs:
            return 1.0
        return sum(pairs) / len(pairs)

    @staticmethod
    def groove_similarity(pattern1: RhythmPattern, pattern2: RhythmPattern) -> float:
        if not pattern1.beats or not pattern2.beats:
            return 0.0
        n = max(len(pattern1.beats), len(pattern2.beats))
        b1 = pattern1.beats + [1.0] * (n - len(pattern1.beats))
        b2 = pattern2.beats + [1.0] * (n - len(pattern2.beats))
        dist = sum((a - b) ** 2 for a, b in zip(b1, b2))
        return max(0.0, 1.0 - math.sqrt(dist / n))
