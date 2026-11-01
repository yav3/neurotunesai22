"""World rhythm systems: Aksak, Tala, Euclidean rhythms, timeline patterns."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TimelinePattern:
    name: str
    pattern: list[int]
    tradition: str = ""
    time_unit: str = "pulse"

    @property
    def length(self) -> int:
        return len(self.pattern)

    @property
    def density(self) -> float:
        if self.length == 0:
            return 0.0
        return sum(self.pattern) / self.length

    @property
    def onsets(self) -> list[int]:
        return [i for i, v in enumerate(self.pattern) if v == 1]

    def rotate(self, steps: int) -> "TimelinePattern":
        n = self.length
        if n == 0:
            return TimelinePattern(self.name, [], self.tradition)
        s = steps % n
        rotated = self.pattern[s:] + self.pattern[:s]
        return TimelinePattern(f"{self.name}_rot{steps}", rotated, self.tradition)

    def complement(self) -> "TimelinePattern":
        return TimelinePattern(
            f"{self.name}_complement",
            [1 - x for x in self.pattern],
            self.tradition,
        )

    def inter_onset_intervals(self) -> list[int]:
        ons = self.onsets
        if len(ons) < 2:
            return []
        ioi = []
        for i in range(len(ons) - 1):
            ioi.append(ons[i + 1] - ons[i])
        ioi.append(self.length - ons[-1] + ons[0])
        return ioi


STANDARD_TIMELINES: dict[str, TimelinePattern] = {
    "son_clave": TimelinePattern("Son Clave", [1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0], "Afro-Cuban"),
    "rumba_clave": TimelinePattern("Rumba Clave", [1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0], "Afro-Cuban"),
    "bossa_nova": TimelinePattern("Bossa Nova", [1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0], "Brazilian"),
    "gahu": TimelinePattern("Gahu", [1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0], "West African"),
    "soukous": TimelinePattern("Soukous", [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0], "Congolese"),
    "shiko": TimelinePattern("Shiko", [1,0,0,0,1,0,1,0,0,0,1,0], "West African"),
    "bembe": TimelinePattern("Bembe", [1,0,1,0,1,1,0,1,0,1,0,1], "West African", "eighth"),
}


class Aksak:
    PATTERNS: dict[str, list[int]] = {
        "7/8_a": [2, 2, 3],
        "7/8_b": [3, 2, 2],
        "7/8_c": [2, 3, 2],
        "9/8_a": [2, 2, 2, 3],
        "9/8_b": [2, 3, 2, 2],
        "11/8": [2, 2, 3, 2, 2],
        "5/8": [2, 3],
        "15/8": [2, 2, 2, 3, 2, 2, 2],
    }

    def get_pattern(self, name: str) -> list[int] | None:
        return self.PATTERNS.get(name)

    def total_pulses(self, name: str) -> int:
        p = self.get_pattern(name)
        return sum(p) if p else 0

    def to_binary(self, name: str) -> list[int]:
        p = self.get_pattern(name)
        if not p:
            return []
        binary = []
        for group in p:
            binary.append(1)
            binary.extend([0] * (group - 1))
        return binary

    def all_patterns(self) -> dict[str, list[int]]:
        return dict(self.PATTERNS)


@dataclass
class TalaCycle:
    name: str
    beats: list[int]
    clap_pattern: list[str]

    @property
    def total_beats(self) -> int:
        return sum(self.beats)


class Tala:
    CYCLES: dict[str, TalaCycle] = {
        "tintal": TalaCycle("Tintal", [4, 4, 4, 4], ["clap", "clap", "wave", "clap"]),
        "jhaptal": TalaCycle("Jhaptal", [2, 3, 2, 3], ["clap", "clap", "wave", "clap"]),
        "rupak": TalaCycle("Rupak", [3, 2, 2], ["wave", "clap", "clap"]),
        "ektal": TalaCycle("Ektal", [2, 2, 2, 2, 2, 2], ["clap", "clap", "wave", "clap", "wave", "clap"]),
        "dadra": TalaCycle("Dadra", [3, 3], ["clap", "wave"]),
        "kaherva": TalaCycle("Kaherva", [4, 4], ["clap", "wave"]),
        "adi": TalaCycle("Adi", [4, 2, 2], ["clap", "finger", "finger"]),
        "rupaka": TalaCycle("Rupaka", [2, 4], ["wave", "clap"]),
        "misra_chapu": TalaCycle("Misra Chapu", [3, 2, 2], ["clap", "finger", "finger"]),
    }

    def get_cycle(self, name: str) -> TalaCycle | None:
        return self.CYCLES.get(name)

    def beat_positions(self, name: str) -> list[int] | None:
        cycle = self.get_cycle(name)
        if not cycle:
            return None
        positions = []
        pos = 0
        for b in cycle.beats:
            positions.append(pos)
            pos += b
        return positions

    def sam_position(self, name: str) -> int:
        return 0

    def all_talas(self) -> dict[str, TalaCycle]:
        return dict(self.CYCLES)


class PolyrhythmAnalyzer:
    def lcm(self, a: int, b: int) -> int:
        from math import gcd
        return abs(a * b) // gcd(a, b)

    def cycle_length(self, periods: list[int]) -> int:
        if not periods:
            return 0
        result = periods[0]
        for p in periods[1:]:
            result = self.lcm(result, p)
        return result

    def coincidence_points(self, period_a: int, period_b: int) -> list[int]:
        cl = self.lcm(period_a, period_b)
        return [i for i in range(0, cl, cl // max(1, min(period_a, period_b)))
                if i % period_a == 0 and i % period_b == 0]

    def cross_rhythm_ratio(self, a: int, b: int) -> tuple[int, int]:
        from math import gcd
        g = gcd(a, b)
        return (a // g, b // g)

    def generate_polyrhythm(self, beats_a: int, beats_b: int, total_pulses: int) -> tuple[list[int], list[int]]:
        if beats_a <= 0 or beats_b <= 0 or total_pulses <= 0:
            return ([], [])
        pattern_a = [0] * total_pulses
        pattern_b = [0] * total_pulses
        for i in range(beats_a):
            idx = int(round(i * total_pulses / beats_a)) % total_pulses
            pattern_a[idx] = 1
        for i in range(beats_b):
            idx = int(round(i * total_pulses / beats_b)) % total_pulses
            pattern_b[idx] = 1
        return (pattern_a, pattern_b)


def euclidean_rhythm(pulses: int, steps: int) -> list[int]:
    if steps <= 0:
        return []
    if pulses <= 0:
        return [0] * steps
    if pulses >= steps:
        return [1] * steps
    pattern = [[1] for _ in range(pulses)] + [[0] for _ in range(steps - pulses)]
    while True:
        n_full = sum(1 for p in pattern if len(p) > 0)
        remainders = [p for p in pattern if len(p) < max(len(x) for x in pattern)]
        non_remainders = [p for p in pattern if len(p) == max(len(x) for x in pattern)]
        if len(remainders) <= 1 or len(non_remainders) == 0:
            break
        new_pattern = []
        for i in range(min(len(non_remainders), len(remainders))):
            new_pattern.append(non_remainders[i] + remainders[i])
        for i in range(min(len(non_remainders), len(remainders)), len(non_remainders)):
            new_pattern.append(non_remainders[i])
        for i in range(min(len(non_remainders), len(remainders)), len(remainders)):
            new_pattern.append(remainders[i])
        pattern = new_pattern
    result = []
    for p in pattern:
        result.extend(p)
    return result


def necklace_rotation(pattern: list[int]) -> list[list[int]]:
    n = len(pattern)
    if n == 0:
        return []
    rotations = []
    for i in range(n):
        rotations.append(pattern[i:] + pattern[:i])
    return rotations
