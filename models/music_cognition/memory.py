"""Musical memory: repetition detection, familiarity/preference curves."""
from __future__ import annotations
import numpy as np


class RepetitionDetector:
    def find_repetitions(
        self, sequence: list, min_length: int = 3, tolerance: int = 0,
    ) -> list[tuple[int, int, int]]:
        n = len(sequence)
        results = []
        for length in range(min_length, n // 2 + 1):
            for i in range(n - length):
                for j in range(i + length, n - length + 1):
                    mismatches = sum(
                        1 for k in range(length)
                        if _differ(sequence[i + k], sequence[j + k], tolerance)
                    )
                    if mismatches == 0:
                        results.append((i, j, length))
        seen = set()
        filtered = []
        for r in results:
            key = (r[0], r[1])
            if key not in seen:
                seen.add(key)
                filtered.append(r)
        return filtered

    def repetition_density(self, sequence: list, min_length: int = 3) -> float:
        reps = self.find_repetitions(sequence, min_length)
        if not reps or len(sequence) == 0:
            return 0.0
        covered = set()
        for start1, start2, length in reps:
            for k in range(length):
                covered.add(start1 + k)
                covered.add(start2 + k)
        return float(len(covered) / len(sequence))


class FamiliarityModel:
    def preference(self, exposure_count: int, peak: int = 10) -> float:
        if exposure_count <= 0:
            return 0.0
        x = exposure_count / peak
        return float(x * np.exp(1 - x))

    def preference_curve(self, max_exposure: int = 50, peak: int = 10) -> list[tuple[int, float]]:
        return [(i, self.preference(i, peak)) for i in range(max_exposure + 1)]

    def optimal_exposure(self, peak: int = 10) -> int:
        return peak


def find_repetitions(sequence: list, min_length: int = 3) -> list[tuple[int, int, int]]:
    return RepetitionDetector().find_repetitions(sequence, min_length)


def familiarity_preference(exposure_count: int, peak: int = 10) -> float:
    return FamiliarityModel().preference(exposure_count, peak)


def _differ(a, b, tolerance: int) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > tolerance
    return a != b
