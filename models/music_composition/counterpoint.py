"""Species counterpoint rules: parallel motion, consonance, voice independence."""
from __future__ import annotations

CONSONANT_INTERVALS = {0, 3, 4, 7, 8, 9, 12}
PERFECT_CONSONANCES = {0, 7, 12}


class FirstSpecies:
    def check_parallel_fifths(self, voice1: list[int], voice2: list[int]) -> list[int]:
        violations = []
        for i in range(1, min(len(voice1), len(voice2))):
            prev_int = abs(voice2[i - 1] - voice1[i - 1]) % 12
            curr_int = abs(voice2[i] - voice1[i]) % 12
            if prev_int == curr_int and curr_int in (7, 0):
                if (voice1[i] - voice1[i - 1]) == (voice2[i] - voice2[i - 1]):
                    violations.append(i)
        return violations

    def check_parallel_octaves(self, voice1: list[int], voice2: list[int]) -> list[int]:
        violations = []
        for i in range(1, min(len(voice1), len(voice2))):
            prev_int = abs(voice2[i - 1] - voice1[i - 1]) % 12
            curr_int = abs(voice2[i] - voice1[i]) % 12
            if prev_int == 0 and curr_int == 0:
                if (voice1[i] - voice1[i - 1]) == (voice2[i] - voice2[i - 1]):
                    violations.append(i)
        return violations

    def check_voice_crossing(self, voice1: list[int], voice2: list[int]) -> list[int]:
        crossings = []
        for i in range(min(len(voice1), len(voice2))):
            if voice1[i] > voice2[i]:
                crossings.append(i)
        return crossings

    def score(self, cantus: list[int], counterpoint: list[int]) -> float:
        n = min(len(cantus), len(counterpoint))
        if n == 0:
            return 0.0
        penalty = 0.0
        penalty += len(self.check_parallel_fifths(cantus, counterpoint)) * 0.2
        penalty += len(self.check_parallel_octaves(cantus, counterpoint)) * 0.2
        penalty += len(self.check_voice_crossing(cantus, counterpoint)) * 0.1
        contrary = contrary_motion_score(cantus, counterpoint)
        consonant_count = sum(
            1 for i in range(n) if is_consonant(abs(counterpoint[i] - cantus[i]) % 12)
        )
        return max(0.0, min(1.0, 0.4 * contrary + 0.6 * (consonant_count / n) - penalty))


class SecondSpecies:
    def check_weak_beat_dissonance(
        self, cantus: list[int], counterpoint: list[int],
    ) -> list[int]:
        violations = []
        for i in range(1, min(len(cantus), len(counterpoint)), 2):
            interval = abs(counterpoint[i] - cantus[i // 2 if i // 2 < len(cantus) else -1]) % 12
            if interval not in CONSONANT_INTERVALS:
                prev = counterpoint[i - 1] if i > 0 else counterpoint[i]
                nxt = counterpoint[i + 1] if i + 1 < len(counterpoint) else counterpoint[i]
                if not (min(prev, nxt) <= counterpoint[i] <= max(prev, nxt)):
                    violations.append(i)
        return violations


def check_parallel_motion(voice1: list[int], voice2: list[int]) -> list[int]:
    return FirstSpecies().check_parallel_fifths(voice1, voice2)


def is_consonant(interval: int) -> bool:
    return (abs(interval) % 12) in CONSONANT_INTERVALS


def contrary_motion_score(v1: list[int], v2: list[int]) -> float:
    n = min(len(v1), len(v2))
    if n < 2:
        return 0.0
    contrary = 0
    for i in range(1, n):
        d1 = v1[i] - v1[i - 1]
        d2 = v2[i] - v2[i - 1]
        if d1 != 0 and d2 != 0 and (d1 > 0) != (d2 > 0):
            contrary += 1
    return float(contrary / (n - 1))
