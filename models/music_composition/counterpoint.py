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


# ---------------------------------------------------------------------------
# Motion classification
# ---------------------------------------------------------------------------

class MotionType:
    PARALLEL = "parallel"
    SIMILAR = "similar"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"


def classify_motion(d1: int, d2: int) -> str:
    if d1 == 0 and d2 == 0:
        return MotionType.OBLIQUE
    if d1 == 0 or d2 == 0:
        return MotionType.OBLIQUE
    if d1 == d2:
        return MotionType.PARALLEL
    if (d1 > 0) == (d2 > 0):
        return MotionType.SIMILAR
    return MotionType.CONTRARY


def motion_profile(v1: list[int], v2: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {MotionType.PARALLEL: 0, MotionType.SIMILAR: 0,
                               MotionType.CONTRARY: 0, MotionType.OBLIQUE: 0}
    for i in range(1, min(len(v1), len(v2))):
        m = classify_motion(v1[i] - v1[i - 1], v2[i] - v2[i - 1])
        counts[m] += 1
    return counts


# ---------------------------------------------------------------------------
# Counterpoint generation (first species, brute force)
# ---------------------------------------------------------------------------

import numpy as np


def generate_first_species(cantus: list[int], above: bool = True,
                           rng: np.random.RandomState | None = None) -> list[int]:
    """Generate valid first-species counterpoint via greedy search."""
    if rng is None:
        rng = np.random.RandomState()
    cp = []
    for i, cf_note in enumerate(cantus):
        candidates = []
        for offset in sorted(CONSONANT_INTERVALS):
            pitch = cf_note + offset if above else cf_note - offset
            if 48 <= pitch <= 84:
                candidates.append(pitch)
        if i > 0 and cp:
            prev = cp[-1]
            candidates.sort(key=lambda p: abs(p - prev))
        best = None
        best_score = -1.0
        for c in candidates[:8]:
            trial = cp + [c]
            if i > 0:
                d1 = cantus[i] - cantus[i - 1]
                d2 = c - cp[-1]
                interval = abs(c - cf_note) % 12
                if interval in PERFECT_CONSONANCES and classify_motion(d1, d2) == MotionType.PARALLEL:
                    continue
            step = abs(c - cp[-1]) if cp else 0
            score = 1.0 - step / 24.0
            if score > best_score:
                best_score = score
                best = c
        cp.append(best if best is not None else candidates[0])
    return cp


# ---------------------------------------------------------------------------
# Third and fourth species stubs
# ---------------------------------------------------------------------------

class ThirdSpecies:
    """Third species: four notes against one. Passing and neighbor tones."""

    def check_dissonance_resolution(self, cantus: list[int],
                                     counterpoint: list[int]) -> list[int]:
        violations = []
        for i in range(len(counterpoint)):
            cf_idx = i // 4
            if cf_idx >= len(cantus):
                break
            interval = abs(counterpoint[i] - cantus[cf_idx]) % 12
            beat_pos = i % 4
            if interval not in CONSONANT_INTERVALS and beat_pos in (0, 2):
                violations.append(i)
        return violations


class FourthSpecies:
    """Fourth species: suspensions (syncopated counterpoint)."""

    @staticmethod
    def find_suspensions(cantus: list[int], counterpoint: list[int]) -> list[dict]:
        suspensions = []
        for i in range(1, min(len(cantus), len(counterpoint)) - 1):
            prep_int = abs(counterpoint[i - 1] - cantus[i - 1]) % 12
            sus_int = abs(counterpoint[i - 1] - cantus[i]) % 12
            res_int = abs(counterpoint[i] - cantus[i]) % 12
            if prep_int in CONSONANT_INTERVALS and sus_int not in CONSONANT_INTERVALS and res_int in CONSONANT_INTERVALS:
                suspensions.append({"position": i, "suspension_interval": sus_int,
                                    "resolution_interval": res_int})
        return suspensions
