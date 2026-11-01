"""Voice leading analysis and optimization for arrangement."""
from __future__ import annotations

from enum import Enum


class VoiceLeadingRule(Enum):
    PARALLEL_FIFTHS = "parallel_fifths"
    PARALLEL_OCTAVES = "parallel_octaves"
    HIDDEN_FIFTHS = "hidden_fifths"
    VOICE_CROSSING = "voice_crossing"
    LARGE_LEAP = "large_leap"
    UNRESOLVED_SEVENTH = "unresolved_seventh"


def check_parallel_motion(
    voice1: list[int], voice2: list[int],
) -> list[tuple[int, VoiceLeadingRule]]:
    """Detect parallel 5ths and 8ves between two voices."""
    violations = []
    for i in range(len(voice1) - 1):
        if i >= len(voice2) - 1:
            break
        interval_a = abs(voice1[i] - voice2[i]) % 12
        interval_b = abs(voice1[i + 1] - voice2[i + 1]) % 12
        motion1 = voice1[i + 1] - voice1[i]
        motion2 = voice2[i + 1] - voice2[i]
        same_direction = (motion1 > 0 and motion2 > 0) or (motion1 < 0 and motion2 < 0)
        if not same_direction:
            continue
        if interval_a == 7 and interval_b == 7:
            violations.append((i, VoiceLeadingRule.PARALLEL_FIFTHS))
        elif interval_a == 0 and interval_b == 0 and (motion1 != 0 or motion2 != 0):
            violations.append((i, VoiceLeadingRule.PARALLEL_OCTAVES))
    return violations


def voice_crossing_check(
    soprano: list[int], alto: list[int], tenor: list[int], bass: list[int],
) -> list[tuple[int, str, str]]:
    """Detect voice crossings at each beat index."""
    voices = {"soprano": soprano, "alto": alto, "tenor": tenor, "bass": bass}
    order = ["soprano", "alto", "tenor", "bass"]
    crossings = []
    n = min(len(v) for v in voices.values())
    for i in range(n):
        for j in range(len(order) - 1):
            upper = voices[order[j]][i]
            lower = voices[order[j + 1]][i]
            if upper < lower:
                crossings.append((i, order[j], order[j + 1]))
    return crossings


def smooth_voice_leading(
    from_chord: list[int], to_chord_pcs: list[int],
) -> list[int]:
    """Find voicing of to_chord_pcs minimizing total semitone motion from from_chord."""
    if not from_chord or not to_chord_pcs:
        return list(to_chord_pcs)
    result = []
    used = set()
    for src in from_chord:
        best_pitch = None
        best_dist = 999
        for pc in to_chord_pcs:
            for octave_shift in range(-1, 2):
                candidate = pc + (src // 12) * 12 + octave_shift * 12
                dist = abs(candidate - src)
                key = (pc, octave_shift)
                if dist < best_dist and key not in used:
                    best_dist = dist
                    best_pitch = candidate
                    best_key = key
        if best_pitch is not None:
            result.append(best_pitch)
            used.add(best_key)
        else:
            result.append(to_chord_pcs[0] + (src // 12) * 12)
    return result


def voice_leading_cost_matrix(
    chord_a: list[int], chord_b: list[int],
) -> list[list[int]]:
    """Pairwise absolute semitone distances between all voice pairs."""
    return [[abs(a - b) for b in chord_b] for a in chord_a]


def optimal_voicing(
    chord_pcs: list[int],
    prev_voicing: list[int],
    range_low: int = 36,
    range_high: int = 84,
) -> list[int]:
    """Find voicing of pitch classes minimizing motion from previous voicing."""
    candidates = []
    for pc in chord_pcs:
        octave_options = []
        p = pc % 12
        for midi in range(range_low, range_high + 1):
            if midi % 12 == p:
                octave_options.append(midi)
        candidates.append(octave_options)
    if not candidates or not prev_voicing:
        return [c[0] if c else 0 for c in candidates]
    best = None
    best_cost = float("inf")
    from itertools import product
    for combo in product(*candidates):
        cost = sum(abs(c - p) for c, p in zip(combo, prev_voicing[:len(combo)]))
        if cost < best_cost:
            best_cost = cost
            best = list(combo)
    return best or [c[0] if c else 0 for c in candidates]


def total_voice_motion(progression: list[list[int]]) -> float:
    """Sum of absolute semitone motion across all voices for a chord progression."""
    if len(progression) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(progression)):
        for a, b in zip(progression[i - 1], progression[i]):
            total += abs(b - a)
    return total
