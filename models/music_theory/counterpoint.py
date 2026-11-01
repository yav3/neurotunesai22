"""Species counterpoint: cantus firmus validation, consonance/dissonance,
parallel-motion detection, and voice-leading error checking.

Based on Fux/Jeppesen rules.  All intervals are in semitones (MIDI).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class IntervalClass(Enum):
    PERFECT_CONSONANCE = auto()
    IMPERFECT_CONSONANCE = auto()
    DISSONANCE = auto()


PERFECT_CONSONANCES = frozenset({0, 7, 12})  # P1, P5, P8
IMPERFECT_CONSONANCES = frozenset({3, 4, 8, 9})  # m3, M3, m6, M6
CONSONANCES = PERFECT_CONSONANCES | IMPERFECT_CONSONANCES


def classify_interval(semitones: int) -> IntervalClass:
    st = abs(semitones) % 12
    if st in {0, 7}:
        return IntervalClass.PERFECT_CONSONANCE
    if st in {3, 4, 8, 9}:
        return IntervalClass.IMPERFECT_CONSONANCE
    return IntervalClass.DISSONANCE


def is_consonant(semitones: int) -> bool:
    return classify_interval(semitones) != IntervalClass.DISSONANCE


def is_step(semitones: int) -> bool:
    return abs(semitones) <= 2


class MotionType(Enum):
    PARALLEL = auto()
    SIMILAR = auto()
    CONTRARY = auto()
    OBLIQUE = auto()


def motion_type(lower_interval: int, upper_interval: int) -> MotionType:
    if lower_interval == 0 and upper_interval == 0:
        return MotionType.OBLIQUE
    if lower_interval == 0 or upper_interval == 0:
        return MotionType.OBLIQUE
    if lower_interval == upper_interval:
        return MotionType.PARALLEL
    s_lo = 1 if lower_interval > 0 else -1
    s_up = 1 if upper_interval > 0 else -1
    if s_lo == s_up:
        return MotionType.SIMILAR
    return MotionType.CONTRARY


# ── Cantus firmus ────────────────────────────────────────────────────

@dataclass
class CFError:
    position: int
    message: str


def validate_cantus_firmus(notes: list[int]) -> list[CFError]:
    """Check a cantus firmus against standard rules.

    Rules: predominantly stepwise, single climax, no repeated notes,
    starts and ends on the tonic (first note), final approach by step,
    range ≤ an octave (roughly).
    """
    errors: list[CFError] = []
    if len(notes) < 4:
        errors.append(CFError(0, "Too short (minimum 4 notes)"))
        return errors

    # No repeated consecutive notes
    for i in range(1, len(notes)):
        if notes[i] == notes[i - 1]:
            errors.append(CFError(i, "Repeated note"))

    # Single climax
    max_note = max(notes)
    climax_positions = [i for i, n in enumerate(notes) if n == max_note]
    if len(climax_positions) > 1:
        errors.append(CFError(climax_positions[1], "Multiple climaxes"))

    # Predominantly stepwise
    leaps = sum(1 for i in range(1, len(notes)) if abs(notes[i] - notes[i - 1]) > 2)
    step_ratio = 1.0 - leaps / max(len(notes) - 1, 1)
    if step_ratio < 0.5:
        errors.append(CFError(0, f"Too many leaps ({leaps}/{len(notes)-1})"))

    # Start and end on same pitch class
    if notes[0] % 12 != notes[-1] % 12:
        errors.append(CFError(len(notes) - 1, "Does not end on tonic"))

    # Final approach by step
    if len(notes) >= 2 and abs(notes[-1] - notes[-2]) > 2:
        errors.append(CFError(len(notes) - 1, "Final approach not stepwise"))

    # Range check (typically ≤ 10th = 16 semitones)
    rng = max(notes) - min(notes)
    if rng > 16:
        errors.append(CFError(0, f"Range too wide: {rng} semitones"))

    return errors


# ── First species ────────────────────────────────────────────────────

@dataclass
class VoiceLeadingError:
    position: int
    kind: str
    message: str


def check_first_species(
    cantus: list[int],
    counterpoint: list[int],
) -> list[VoiceLeadingError]:
    """Validate note-against-note (first species) counterpoint."""
    errors: list[VoiceLeadingError] = []
    n = min(len(cantus), len(counterpoint))
    if n < 2:
        return errors

    for i in range(n):
        interval = abs(counterpoint[i] - cantus[i])
        if not is_consonant(interval):
            errors.append(VoiceLeadingError(
                i, "dissonance",
                f"Dissonant interval {interval} semitones at beat {i}"))

    for i in range(1, n):
        lo_motion = cantus[i] - cantus[i - 1]
        hi_motion = counterpoint[i] - counterpoint[i - 1]
        mt = motion_type(lo_motion, hi_motion)

        prev_iv = abs(counterpoint[i - 1] - cantus[i - 1]) % 12
        curr_iv = abs(counterpoint[i] - cantus[i]) % 12

        if mt == MotionType.PARALLEL and curr_iv in (0, 7):
            errors.append(VoiceLeadingError(
                i, "parallel_perfects",
                f"Parallel {'unisons' if curr_iv == 0 else 'fifths'} at beat {i}"))

        if mt == MotionType.SIMILAR and curr_iv in (0, 7):
            errors.append(VoiceLeadingError(
                i, "hidden_perfects",
                f"Hidden {'octave' if curr_iv == 0 else 'fifth'} at beat {i}"))

        if counterpoint[i] < cantus[i] and counterpoint[i - 1] > cantus[i - 1]:
            errors.append(VoiceLeadingError(
                i, "voice_crossing", "Voice crossing at beat " + str(i)))
        elif counterpoint[i] > cantus[i] and counterpoint[i - 1] < cantus[i - 1]:
            errors.append(VoiceLeadingError(
                i, "voice_crossing", "Voice crossing at beat " + str(i)))

    return errors


# ── Second species ───────────────────────────────────────────────────

def check_second_species(
    cantus: list[int],
    counterpoint: list[int],
) -> list[VoiceLeadingError]:
    """Validate two-notes-against-one (second species).

    *counterpoint* has twice as many notes as *cantus* (two per beat).
    Strong beats must be consonant; weak beats may be dissonant only if
    approached and left by step (passing tone).
    """
    errors: list[VoiceLeadingError] = []
    n = len(cantus)
    if len(counterpoint) < 2 * n:
        errors.append(VoiceLeadingError(
            0, "length", "Counterpoint must have 2× as many notes as cantus"))
        return errors

    for beat in range(n):
        strong = counterpoint[2 * beat]
        iv_strong = abs(strong - cantus[beat])
        if not is_consonant(iv_strong):
            errors.append(VoiceLeadingError(
                2 * beat, "strong_dissonance",
                f"Dissonance on strong beat {beat}"))

        if 2 * beat + 1 < len(counterpoint):
            weak = counterpoint[2 * beat + 1]
            iv_weak = abs(weak - cantus[beat])
            if not is_consonant(iv_weak):
                prev_step = is_step(weak - strong)
                nxt = counterpoint[2 * beat + 2] if 2 * beat + 2 < len(counterpoint) else None
                nxt_step = is_step(nxt - weak) if nxt is not None else False
                if not (prev_step and nxt_step):
                    errors.append(VoiceLeadingError(
                        2 * beat + 1, "weak_dissonance",
                        f"Non-passing dissonance on weak beat {beat}"))

    return errors


# ── Quality scoring ──────────────────────────────────────────────────

def voice_leading_score(
    cantus: list[int],
    counterpoint: list[int],
) -> float:
    """Score 0–1: 1.0 = no errors; deductions per error type."""
    errors = check_first_species(cantus, counterpoint)
    if not errors:
        return 1.0
    penalty = 0.0
    for e in errors:
        if e.kind == "parallel_perfects":
            penalty += 0.15
        elif e.kind == "hidden_perfects":
            penalty += 0.05
        elif e.kind == "dissonance":
            penalty += 0.1
        elif e.kind == "voice_crossing":
            penalty += 0.08
        else:
            penalty += 0.05
    return max(0.0, 1.0 - penalty)
