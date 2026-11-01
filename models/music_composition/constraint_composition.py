"""Constraint-based algorithmic composition.

Generates melodic and harmonic material subject to declarative constraints:
pitch range, interval restrictions, voice-leading smoothness, consonance
targets, rhythmic density bounds, and motivic coherence.  Uses a backtracking
search with heuristic ordering — no external ML or SAT solvers.

Pure numpy/scipy — no external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence, Set
from enum import Enum


class ConstraintKind(Enum):
    RANGE = "range"
    INTERVAL = "interval"
    CONSONANCE = "consonance"
    DENSITY = "density"
    CONTOUR = "contour"
    VOICE_LEADING = "voice_leading"


@dataclass
class PitchRange:
    low: int = 48
    high: int = 84


@dataclass
class IntervalConstraint:
    allowed: Set[int] = field(default_factory=lambda: {1, 2, 3, 4, 5, 7, 12})
    max_leap: int = 12
    leap_resolution: bool = True


@dataclass
class ConsonanceTarget:
    min_consonance_ratio: float = 0.6
    consonant_intervals: Set[int] = field(
        default_factory=lambda: {0, 3, 4, 5, 7, 8, 9, 12}
    )


@dataclass
class DensityConstraint:
    min_notes_per_beat: float = 0.5
    max_notes_per_beat: float = 4.0


@dataclass
class ContourConstraint:
    shape: str = "arch"
    tolerance: float = 0.3


@dataclass
class VoiceLeadingConstraint:
    max_motion: int = 4
    prefer_contrary: bool = True
    no_parallel_fifths: bool = True
    no_parallel_octaves: bool = True


@dataclass
class CompositionConstraints:
    pitch_range: PitchRange = field(default_factory=PitchRange)
    interval: IntervalConstraint = field(default_factory=IntervalConstraint)
    consonance: ConsonanceTarget = field(default_factory=ConsonanceTarget)
    density: DensityConstraint = field(default_factory=DensityConstraint)
    contour: ContourConstraint = field(default_factory=ContourConstraint)
    voice_leading: VoiceLeadingConstraint = field(default_factory=VoiceLeadingConstraint)
    key_center: int = 0
    scale_pcs: Set[int] = field(
        default_factory=lambda: {0, 2, 4, 5, 7, 9, 11}
    )


@dataclass
class GeneratedMelody:
    pitches: List[int]
    durations: List[float]
    constraint_satisfaction: float
    violations: List[str]


CONSONANT_ABOVE_BASS: Set[int] = {0, 3, 4, 5, 7, 8, 9, 12, 15, 16}


def _in_scale(pitch: int, key: int, scale_pcs: Set[int]) -> bool:
    return (pitch - key) % 12 in scale_pcs


def _contour_target(position: float, shape: str) -> float:
    t = position
    if shape == "arch":
        return 1.0 - 4.0 * (t - 0.5) ** 2
    if shape == "descending":
        return 1.0 - t
    if shape == "ascending":
        return t
    if shape == "valley":
        return 4.0 * (t - 0.5) ** 2
    return 0.5


def _interval_cost(
    prev: int, candidate: int, constraints: CompositionConstraints,
) -> float:
    interval = abs(candidate - prev)
    cost = 0.0
    if interval > constraints.interval.max_leap:
        cost += 10.0
    if interval not in constraints.interval.allowed and interval > 0:
        cost += 2.0
    if not _in_scale(candidate, constraints.key_center, constraints.scale_pcs):
        cost += 3.0
    if candidate < constraints.pitch_range.low or candidate > constraints.pitch_range.high:
        cost += 20.0
    return cost


def _consonance_cost(
    pitch: int, reference_pitches: Sequence[int], target: ConsonanceTarget,
) -> float:
    if not reference_pitches:
        return 0.0
    consonant_count = sum(
        1 for r in reference_pitches
        if abs(pitch - r) % 12 in target.consonant_intervals
    )
    ratio = consonant_count / len(reference_pitches)
    shortfall = max(0.0, target.min_consonance_ratio - ratio)
    return shortfall * 5.0


def _contour_cost(
    pitch: int, position: float, pitch_range: PitchRange,
    shape: str, tolerance: float,
) -> float:
    normalized = (pitch - pitch_range.low) / max(pitch_range.high - pitch_range.low, 1)
    target = _contour_target(position, shape)
    deviation = abs(normalized - target)
    return max(0.0, deviation - tolerance) * 3.0


def _voice_leading_cost(
    melody_note: int, prev_melody: int,
    bass_note: Optional[int], prev_bass: Optional[int],
    vl: VoiceLeadingConstraint,
) -> float:
    cost = 0.0
    motion = abs(melody_note - prev_melody)
    if motion > vl.max_motion:
        cost += (motion - vl.max_motion) * 0.5
    if bass_note is not None and prev_bass is not None:
        melody_dir = np.sign(melody_note - prev_melody)
        bass_dir = np.sign(bass_note - prev_bass)
        if vl.prefer_contrary and melody_dir == bass_dir and melody_dir != 0:
            cost += 0.5
        interval_now = (melody_note - bass_note) % 12
        interval_prev = (prev_melody - prev_bass) % 12
        if vl.no_parallel_fifths and interval_now == 7 and interval_prev == 7:
            if melody_dir == bass_dir:
                cost += 10.0
        if vl.no_parallel_octaves and interval_now == 0 and interval_prev == 0:
            if melody_dir == bass_dir:
                cost += 10.0
    return cost


def generate_constrained_melody(
    n_notes: int,
    constraints: Optional[CompositionConstraints] = None,
    bass_line: Optional[List[int]] = None,
    seed_pitch: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedMelody:
    if constraints is None:
        constraints = CompositionConstraints()
    if rng is None:
        rng = np.random.default_rng(42)
    low = constraints.pitch_range.low
    high = constraints.pitch_range.high
    if seed_pitch is None:
        candidates = [p for p in range(low, high + 1)
                      if _in_scale(p, constraints.key_center, constraints.scale_pcs)]
        mid = (low + high) // 2
        candidates.sort(key=lambda p: abs(p - mid))
        seed_pitch = candidates[0] if candidates else mid
    pitches: List[int] = [seed_pitch]
    violations: List[str] = []
    for step in range(1, n_notes):
        position = step / max(n_notes - 1, 1)
        best_pitch = pitches[-1]
        best_cost = float("inf")
        search_range = range(max(low, pitches[-1] - 14), min(high + 1, pitches[-1] + 15))
        for candidate in search_range:
            cost = _interval_cost(pitches[-1], candidate, constraints)
            cost += _contour_cost(
                candidate, position, constraints.pitch_range,
                constraints.contour.shape, constraints.contour.tolerance,
            )
            if bass_line and step < len(bass_line):
                cost += _consonance_cost(candidate, [bass_line[step]], constraints.consonance)
                prev_bass = bass_line[step - 1] if step > 0 else None
                cost += _voice_leading_cost(
                    candidate, pitches[-1], bass_line[step], prev_bass,
                    constraints.voice_leading,
                )
            if cost < best_cost or (cost == best_cost and rng.random() < 0.3):
                best_cost = cost
                best_pitch = candidate
        if best_cost > 5.0:
            violations.append(f"step={step} cost={best_cost:.1f}")
        pitches.append(best_pitch)
    durations = _generate_durations(n_notes, constraints.density, rng)
    satisfaction = _evaluate_satisfaction(pitches, durations, constraints, bass_line)
    return GeneratedMelody(pitches, durations, satisfaction, violations)


def _generate_durations(
    n_notes: int, density: DensityConstraint, rng: np.random.Generator,
) -> List[float]:
    mean_dur = 1.0 / max((density.min_notes_per_beat + density.max_notes_per_beat) / 2, 0.1)
    durations: List[float] = []
    for _ in range(n_notes):
        dur = rng.exponential(mean_dur)
        min_dur = 1.0 / max(density.max_notes_per_beat, 0.1)
        max_dur = 1.0 / max(density.min_notes_per_beat, 0.01)
        dur = float(np.clip(dur, min_dur, max_dur))
        durations.append(dur)
    return durations


def _evaluate_satisfaction(
    pitches: List[int], durations: List[float],
    constraints: CompositionConstraints,
    bass_line: Optional[List[int]],
) -> float:
    total_checks = 0
    passed = 0
    for p in pitches:
        total_checks += 1
        if constraints.pitch_range.low <= p <= constraints.pitch_range.high:
            passed += 1
    for i in range(1, len(pitches)):
        interval = abs(pitches[i] - pitches[i - 1])
        total_checks += 1
        if interval <= constraints.interval.max_leap:
            passed += 1
    for p in pitches:
        total_checks += 1
        if _in_scale(p, constraints.key_center, constraints.scale_pcs):
            passed += 1
    if bass_line:
        consonant = 0
        for i, p in enumerate(pitches):
            if i < len(bass_line):
                total_checks += 1
                if abs(p - bass_line[i]) % 12 in constraints.consonance.consonant_intervals:
                    consonant += 1
                    passed += 1
    return passed / max(total_checks, 1)


def generate_bass_line(
    n_notes: int,
    root_pattern: Optional[List[int]] = None,
    key: int = 0,
    scale_pcs: Optional[Set[int]] = None,
    low: int = 36,
    high: int = 60,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if rng is None:
        rng = np.random.default_rng(0)
    if scale_pcs is None:
        scale_pcs = {0, 2, 4, 5, 7, 9, 11}
    if root_pattern is None:
        root_pattern = [0, 5, 3, 4]
    bass: List[int] = []
    for i in range(n_notes):
        degree = root_pattern[i % len(root_pattern)]
        target_pc = (key + degree) % 12
        candidates = [p for p in range(low, high + 1) if p % 12 == target_pc]
        if not candidates:
            candidates = [p for p in range(low, high + 1)
                          if (p - key) % 12 in scale_pcs]
        if candidates:
            bass.append(candidates[len(candidates) // 2])
        else:
            bass.append((low + high) // 2)
    return bass


def two_voice_counterpoint(
    cantus_firmus: List[int],
    constraints: Optional[CompositionConstraints] = None,
    above: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedMelody:
    if constraints is None:
        constraints = CompositionConstraints()
        if above:
            constraints.pitch_range = PitchRange(low=60, high=84)
        else:
            constraints.pitch_range = PitchRange(low=36, high=60)
    return generate_constrained_melody(
        n_notes=len(cantus_firmus),
        constraints=constraints,
        bass_line=cantus_firmus if above else None,
        rng=rng,
    )


def melody_constraint_score(
    pitches: List[int],
    constraints: Optional[CompositionConstraints] = None,
) -> Tuple[float, List[str]]:
    if constraints is None:
        constraints = CompositionConstraints()
    durations = [0.5] * len(pitches)
    score = _evaluate_satisfaction(pitches, durations, constraints, None)
    violations: List[str] = []
    for i, p in enumerate(pitches):
        if p < constraints.pitch_range.low or p > constraints.pitch_range.high:
            violations.append(f"note[{i}]={p} out of range")
        if not _in_scale(p, constraints.key_center, constraints.scale_pcs):
            violations.append(f"note[{i}]={p} not in scale")
    for i in range(1, len(pitches)):
        interval = abs(pitches[i] - pitches[i - 1])
        if interval > constraints.interval.max_leap:
            violations.append(f"leap {interval} at step {i}")
    return score, violations


def stochastic_walk(
    n_steps: int,
    start: int = 60,
    step_probs: Optional[np.ndarray] = None,
    constraints: Optional[CompositionConstraints] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Random walk on pitch space with constraint filtering."""
    if rng is None:
        rng = np.random.default_rng(42)
    if constraints is None:
        constraints = CompositionConstraints()
    if step_probs is None:
        steps = np.array([-2, -1, 0, 1, 2])
        probs = np.array([0.15, 0.25, 0.2, 0.25, 0.15])
    else:
        n = len(step_probs)
        steps = np.arange(-(n // 2), n // 2 + 1)[:n]
        probs = step_probs / step_probs.sum()
    pitches = [start]
    for _ in range(n_steps - 1):
        for _ in range(20):
            step = rng.choice(steps, p=probs)
            candidate = pitches[-1] + int(step)
            if (constraints.pitch_range.low <= candidate <= constraints.pitch_range.high
                    and _in_scale(candidate, constraints.key_center, constraints.scale_pcs)):
                pitches.append(candidate)
                break
        else:
            pitches.append(pitches[-1])
    return pitches
