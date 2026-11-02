"""Dynamic markings: symbolic dynamics representation and interpretation."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class DynamicLevel(enum.IntEnum):
    PPP = 0
    PP = 1
    P = 2
    MP = 3
    MF = 4
    F = 5
    FF = 6
    FFF = 7


DYNAMIC_VELOCITY_RANGES: dict[DynamicLevel, tuple[int, int]] = {
    DynamicLevel.PPP: (1, 15),
    DynamicLevel.PP: (16, 33),
    DynamicLevel.P: (34, 49),
    DynamicLevel.MP: (50, 64),
    DynamicLevel.MF: (65, 80),
    DynamicLevel.F: (81, 96),
    DynamicLevel.FF: (97, 112),
    DynamicLevel.FFF: (113, 127),
}

DYNAMIC_NAMES: dict[DynamicLevel, str] = {
    DynamicLevel.PPP: "ppp",
    DynamicLevel.PP: "pp",
    DynamicLevel.P: "p",
    DynamicLevel.MP: "mp",
    DynamicLevel.MF: "mf",
    DynamicLevel.F: "f",
    DynamicLevel.FF: "ff",
    DynamicLevel.FFF: "fff",
}


@dataclass
class DynamicMarking:
    """A dynamic marking at a position in the score."""
    level: DynamicLevel
    position_beats: float
    is_hairpin_start: bool = False
    is_hairpin_end: bool = False


def dynamic_to_velocity_range(level: DynamicLevel) -> tuple[int, int]:
    """Return the canonical velocity range for a dynamic level."""
    return DYNAMIC_VELOCITY_RANGES[level]


def dynamic_to_velocity(level: DynamicLevel) -> int:
    """Return the midpoint velocity for a dynamic level."""
    lo, hi = DYNAMIC_VELOCITY_RANGES[level]
    return (lo + hi) // 2


def velocity_to_dynamic(velocity: int) -> DynamicLevel:
    """Find the nearest dynamic level for a MIDI velocity."""
    v = max(1, min(127, velocity))
    best = DynamicLevel.MF
    best_dist = 999
    for level, (lo, hi) in DYNAMIC_VELOCITY_RANGES.items():
        mid = (lo + hi) // 2
        dist = abs(v - mid)
        if dist < best_dist:
            best_dist = dist
            best = level
    return best


def interpret_hairpin(
    start_level: DynamicLevel,
    end_level: DynamicLevel,
    n_notes: int,
) -> list[int]:
    """Interpret a hairpin (crescendo/diminuendo) as a velocity ramp."""
    if n_notes <= 0:
        return []
    if n_notes == 1:
        return [dynamic_to_velocity(start_level)]
    start_vel = dynamic_to_velocity(start_level)
    end_vel = dynamic_to_velocity(end_level)
    return [
        int(round(start_vel + (end_vel - start_vel) * i / (n_notes - 1)))
        for i in range(n_notes)
    ]


def sforzando(base_velocity: int, sf_boost: int = 40) -> int:
    """Single accented note (sforzando)."""
    return min(127, base_velocity + sf_boost)


def subito_change(current_vel: int, target_level: DynamicLevel) -> int:
    """Immediate dynamic shift (subito piano, subito forte, etc.)."""
    return dynamic_to_velocity(target_level)


def parse_dynamic_sequence(
    markings: list[DynamicMarking],
    total_beats: float,
    resolution: float = 0.25,
) -> list[int]:
    """Produce a velocity curve from a sequence of dynamic markings.

    Args:
        markings: sorted list of DynamicMarking by position_beats.
        total_beats: total duration in beats.
        resolution: time step in beats.

    Returns:
        List of velocities, one per resolution step.
    """
    if not markings:
        n_steps = max(1, int(total_beats / resolution))
        return [dynamic_to_velocity(DynamicLevel.MF)] * n_steps

    sorted_marks = sorted(markings, key=lambda m: m.position_beats)
    n_steps = max(1, int(total_beats / resolution))
    velocities = [0] * n_steps

    for step in range(n_steps):
        t = step * resolution
        prev_mark = sorted_marks[0]
        next_mark = sorted_marks[-1]
        for i, m in enumerate(sorted_marks):
            if m.position_beats <= t:
                prev_mark = m
                if i + 1 < len(sorted_marks):
                    next_mark = sorted_marks[i + 1]
                else:
                    next_mark = m

        if prev_mark is next_mark or next_mark.position_beats <= prev_mark.position_beats:
            velocities[step] = dynamic_to_velocity(prev_mark.level)
        elif prev_mark.is_hairpin_start:
            span = next_mark.position_beats - prev_mark.position_beats
            frac = (t - prev_mark.position_beats) / span
            frac = max(0.0, min(1.0, frac))
            v_start = dynamic_to_velocity(prev_mark.level)
            v_end = dynamic_to_velocity(next_mark.level)
            velocities[step] = int(round(v_start + (v_end - v_start) * frac))
        else:
            velocities[step] = dynamic_to_velocity(prev_mark.level)

    return velocities
