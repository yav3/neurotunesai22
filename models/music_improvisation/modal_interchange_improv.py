"""Modal interchange techniques for improvisation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Mode(Enum):
    IONIAN = "ionian"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    MELODIC_MINOR = "melodic_minor"
    HARMONIC_MINOR = "harmonic_minor"
    LYDIAN_DOMINANT = "lydian_dominant"
    ALTERED = "altered"


# Intervals from root (semitones) for each mode
MODE_INTERVALS: dict[Mode, tuple[int, ...]] = {
    Mode.IONIAN: (0, 2, 4, 5, 7, 9, 11),
    Mode.DORIAN: (0, 2, 3, 5, 7, 9, 10),
    Mode.PHRYGIAN: (0, 1, 3, 5, 7, 8, 10),
    Mode.LYDIAN: (0, 2, 4, 6, 7, 9, 11),
    Mode.MIXOLYDIAN: (0, 2, 4, 5, 7, 9, 10),
    Mode.AEOLIAN: (0, 2, 3, 5, 7, 8, 10),
    Mode.LOCRIAN: (0, 1, 3, 5, 6, 8, 10),
    Mode.MELODIC_MINOR: (0, 2, 3, 5, 7, 9, 11),
    Mode.HARMONIC_MINOR: (0, 2, 3, 5, 7, 8, 11),
    Mode.LYDIAN_DOMINANT: (0, 2, 4, 6, 7, 9, 10),
    Mode.ALTERED: (0, 1, 3, 4, 6, 8, 10),
}


def mode_pitch_classes(root: int, mode: Mode) -> set[int]:
    """Return the pitch-class set (mod 12) for a given root and mode."""
    return {(root + iv) % 12 for iv in MODE_INTERVALS[mode]}


def common_tones(root_a: int, mode_a: Mode, root_b: int, mode_b: Mode) -> set[int]:
    """Pitch classes shared between two modes."""
    return mode_pitch_classes(root_a, mode_a) & mode_pitch_classes(root_b, mode_b)


def mode_distance(root_a: int, mode_a: Mode, root_b: int, mode_b: Mode) -> int:
    """Number of pitch-class differences between two modes (0–7)."""
    pcs_a = mode_pitch_classes(root_a, mode_a)
    pcs_b = mode_pitch_classes(root_b, mode_b)
    return len(pcs_a.symmetric_difference(pcs_b))


@dataclass
class InterchangeOption:
    """A modal interchange candidate."""
    target_mode: Mode
    common_tone_count: int
    distance: int
    borrowed_pcs: set[int]

    @property
    def smoothness(self) -> float:
        """Higher = smoother transition (fewer borrowed notes)."""
        return self.common_tone_count / 7.0


def find_interchange_options(
    root: int,
    current_mode: Mode,
    candidates: list[Mode] | None = None,
) -> list[InterchangeOption]:
    """Find modal interchange options ranked by smoothness."""
    if candidates is None:
        candidates = [m for m in Mode if m != current_mode]
    current_pcs = mode_pitch_classes(root, current_mode)
    options = []
    for m in candidates:
        target_pcs = mode_pitch_classes(root, m)
        ct = len(current_pcs & target_pcs)
        dist = len(current_pcs.symmetric_difference(target_pcs))
        borrowed = target_pcs - current_pcs
        options.append(InterchangeOption(
            target_mode=m,
            common_tone_count=ct,
            distance=dist,
            borrowed_pcs=borrowed,
        ))
    return sorted(options, key=lambda o: -o.smoothness)


def borrowed_chord_tones(
    root: int,
    chord_root_degree: int,
    source_mode: Mode,
) -> list[int]:
    """Return chord tones (1, 3, 5, 7) built on a degree of a borrowed mode."""
    intervals = MODE_INTERVALS[source_mode]
    base = (root + intervals[chord_root_degree % 7]) % 12
    return [
        base,
        (base + intervals[(chord_root_degree + 2) % 7] - intervals[chord_root_degree % 7]) % 12,
        (base + intervals[(chord_root_degree + 4) % 7] - intervals[chord_root_degree % 7]) % 12,
    ]


def pivot_note(root: int, from_mode: Mode, to_mode: Mode) -> int | None:
    """Find the strongest pivot note (common tone closest to root)."""
    ct = common_tones(root, from_mode, root, to_mode)
    if not ct:
        return None
    return min(ct, key=lambda pc: (pc - root) % 12)


def interchange_tension(root: int, from_mode: Mode, to_mode: Mode) -> float:
    """Tension level of the interchange in [0, 1]. Higher = more chromatic tension."""
    dist = mode_distance(root, from_mode, root, to_mode)
    return dist / 12.0


@dataclass
class InterchangeSequence:
    """A planned sequence of modal interchanges."""
    root: int
    modes: list[Mode]
    durations_beats: list[float]

    @property
    def total_beats(self) -> float:
        return sum(self.durations_beats)

    @property
    def n_changes(self) -> int:
        return max(0, len(self.modes) - 1)

    def tension_profile(self) -> list[float]:
        """Tension at each mode change."""
        tensions = []
        for i in range(1, len(self.modes)):
            tensions.append(interchange_tension(self.root, self.modes[i - 1], self.modes[i]))
        return tensions


def build_interchange_sequence(
    root: int,
    start_mode: Mode,
    n_sections: int,
    beats_per_section: float = 8.0,
    max_distance: int = 4,
    rng: np.random.Generator | None = None,
) -> InterchangeSequence:
    """Build a modal interchange sequence preferring smooth transitions."""
    if rng is None:
        rng = np.random.default_rng()
    modes = [start_mode]
    for _ in range(n_sections - 1):
        options = find_interchange_options(root, modes[-1])
        smooth = [o for o in options if o.distance <= max_distance]
        if not smooth:
            smooth = options[:3]
        chosen = smooth[int(rng.integers(0, len(smooth)))]
        modes.append(chosen.target_mode)
    durations = [beats_per_section] * n_sections
    return InterchangeSequence(root=root, modes=modes, durations_beats=durations)
