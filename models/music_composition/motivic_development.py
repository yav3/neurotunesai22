"""Motivic development: thematic transformation, fragmentation, augmentation, inversion.

Implements classical techniques of motivic development: exact/tonal transposition,
inversion, retrograde, augmentation/diminution, fragmentation, sequence,
and motivic similarity via edit distance on interval sequences.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class TransformType(str, Enum):
    TRANSPOSE = "transpose"
    INVERT = "invert"
    RETROGRADE = "retrograde"
    AUGMENT = "augment"
    DIMINISH = "diminish"
    FRAGMENT = "fragment"
    SEQUENCE = "sequence"
    RETROGRADE_INVERSION = "retrograde_inversion"


@dataclass(frozen=True)
class Motive:
    pitches: Tuple[int, ...]      # MIDI pitches
    durations: Tuple[float, ...]  # in beats
    label: str = ""

    @property
    def intervals(self) -> Tuple[int, ...]:
        return tuple(self.pitches[i + 1] - self.pitches[i]
                     for i in range(len(self.pitches) - 1))

    @property
    def contour(self) -> Tuple[int, ...]:
        return tuple(
            1 if d > 0 else (-1 if d < 0 else 0)
            for d in self.intervals
        )


def transpose(motive: Motive, semitones: int) -> Motive:
    """Exact transposition by semitones."""
    return Motive(
        pitches=tuple(p + semitones for p in motive.pitches),
        durations=motive.durations,
        label=f"{motive.label}_T{semitones}",
    )


def invert(motive: Motive, axis: Optional[int] = None) -> Motive:
    """Interval inversion around axis (default: first pitch)."""
    ax = axis if axis is not None else motive.pitches[0]
    return Motive(
        pitches=tuple(2 * ax - p for p in motive.pitches),
        durations=motive.durations,
        label=f"{motive.label}_I",
    )


def retrograde(motive: Motive) -> Motive:
    """Reverse pitch and duration order."""
    return Motive(
        pitches=motive.pitches[::-1],
        durations=motive.durations[::-1],
        label=f"{motive.label}_R",
    )


def retrograde_inversion(motive: Motive, axis: Optional[int] = None) -> Motive:
    """Retrograde of the inversion."""
    inv = invert(motive, axis)
    return Motive(
        pitches=inv.pitches[::-1],
        durations=inv.durations[::-1],
        label=f"{motive.label}_RI",
    )


def augment(motive: Motive, factor: float = 2.0) -> Motive:
    """Rhythmic augmentation: multiply durations."""
    return Motive(
        pitches=motive.pitches,
        durations=tuple(d * factor for d in motive.durations),
        label=f"{motive.label}_aug{factor}",
    )


def diminish(motive: Motive, factor: float = 2.0) -> Motive:
    """Rhythmic diminution: divide durations."""
    return Motive(
        pitches=motive.pitches,
        durations=tuple(d / factor for d in motive.durations),
        label=f"{motive.label}_dim{factor}",
    )


def fragment(motive: Motive, start: int, length: int) -> Motive:
    """Extract a fragment from the motive."""
    end = min(start + length, len(motive.pitches))
    return Motive(
        pitches=motive.pitches[start:end],
        durations=motive.durations[start:end],
        label=f"{motive.label}_frag{start}_{length}",
    )


def sequence(motive: Motive, steps: List[int]) -> List[Motive]:
    """Generate a melodic sequence: repeat motive at each transposition level."""
    return [transpose(motive, s) for s in steps]


def tonal_transpose(motive: Motive, steps: int,
                    scale: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)) -> Motive:
    """Transpose within a diatonic scale by scale steps."""
    scale_arr = np.array(sorted(set(s % 12 for s in scale)))
    new_pitches = []
    for p in motive.pitches:
        pc = p % 12
        octave = p // 12
        idx = np.argmin(np.abs(scale_arr - pc))
        new_idx = (idx + steps) % len(scale_arr)
        oct_shift = (idx + steps) // len(scale_arr)
        new_pitches.append(int(scale_arr[new_idx] + (octave + oct_shift) * 12))
    return Motive(
        pitches=tuple(new_pitches),
        durations=motive.durations,
        label=f"{motive.label}_Td{steps}",
    )


def interval_edit_distance(a: Motive, b: Motive) -> int:
    """Edit distance on interval sequences (Levenshtein)."""
    sa, sb = a.intervals, b.intervals
    m, n = len(sa), len(sb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if sa[i - 1] == sb[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[m][n]


def contour_similarity(a: Motive, b: Motive) -> float:
    """Fraction of matching contour directions."""
    ca, cb = a.contour, b.contour
    if not ca or not cb:
        return 0.0
    min_len = min(len(ca), len(cb))
    matches = sum(1 for i in range(min_len) if ca[i] == cb[i])
    return matches / min_len


def develop(motive: Motive, transforms: List[TransformType],
            **kwargs) -> List[Motive]:
    """Apply a chain of transformations to a motive."""
    result = [motive]
    current = motive
    for t in transforms:
        if t == TransformType.TRANSPOSE:
            current = transpose(current, kwargs.get("semitones", 2))
        elif t == TransformType.INVERT:
            current = invert(current)
        elif t == TransformType.RETROGRADE:
            current = retrograde(current)
        elif t == TransformType.RETROGRADE_INVERSION:
            current = retrograde_inversion(current)
        elif t == TransformType.AUGMENT:
            current = augment(current, kwargs.get("factor", 2.0))
        elif t == TransformType.DIMINISH:
            current = diminish(current, kwargs.get("factor", 2.0))
        elif t == TransformType.FRAGMENT:
            current = fragment(current, kwargs.get("start", 0),
                               kwargs.get("length", 3))
        elif t == TransformType.SEQUENCE:
            seq = sequence(current, kwargs.get("steps", [0, 2, 4]))
            result.extend(seq[1:])
            current = seq[-1] if seq else current
            continue
        result.append(current)
    return result


def find_motive_occurrences(melody_pitches: np.ndarray,
                            motive: Motive,
                            max_transposition: int = 12) -> List[Tuple[int, int]]:
    """Find all transposed occurrences of motive in melody.

    Returns list of (start_index, transposition_semitones).
    """
    m_intervals = np.array(motive.intervals)
    if len(m_intervals) == 0:
        return []
    mel = melody_pitches.astype(int)
    mel_intervals = np.diff(mel)
    results = []
    for i in range(len(mel_intervals) - len(m_intervals) + 1):
        segment = mel_intervals[i:i + len(m_intervals)]
        if np.array_equal(segment, m_intervals):
            transposition = int(mel[i] - motive.pitches[0])
            if abs(transposition) <= max_transposition:
                results.append((i, transposition))
    return results
