"""Jazz harmony: ii-V-I detection, tritone substitution, shell/rootless voicings, jazz modes.

Pure numpy — no external ML frameworks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class JazzChordType(str, Enum):
    MAJ7 = "maj7"
    DOM7 = "dom7"
    MIN7 = "min7"
    HALF_DIM = "half_dim"
    DIM7 = "dim7"
    MIN_MAJ7 = "min_maj7"
    AUG7 = "aug7"
    SUS4 = "sus4"
    ALT = "alt"


CHORD_INTERVALS: dict[JazzChordType, tuple[int, ...]] = {
    JazzChordType.MAJ7: (0, 4, 7, 11),
    JazzChordType.DOM7: (0, 4, 7, 10),
    JazzChordType.MIN7: (0, 3, 7, 10),
    JazzChordType.HALF_DIM: (0, 3, 6, 10),
    JazzChordType.DIM7: (0, 3, 6, 9),
    JazzChordType.MIN_MAJ7: (0, 3, 7, 11),
    JazzChordType.AUG7: (0, 4, 8, 10),
    JazzChordType.SUS4: (0, 5, 7, 10),
    JazzChordType.ALT: (0, 4, 8, 10),
}


class JazzMode(str, Enum):
    IONIAN = "ionian"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    MELODIC_MINOR = "melodic_minor"
    LYDIAN_DOMINANT = "lydian_dominant"
    ALTERED = "altered"
    WHOLE_TONE = "whole_tone"
    DIMINISHED_HW = "diminished_hw"
    DIMINISHED_WH = "diminished_wh"
    BEBOP_DOMINANT = "bebop_dominant"
    BLUES = "blues"


MODE_INTERVALS: dict[JazzMode, tuple[int, ...]] = {
    JazzMode.IONIAN: (0, 2, 4, 5, 7, 9, 11),
    JazzMode.DORIAN: (0, 2, 3, 5, 7, 9, 10),
    JazzMode.PHRYGIAN: (0, 1, 3, 5, 7, 8, 10),
    JazzMode.LYDIAN: (0, 2, 4, 6, 7, 9, 11),
    JazzMode.MIXOLYDIAN: (0, 2, 4, 5, 7, 9, 10),
    JazzMode.AEOLIAN: (0, 2, 3, 5, 7, 8, 10),
    JazzMode.LOCRIAN: (0, 1, 3, 5, 6, 8, 10),
    JazzMode.MELODIC_MINOR: (0, 2, 3, 5, 7, 9, 11),
    JazzMode.LYDIAN_DOMINANT: (0, 2, 4, 6, 7, 9, 10),
    JazzMode.ALTERED: (0, 1, 3, 4, 6, 8, 10),
    JazzMode.WHOLE_TONE: (0, 2, 4, 6, 8, 10),
    JazzMode.DIMINISHED_HW: (0, 1, 3, 4, 6, 7, 9, 10),
    JazzMode.DIMINISHED_WH: (0, 2, 3, 5, 6, 8, 9, 11),
    JazzMode.BEBOP_DOMINANT: (0, 2, 4, 5, 7, 9, 10, 11),
    JazzMode.BLUES: (0, 3, 5, 6, 7, 10),
}


@dataclass(frozen=True)
class JazzChord:
    root: int
    chord_type: JazzChordType
    extensions: tuple[int, ...] = ()

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        base = tuple((self.root + i) % 12 for i in CHORD_INTERVALS[self.chord_type])
        ext = tuple((self.root + e) % 12 for e in self.extensions)
        return base + ext


@dataclass(frozen=True)
class IIVIProgression:
    key: int
    ii_root: int
    v_root: int
    i_root: int
    is_minor: bool = False
    has_tritone_sub: bool = False


@dataclass(frozen=True)
class VoicingResult:
    pitches: tuple[int, ...]
    voicing_type: str
    spread: int


def detect_ii_v_i(
    roots: list[int],
    chord_types: list[JazzChordType],
) -> list[IIVIProgression]:
    if len(roots) < 3 or len(chord_types) < 3:
        return []
    results: list[IIVIProgression] = []
    for i in range(len(roots) - 2):
        r0, r1, r2 = roots[i], roots[i + 1], roots[i + 2]
        t0, t1, t2 = chord_types[i], chord_types[i + 1], chord_types[i + 2]
        # major ii-V-I: min7 → dom7 → maj7, roots descend by 5ths
        if (t0 == JazzChordType.MIN7
                and t1 == JazzChordType.DOM7
                and t2 == JazzChordType.MAJ7
                and (r1 - r0) % 12 == 5
                and (r2 - r1) % 12 == 5):
            results.append(IIVIProgression(
                key=r2, ii_root=r0, v_root=r1, i_root=r2, is_minor=False,
            ))
        # minor ii-V-i: half_dim → dom7 → min7
        if (t0 == JazzChordType.HALF_DIM
                and t1 == JazzChordType.DOM7
                and t2 == JazzChordType.MIN7
                and (r1 - r0) % 12 == 5
                and (r2 - r1) % 12 == 5):
            results.append(IIVIProgression(
                key=r2, ii_root=r0, v_root=r1, i_root=r2, is_minor=True,
            ))
    return results


def tritone_substitute(root: int, chord_type: JazzChordType = JazzChordType.DOM7) -> JazzChord:
    sub_root = (root + 6) % 12
    return JazzChord(root=sub_root, chord_type=chord_type)


def is_tritone_sub(original_root: int, candidate_root: int) -> bool:
    return (candidate_root - original_root) % 12 == 6


def shell_voicing(chord: JazzChord, bass_octave: int = 3) -> VoicingResult:
    intervals = CHORD_INTERVALS[chord.chord_type]
    root_midi = chord.root + 12 * bass_octave
    third = root_midi + intervals[1]
    seventh = root_midi + intervals[3]
    return VoicingResult(
        pitches=(root_midi, third, seventh),
        voicing_type="shell",
        spread=seventh - root_midi,
    )


def rootless_voicing_a(chord: JazzChord, octave: int = 4) -> VoicingResult:
    intervals = CHORD_INTERVALS[chord.chord_type]
    base = chord.root + 12 * octave
    pitches = tuple(base + intervals[i] for i in (1, 3) if i < len(intervals))
    ext_9 = base + 2  # add 9
    ext_13 = base + 9  # add 13 (or 6)
    all_p = (pitches[0], ext_9, pitches[1] if len(pitches) > 1 else ext_13, ext_13)
    return VoicingResult(pitches=all_p, voicing_type="rootless_A", spread=all_p[-1] - all_p[0])


def rootless_voicing_b(chord: JazzChord, octave: int = 4) -> VoicingResult:
    intervals = CHORD_INTERVALS[chord.chord_type]
    base = chord.root + 12 * octave
    seventh = base + intervals[3]
    ninth = base + 14  # 9th in next octave
    third = base + intervals[1] + 12
    thirteenth = base + 9 + 12
    all_p = (seventh, ninth, third, thirteenth)
    return VoicingResult(pitches=all_p, voicing_type="rootless_B", spread=all_p[-1] - all_p[0])


def drop_2_voicing(pitches: tuple[int, ...]) -> tuple[int, ...]:
    if len(pitches) < 4:
        return pitches
    lst = list(pitches)
    lst[1] -= 12
    return tuple(sorted(lst))


def drop_3_voicing(pitches: tuple[int, ...]) -> tuple[int, ...]:
    if len(pitches) < 4:
        return pitches
    lst = list(pitches)
    lst[2] -= 12
    return tuple(sorted(lst))


def chord_scale(chord: JazzChord) -> JazzMode:
    ct = chord.chord_type
    if ct == JazzChordType.MAJ7:
        return JazzMode.IONIAN
    if ct == JazzChordType.MIN7:
        return JazzMode.DORIAN
    if ct == JazzChordType.DOM7:
        return JazzMode.MIXOLYDIAN
    if ct == JazzChordType.HALF_DIM:
        return JazzMode.LOCRIAN
    if ct == JazzChordType.DIM7:
        return JazzMode.DIMINISHED_WH
    if ct == JazzChordType.ALT:
        return JazzMode.ALTERED
    if ct == JazzChordType.MIN_MAJ7:
        return JazzMode.MELODIC_MINOR
    if ct == JazzChordType.AUG7:
        return JazzMode.WHOLE_TONE
    if ct == JazzChordType.SUS4:
        return JazzMode.MIXOLYDIAN
    return JazzMode.IONIAN


def scale_pitches(root: int, mode: JazzMode) -> tuple[int, ...]:
    return tuple((root + i) % 12 for i in MODE_INTERVALS[mode])


def available_tensions(chord: JazzChord) -> list[int]:
    base_pcs = set(chord.pitch_classes)
    mode = chord_scale(chord)
    scale_pcs = set(scale_pitches(chord.root, mode))
    tensions: list[int] = []
    for pc in scale_pcs:
        if pc not in base_pcs:
            interval = (pc - chord.root) % 12
            tensions.append(interval)
    return sorted(tensions)


def voice_leading_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if len(a) != len(b):
        return sum(abs(x - y) for x, y in zip(a, b))
    cost_matrix = np.array([[abs(ai - bj) for bj in b] for ai in a])
    n = len(a)
    total = 0
    used: set[int] = set()
    for _ in range(n):
        best_val = 999
        best_i, best_j = 0, 0
        for i in range(n):
            for j in range(n):
                if j not in used and cost_matrix[i, j] < best_val:
                    best_val = cost_matrix[i, j]
                    best_i, best_j = i, j
        total += best_val
        used.add(best_j)
        cost_matrix[best_i, :] = 999
    return total


def smooth_voice_lead(
    chords: list[JazzChord],
    octave: int = 4,
) -> list[tuple[int, ...]]:
    if not chords:
        return []
    result: list[tuple[int, ...]] = []
    prev = shell_voicing(chords[0], octave).pitches
    result.append(prev)
    for ch in chords[1:]:
        candidates = [
            shell_voicing(ch, octave).pitches,
            rootless_voicing_a(ch, octave).pitches,
            rootless_voicing_b(ch, octave).pitches,
        ]
        best = min(candidates, key=lambda c: voice_leading_distance(prev, c))
        result.append(best)
        prev = best
    return result


def bebop_chromatic_approach(target_pc: int, direction: int = -1) -> tuple[int, int]:
    return ((target_pc + direction) % 12, target_pc)


def guide_tone_line(chords: list[JazzChord]) -> list[int]:
    line: list[int] = []
    for ch in chords:
        intervals = CHORD_INTERVALS[ch.chord_type]
        third = (ch.root + intervals[1]) % 12
        seventh = (ch.root + intervals[3]) % 12
        if not line:
            line.append(third)
        else:
            prev = line[-1]
            d3 = min(abs(third - prev), 12 - abs(third - prev))
            d7 = min(abs(seventh - prev), 12 - abs(seventh - prev))
            line.append(third if d3 <= d7 else seventh)
    return line


def backdoor_progression(key: int) -> list[JazzChord]:
    iv = JazzChord(root=(key + 5) % 12, chord_type=JazzChordType.MIN7)
    bvii = JazzChord(root=(key + 10) % 12, chord_type=JazzChordType.DOM7)
    i = JazzChord(root=key, chord_type=JazzChordType.MAJ7)
    return [iv, bvii, i]


def coltrane_changes(key: int) -> list[JazzChord]:
    roots = [key, (key + 4) % 12, (key + 8) % 12]
    chords: list[JazzChord] = []
    for r in roots:
        v = JazzChord(root=(r + 7) % 12, chord_type=JazzChordType.DOM7)
        i = JazzChord(root=r, chord_type=JazzChordType.MAJ7)
        chords.extend([v, i])
    return chords


def turnaround(key: int, minor: bool = False) -> list[JazzChord]:
    if minor:
        return [
            JazzChord(root=key, chord_type=JazzChordType.MIN7),
            JazzChord(root=(key + 5) % 12, chord_type=JazzChordType.HALF_DIM),
            JazzChord(root=(key + 7) % 12, chord_type=JazzChordType.DOM7),
            JazzChord(root=key, chord_type=JazzChordType.MIN7),
        ]
    return [
        JazzChord(root=key, chord_type=JazzChordType.MAJ7),
        JazzChord(root=(key + 9) % 12, chord_type=JazzChordType.MIN7),
        JazzChord(root=(key + 2) % 12, chord_type=JazzChordType.DOM7),
        JazzChord(root=(key + 7) % 12, chord_type=JazzChordType.DOM7),
    ]


def harmonic_density(
    chord_types: list[JazzChordType],
    extensions_per_chord: list[int],
) -> float:
    if not chord_types:
        return 0.0
    total = sum(
        len(CHORD_INTERVALS[ct]) + ext
        for ct, ext in zip(chord_types, extensions_per_chord)
    )
    return total / len(chord_types)
