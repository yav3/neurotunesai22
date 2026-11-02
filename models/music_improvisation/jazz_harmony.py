"""Jazz harmony: chord voicings, guide tone lines, approach notes, reharmonization, tension analysis."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ChordTone:
    root: int
    quality: str = "maj7"
    extensions: list[int] = field(default_factory=list)


CHORD_INTERVALS: dict[str, list[int]] = {
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "dim7": [0, 3, 6, 9],
    "halfdim7": [0, 3, 6, 10],
    "aug": [0, 4, 8],
    "min_maj7": [0, 3, 7, 11],
    "sus4": [0, 5, 7, 10],
}

SCALE_INTERVALS: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "altered": [0, 1, 3, 4, 6, 8, 10],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "diminished": [0, 2, 3, 5, 6, 8, 9, 11],
}


def chord_tones(chord: ChordTone) -> np.ndarray:
    intervals = CHORD_INTERVALS.get(chord.quality, [0, 4, 7, 11])
    pcs = [(chord.root + i) % 12 for i in intervals]
    return np.array(sorted(set(pcs)), dtype=int)


AVOID_NOTES: dict[str, set[int]] = {
    "maj7": {11},
    "dom7": set(),
    "min7": {8},
    "halfdim7": set(),
    "dim7": set(),
}


def available_tensions(chord: ChordTone, scale_root: int = 0, scale_type: str = "major") -> np.ndarray:
    ct = set(int(x) for x in chord_tones(chord))
    scale = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])
    scale_pcs = {(scale_root + s) % 12 for s in scale}
    avoid = AVOID_NOTES.get(chord.quality, set())
    avoid_pcs = {(chord.root + a) % 12 for a in avoid}
    tensions = []
    for pc in scale_pcs:
        if pc not in ct and pc not in avoid_pcs:
            tensions.append(pc)
    return np.array(sorted(tensions), dtype=int)


def guide_tone_line(chord_sequence: list[ChordTone]) -> list[int]:
    if not chord_sequence:
        return []
    line = []
    for i, chord in enumerate(chord_sequence):
        intervals = CHORD_INTERVALS.get(chord.quality, [0, 4, 7, 11])
        third = (chord.root + intervals[1]) % 12 if len(intervals) > 1 else chord.root
        seventh = (chord.root + intervals[-1]) % 12 if len(intervals) > 3 else third
        if not line:
            line.append(third)
        else:
            prev = line[-1]
            d3 = min(abs(third - prev), 12 - abs(third - prev))
            d7 = min(abs(seventh - prev), 12 - abs(seventh - prev))
            line.append(third if d3 <= d7 else seventh)
    return line


def approach_note(target: int, approach_type: str = "chromatic") -> list[int]:
    if approach_type == "chromatic":
        return [(target - 1) % 12, target]
    elif approach_type == "diatonic":
        return [(target - 2) % 12, target]
    elif approach_type == "enclosure":
        return [(target + 1) % 12, (target - 1) % 12, target]
    return [target]


def ii_V_I_voicings(key: int, mode: str = "major") -> list[np.ndarray]:
    if mode == "major":
        ii = np.array([(key + 2) % 12, (key + 5) % 12, (key + 9) % 12, (key + 0) % 12])
        V = np.array([(key + 7) % 12, (key + 11) % 12, (key + 2) % 12, (key + 5) % 12])
        I = np.array([(key + 0) % 12, (key + 4) % 12, (key + 7) % 12, (key + 11) % 12])
    else:
        ii = np.array([(key + 2) % 12, (key + 5) % 12, (key + 8) % 12, (key + 0) % 12])
        V = np.array([(key + 7) % 12, (key + 11) % 12, (key + 2) % 12, (key + 5) % 12])
        I = np.array([(key + 0) % 12, (key + 3) % 12, (key + 7) % 12, (key + 11) % 12])
    return [ii, V, I]


def tritone_substitution(chord: ChordTone) -> ChordTone:
    new_root = (chord.root + 6) % 12
    return ChordTone(root=new_root, quality=chord.quality, extensions=list(chord.extensions))


def backdoor_progression(target: ChordTone) -> list[ChordTone]:
    bVII = ChordTone(root=(target.root - 2) % 12, quality="dom7")
    return [bVII, target]


def coltrane_changes(start_chord: ChordTone) -> list[ChordTone]:
    roots = [start_chord.root, (start_chord.root + 4) % 12, (start_chord.root + 8) % 12]
    result = []
    for r in roots:
        dom = ChordTone(root=(r + 7) % 12, quality="dom7")
        tonic = ChordTone(root=r, quality="maj7")
        result.extend([dom, tonic])
    return result


def tension_resolution_score(melody_pc: int, chord: ChordTone) -> float:
    ct = set(int(x) for x in chord_tones(chord))
    pc = melody_pc % 12
    if pc in ct:
        return 1.0
    tensions = set(int(x) for x in available_tensions(chord, chord.root, "major"))
    if pc in tensions:
        return 0.7
    return 0.3
