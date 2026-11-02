"""Key modulation analysis: distance metrics, pivot chords, tonicization detection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from models.music_harmony.chords import Chord, ChordQuality, CHORD_TYPES, NOTE_NAMES


class ModulationType(str, Enum):
    DIRECT = "direct"
    PIVOT = "pivot"
    CHROMATIC = "chromatic"
    ENHARMONIC = "enharmonic"
    SEQUENTIAL = "sequential"


MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

DIATONIC_QUALITIES_MAJOR = [
    ChordQuality.MAJOR, ChordQuality.MINOR, ChordQuality.MINOR,
    ChordQuality.MAJOR, ChordQuality.MAJOR, ChordQuality.MINOR,
    ChordQuality.DIMINISHED,
]
DIATONIC_QUALITIES_MINOR = [
    ChordQuality.MINOR, ChordQuality.DIMINISHED, ChordQuality.MAJOR,
    ChordQuality.MINOR, ChordQuality.MINOR, ChordQuality.MAJOR,
    ChordQuality.MAJOR,
]


def _key_pitch_classes(root: int, mode: str = "major") -> frozenset[int]:
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    return frozenset((root + s) % 12 for s in scale)


def _diatonic_triads(root: int, mode: str = "major") -> list[Chord]:
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    qualities = DIATONIC_QUALITIES_MAJOR if mode == "major" else DIATONIC_QUALITIES_MINOR
    return [
        Chord(root=(root + scale[i]) % 12, quality=qualities[i])
        for i in range(7)
    ]


@dataclass(frozen=True)
class Key:
    root: int
    mode: str = "major"

    @property
    def name(self) -> str:
        suffix = "" if self.mode == "major" else "m"
        return NOTE_NAMES[self.root % 12] + suffix

    @property
    def pitch_classes(self) -> frozenset[int]:
        return _key_pitch_classes(self.root, self.mode)

    @property
    def diatonic_triads(self) -> list[Chord]:
        return _diatonic_triads(self.root, self.mode)

    @property
    def relative(self) -> "Key":
        if self.mode == "major":
            return Key(root=(self.root + 9) % 12, mode="minor")
        return Key(root=(self.root + 3) % 12, mode="major")

    @property
    def parallel(self) -> "Key":
        other_mode = "minor" if self.mode == "major" else "major"
        return Key(root=self.root, mode=other_mode)


def key_distance_fifths(k1: Key, k2: Key) -> int:
    r1, r2 = k1.root % 12, k2.root % 12
    cw = (r2 - r1) % 12
    ccw = (r1 - r2) % 12
    fifths_cw = 0
    pos = r1
    for _ in range(12):
        if pos == r2:
            break
        pos = (pos + 7) % 12
        fifths_cw += 1
    fifths_ccw = 0
    pos = r1
    for _ in range(12):
        if pos == r2:
            break
        pos = (pos + 5) % 12
        fifths_ccw += 1
    base = min(fifths_cw, fifths_ccw)
    if k1.mode != k2.mode:
        base += 1
    return base


def key_distance_shared_pcs(k1: Key, k2: Key) -> float:
    pc1, pc2 = k1.pitch_classes, k2.pitch_classes
    shared = len(pc1 & pc2)
    return 1.0 - shared / 7.0


@dataclass(frozen=True)
class ModulationEvent:
    from_key: Key
    to_key: Key
    position: int
    modulation_type: ModulationType
    pivot_chord: Chord | None = None
    distance: int = 0


def find_pivot_chords(from_key: Key, to_key: Key) -> list[Chord]:
    triads_from = from_key.diatonic_triads
    triads_to = to_key.diatonic_triads
    pc_set_to = {c.pitch_classes for c in triads_to}
    return [c for c in triads_from if c.pitch_classes in pc_set_to]


def classify_modulation(from_key: Key, to_key: Key) -> ModulationType:
    dist = key_distance_fifths(from_key, to_key)
    pivots = find_pivot_chords(from_key, to_key)
    if from_key.root == to_key.root and from_key.mode != to_key.mode:
        return ModulationType.CHROMATIC
    if dist <= 1 and pivots:
        return ModulationType.PIVOT
    if dist == 0:
        return ModulationType.DIRECT
    if from_key.relative == to_key or to_key.relative == from_key:
        return ModulationType.PIVOT
    if not pivots and dist > 3:
        return ModulationType.DIRECT
    if pivots:
        return ModulationType.PIVOT
    return ModulationType.CHROMATIC


@dataclass
class Tonicization:
    target_chord: Chord
    applied_dominant: Chord
    position: int
    strength: float = 0.0


def detect_tonicizations(
    chords: Sequence[Chord],
) -> list[Tonicization]:
    results = []
    for i in range(len(chords) - 1):
        c_curr = chords[i]
        c_next = chords[i + 1]
        if c_curr.quality not in (ChordQuality.DOMINANT7, ChordQuality.MAJOR):
            continue
        expected_root = (c_curr.root + 5) % 12
        if c_next.root != expected_root:
            continue
        if c_next.quality in (ChordQuality.MAJOR, ChordQuality.MINOR):
            strength = 0.8 if c_curr.quality == ChordQuality.DOMINANT7 else 0.5
            results.append(Tonicization(
                target_chord=c_next,
                applied_dominant=c_curr,
                position=i,
                strength=strength,
            ))
    return results


class ModulationAnalyzer:
    """Detect modulations in a chord sequence given key annotations."""

    def __init__(self, min_chords_in_key: int = 3) -> None:
        self.min_chords_in_key = min_chords_in_key

    def _chord_fits_key(self, chord: Chord, key: Key) -> bool:
        return chord.pitch_classes.issubset(key.pitch_classes)

    def detect_modulations(
        self,
        chords: Sequence[Chord],
        initial_key: Key,
    ) -> list[ModulationEvent]:
        if len(chords) < self.min_chords_in_key:
            return []

        events: list[ModulationEvent] = []
        current_key = initial_key
        mismatch_start: int | None = None
        mismatch_count = 0

        for i, chord in enumerate(chords):
            if self._chord_fits_key(chord, current_key):
                if mismatch_start is not None and mismatch_count >= self.min_chords_in_key:
                    new_key = self._best_key_for_segment(
                        chords[mismatch_start:i],
                    )
                    if new_key is not None and new_key != current_key:
                        mod_type = classify_modulation(current_key, new_key)
                        pivots = find_pivot_chords(current_key, new_key)
                        events.append(ModulationEvent(
                            from_key=current_key,
                            to_key=new_key,
                            position=mismatch_start,
                            modulation_type=mod_type,
                            pivot_chord=pivots[0] if pivots else None,
                            distance=key_distance_fifths(current_key, new_key),
                        ))
                        current_key = new_key
                mismatch_start = None
                mismatch_count = 0
            else:
                if mismatch_start is None:
                    mismatch_start = i
                mismatch_count += 1

        if mismatch_start is not None and mismatch_count >= self.min_chords_in_key:
            new_key = self._best_key_for_segment(chords[mismatch_start:])
            if new_key is not None and new_key != current_key:
                mod_type = classify_modulation(current_key, new_key)
                pivots = find_pivot_chords(current_key, new_key)
                events.append(ModulationEvent(
                    from_key=current_key,
                    to_key=new_key,
                    position=mismatch_start,
                    modulation_type=mod_type,
                    pivot_chord=pivots[0] if pivots else None,
                    distance=key_distance_fifths(current_key, new_key),
                ))

        return events

    def _best_key_for_segment(
        self, chords: Sequence[Chord],
    ) -> Key | None:
        best_key: Key | None = None
        best_score = -1
        for root in range(12):
            for mode in ("major", "minor"):
                key = Key(root=root, mode=mode)
                score = sum(1 for c in chords if self._chord_fits_key(c, key))
                if score > best_score:
                    best_score = score
                    best_key = key
        return best_key

    def modulation_density(
        self, chords: Sequence[Chord], initial_key: Key,
    ) -> float:
        if len(chords) == 0:
            return 0.0
        events = self.detect_modulations(chords, initial_key)
        return len(events) / len(chords)


def key_distance_matrix(keys: Sequence[Key]) -> np.ndarray:
    n = len(keys)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mat[i, j] = key_distance_fifths(keys[i], keys[j])
    return mat
