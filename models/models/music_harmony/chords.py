"""Chord representation, parsing, and similarity."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class ChordQuality(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT7 = "dominant7"
    MAJOR7 = "major7"
    MINOR7 = "minor7"
    HALF_DIM7 = "half_dim7"
    DIM7 = "dim7"
    SUS2 = "sus2"
    SUS4 = "sus4"
    ADD9 = "add9"
    POWER = "power"


CHORD_TYPES: dict[ChordQuality, tuple[int, ...]] = {
    ChordQuality.MAJOR: (0, 4, 7),
    ChordQuality.MINOR: (0, 3, 7),
    ChordQuality.DIMINISHED: (0, 3, 6),
    ChordQuality.AUGMENTED: (0, 4, 8),
    ChordQuality.DOMINANT7: (0, 4, 7, 10),
    ChordQuality.MAJOR7: (0, 4, 7, 11),
    ChordQuality.MINOR7: (0, 3, 7, 10),
    ChordQuality.HALF_DIM7: (0, 3, 6, 10),
    ChordQuality.DIM7: (0, 3, 6, 9),
    ChordQuality.SUS2: (0, 2, 7),
    ChordQuality.SUS4: (0, 5, 7),
    ChordQuality.ADD9: (0, 2, 4, 7),
    ChordQuality.POWER: (0, 7),
}

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_MAP: dict[str, int] = {}
for _i, _n in enumerate(NOTE_NAMES):
    NOTE_MAP[_n] = _i
NOTE_MAP.update({"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11})


@dataclass(frozen=True)
class Chord:
    root: int
    quality: ChordQuality
    bass: int | None = None
    extensions: tuple[int, ...] = ()

    @property
    def pitch_classes(self) -> frozenset[int]:
        base = CHORD_TYPES[self.quality]
        pcs = {(self.root + interval) % 12 for interval in base}
        pcs.update((self.root + e) % 12 for e in self.extensions)
        return frozenset(pcs)

    @property
    def root_name(self) -> str:
        return NOTE_NAMES[self.root % 12]

    @property
    def name(self) -> str:
        suffixes = {
            ChordQuality.MAJOR: "", ChordQuality.MINOR: "m",
            ChordQuality.DIMINISHED: "dim", ChordQuality.AUGMENTED: "aug",
            ChordQuality.DOMINANT7: "7", ChordQuality.MAJOR7: "maj7",
            ChordQuality.MINOR7: "m7", ChordQuality.HALF_DIM7: "m7b5",
            ChordQuality.DIM7: "dim7", ChordQuality.SUS2: "sus2",
            ChordQuality.SUS4: "sus4", ChordQuality.ADD9: "add9",
            ChordQuality.POWER: "5",
        }
        base = self.root_name + suffixes.get(self.quality, "")
        if self.bass is not None and self.bass != self.root:
            base += "/" + NOTE_NAMES[self.bass % 12]
        return base


_QUALITY_SUFFIXES: list[tuple[str, ChordQuality]] = [
    ("maj7", ChordQuality.MAJOR7), ("m7b5", ChordQuality.HALF_DIM7),
    ("dim7", ChordQuality.DIM7), ("m7", ChordQuality.MINOR7),
    ("7", ChordQuality.DOMINANT7), ("aug", ChordQuality.AUGMENTED),
    ("dim", ChordQuality.DIMINISHED), ("sus2", ChordQuality.SUS2),
    ("sus4", ChordQuality.SUS4), ("add9", ChordQuality.ADD9),
    ("5", ChordQuality.POWER), ("m", ChordQuality.MINOR),
]


def chord_from_name(name: str) -> Chord:
    slash = name.find("/")
    bass: int | None = None
    body = name
    if slash > 0:
        bass_str = name[slash + 1:]
        if bass_str in NOTE_MAP:
            bass = NOTE_MAP[bass_str]
        body = name[:slash]
    root_len = 2 if len(body) > 1 and body[1] in "#b" else 1
    root_str, suffix = body[:root_len], body[root_len:]
    root = NOTE_MAP[root_str]
    quality = ChordQuality.MAJOR
    for suf, qual in _QUALITY_SUFFIXES:
        if suffix == suf:
            quality = qual
            break
    return Chord(root=root, quality=quality, bass=bass)


def chord_from_pitches(pitches: Sequence[int]) -> Chord:
    pcs = frozenset(p % 12 for p in pitches)
    best: tuple[int, ChordQuality, int] | None = None
    for root_candidate in pcs:
        for quality, intervals in CHORD_TYPES.items():
            template = frozenset((root_candidate + iv) % 12 for iv in intervals)
            overlap = len(pcs & template)
            total = len(pcs | template)
            if total == 0:
                continue
            score = overlap / total
            if best is None or score > best[2] / (best[2] + 1):
                if overlap == len(template):
                    best = (root_candidate, quality, overlap)
    if best is None:
        return Chord(root=min(pcs), quality=ChordQuality.MAJOR)
    return Chord(root=best[0], quality=best[1])


def chord_similarity(c1: Chord, c2: Chord) -> float:
    pc1, pc2 = c1.pitch_classes, c2.pitch_classes
    if not pc1 and not pc2:
        return 1.0
    return len(pc1 & pc2) / len(pc1 | pc2)
