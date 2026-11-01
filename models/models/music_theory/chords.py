from __future__ import annotations

from dataclasses import dataclass

from models.music_theory.scales import CHROMATIC_NOTES, ENHARMONIC_MAP


CHORD_FORMULAS: dict[str, tuple[int, ...]] = {
    "major":    (0, 4, 7),
    "minor":    (0, 3, 7),
    "dim":      (0, 3, 6),
    "aug":      (0, 4, 8),
    "sus2":     (0, 2, 7),
    "sus4":     (0, 5, 7),
    "dom7":     (0, 4, 7, 10),
    "maj7":     (0, 4, 7, 11),
    "min7":     (0, 3, 7, 10),
    "dim7":     (0, 3, 6, 9),
    "half_dim7":(0, 3, 6, 10),
    "aug7":     (0, 4, 8, 10),
    "min_maj7": (0, 3, 7, 11),
    "add9":     (0, 4, 7, 14),
    "dom9":     (0, 4, 7, 10, 14),
    "maj9":     (0, 4, 7, 11, 14),
    "min9":     (0, 3, 7, 10, 14),
    "dom11":    (0, 4, 7, 10, 14, 17),
    "dom13":    (0, 4, 7, 10, 14, 17, 21),
    "power":    (0, 7),
}


def _note_index(note: str) -> int:
    if note in ENHARMONIC_MAP and note not in CHROMATIC_NOTES:
        note = ENHARMONIC_MAP[note]
    return CHROMATIC_NOTES.index(note)


@dataclass(frozen=True)
class Chord:
    root: str
    quality: str
    intervals: tuple[int, ...]
    notes: tuple[str, ...]


class ChordLibrary:

    @staticmethod
    def build_chord(root: str, quality: str = "major") -> Chord:
        if quality not in CHORD_FORMULAS:
            raise ValueError(f"Unknown chord quality: {quality}")
        intervals = CHORD_FORMULAS[quality]
        root_idx = _note_index(root)
        notes = tuple(CHROMATIC_NOTES[(root_idx + i) % 12] for i in intervals)
        return Chord(root=root, quality=quality, intervals=intervals, notes=notes)

    @staticmethod
    def detect_chord(notes: tuple[str, ...] | list[str]) -> list[Chord]:
        note_set = frozenset(_note_index(n) for n in notes)
        matches: list[Chord] = []
        for root in CHROMATIC_NOTES:
            root_idx = _note_index(root)
            for quality, formula in CHORD_FORMULAS.items():
                chord_set = frozenset((root_idx + i) % 12 for i in formula)
                if chord_set == note_set:
                    chord_notes = tuple(
                        CHROMATIC_NOTES[(root_idx + i) % 12] for i in formula
                    )
                    matches.append(
                        Chord(root=root, quality=quality, intervals=formula, notes=chord_notes)
                    )
        return matches

    @staticmethod
    def chord_tones(chord: Chord) -> list[str]:
        return list(chord.notes)

    @staticmethod
    def available_tensions(chord: Chord) -> list[int]:
        base = set(i % 12 for i in chord.intervals)
        tensions: list[int] = []
        for t in (9, 11, 13):
            if t % 12 not in base:
                tensions.append(t)
        return tensions

    @staticmethod
    def voice_leading_cost(chord1: Chord, chord2: Chord) -> float:
        idx1 = [_note_index(n) for n in chord1.notes]
        idx2 = [_note_index(n) for n in chord2.notes]
        cost = 0.0
        for i, n1 in enumerate(idx1):
            if i < len(idx2):
                diff = abs(n1 - idx2[i])
                cost += min(diff, 12 - diff)
            else:
                cost += 6.0
        for j in range(len(idx1), len(idx2)):
            cost += 6.0
        return cost

    @staticmethod
    def common_tones(chord1: Chord, chord2: Chord) -> list[str]:
        s1 = set(chord1.notes)
        s2 = set(chord2.notes)
        return sorted(s1 & s2)
