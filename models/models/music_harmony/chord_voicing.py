"""Chord voicing and realization: close, open, drop, jazz, guitar voicings.

Realizes abstract chord symbols into concrete pitch sets with voicing-type
selection, jazz shell/rootless voicings, guitar fret-diagram generation,
and voice-leading-optimized progressions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class VoicingType(Enum):
    CLOSE = "close"
    OPEN = "open"
    DROP2 = "drop2"
    DROP3 = "drop3"
    DROP24 = "drop24"
    SPREAD = "spread"
    CLUSTER = "cluster"


CHORD_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "min7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "aug7": [0, 4, 8, 10],
    "6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}


@dataclass
class ChordVoicing:
    root: int
    quality: str
    pitches: list[int]
    voicing_type: VoicingType
    bass_note: int | None = None


def realize_chord(root: int, quality: str, voicing_type: VoicingType = VoicingType.CLOSE,
                  octave: int = 4) -> ChordVoicing:
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])
    base = root + octave * 12
    pitches = [base + iv for iv in intervals]

    if voicing_type == VoicingType.OPEN:
        if len(pitches) >= 3:
            pitches[1] += 12
    elif voicing_type == VoicingType.DROP2:
        if len(pitches) >= 4:
            second_voice = pitches[-2]
            pitches[-2] = second_voice - 12
            pitches.sort()
    elif voicing_type == VoicingType.DROP3:
        if len(pitches) >= 4:
            third_voice = pitches[-3]
            pitches[-3] = third_voice - 12
            pitches.sort()
    elif voicing_type == VoicingType.DROP24:
        if len(pitches) >= 4:
            pitches[-2] -= 12
            pitches[-4] -= 12 if len(pitches) > 4 else 0
            pitches.sort()
    elif voicing_type == VoicingType.SPREAD:
        spread = []
        for i, p in enumerate(pitches):
            spread.append(p + (i % 2) * 12)
        pitches = sorted(spread)
    elif voicing_type == VoicingType.CLUSTER:
        pitches = [base + iv % 12 for iv in intervals]
        pitches.sort()

    return ChordVoicing(root, quality, pitches, voicing_type, bass_note=pitches[0])


class JazzVoicings:
    """Jazz chord voicings: shell, rootless type A/B, quartal."""

    @staticmethod
    def shell_voicing(root: int, quality: str, octave: int = 3) -> list[int]:
        base = root + octave * 12
        intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])
        third = None
        seventh = None
        for iv in intervals:
            if iv in (3, 4):
                third = iv
            if iv in (10, 11):
                seventh = iv
        result = [base]
        if third is not None:
            result.append(base + third)
        if seventh is not None:
            result.append(base + seventh)
        return result

    @staticmethod
    def rootless_a(root: int, quality: str, octave: int = 3) -> list[int]:
        base = root + octave * 12
        if quality in ("dom7", "9"):
            return [base + 4, base + 7, base + 10, base + 14]
        if quality in ("min7", "min9"):
            return [base + 3, base + 7, base + 10, base + 14]
        if quality in ("maj7", "maj9"):
            return [base + 4, base + 7, base + 11, base + 14]
        intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])
        return [base + iv for iv in intervals if iv != 0]

    @staticmethod
    def rootless_b(root: int, quality: str, octave: int = 3) -> list[int]:
        base = root + octave * 12
        if quality in ("dom7", "9"):
            return [base + 10, base + 14, base + 16, base + 19]
        if quality in ("min7", "min9"):
            return [base + 10, base + 14, base + 15, base + 19]
        if quality in ("maj7", "maj9"):
            return [base + 11, base + 14, base + 16, base + 19]
        return JazzVoicings.rootless_a(root, quality, octave)

    @staticmethod
    def quartal_voicing(root: int, n_notes: int = 4, octave: int = 3) -> list[int]:
        base = root + octave * 12
        return [base + 5 * i for i in range(n_notes)]


STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # EADGBE


@dataclass
class FretDiagram:
    frets: list[int | None]
    pitches: list[int | None]


class GuitarVoicings:
    """Guitar chord voicing generation."""

    def __init__(self, tuning: list[int] | None = None) -> None:
        self.tuning = tuning or STANDARD_TUNING

    def find_voicings(self, root: int, quality: str,
                      max_fret: int = 12, max_stretch: int = 4) -> list[FretDiagram]:
        target_pcs = set((root + iv) % 12 for iv in CHORD_INTERVALS.get(quality, [0, 4, 7]))
        voicings: list[FretDiagram] = []
        for start_fret in range(max_fret - max_stretch + 1):
            frets: list[int | None] = []
            pitches_out: list[int | None] = []
            valid = True
            for string_pitch in self.tuning:
                found = None
                for fret in range(start_fret, start_fret + max_stretch + 1):
                    pc = (string_pitch + fret) % 12
                    if pc in target_pcs:
                        found = fret
                        break
                if found is None:
                    frets.append(None)
                    pitches_out.append(None)
                else:
                    frets.append(found)
                    pitches_out.append(string_pitch + found)
            sounding = [f for f in frets if f is not None]
            if len(sounding) >= 3:
                voiced_pcs = set(p % 12 for p in pitches_out if p is not None)
                if root % 12 in voiced_pcs and target_pcs.issubset(voiced_pcs):
                    voicings.append(FretDiagram(frets, pitches_out))
        return voicings

    def barre_chord(self, root: int, quality: str, position: int = 0) -> FretDiagram:
        intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])
        target_pcs = set((root + iv) % 12 for iv in intervals)
        frets: list[int | None] = []
        pitches_out: list[int | None] = []
        for sp in self.tuning:
            for offset in range(5):
                fret = position + offset
                if (sp + fret) % 12 in target_pcs:
                    frets.append(fret)
                    pitches_out.append(sp + fret)
                    break
            else:
                frets.append(None)
                pitches_out.append(None)
        return FretDiagram(frets, pitches_out)


def smooth_progression(chords: list[tuple[int, str]],
                       voicing_type: VoicingType = VoicingType.CLOSE) -> list[ChordVoicing]:
    if not chords:
        return []
    result = [realize_chord(chords[0][0], chords[0][1], voicing_type)]
    for root, quality in chords[1:]:
        prev_pitches = result[-1].pitches
        prev_center = np.mean(prev_pitches)
        best_voicing = None
        best_cost = float("inf")
        for octave in range(2, 6):
            v = realize_chord(root, quality, voicing_type, octave)
            center = np.mean(v.pitches)
            cost = abs(center - prev_center) + sum(
                min(abs(p - q) for q in v.pitches) for p in prev_pitches
            ) / max(len(prev_pitches), 1)
            if cost < best_cost:
                best_cost = cost
                best_voicing = v
        if best_voicing is not None:
            result.append(best_voicing)
    return result
