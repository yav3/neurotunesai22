"""Score format conversion: reduction, expansion, part extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScoreFormat(Enum):
    FULL_SCORE = "full_score"
    PIANO_REDUCTION = "piano_reduction"
    LEAD_SHEET = "lead_sheet"
    PARTS = "parts"
    SATB = "satb"


@dataclass
class NotatedPart:
    instrument: str
    pitches: list[list[int]]
    durations: list[float]


def reduce_to_piano(parts: list[NotatedPart]) -> tuple[list[list[int]], list[list[int]]]:
    if not parts:
        return [], []
    max_len = max(len(p.pitches) for p in parts)
    treble: list[list[int]] = []
    bass: list[list[int]] = []
    for beat_idx in range(max_len):
        all_pitches = []
        for part in parts:
            if beat_idx < len(part.pitches):
                all_pitches.extend(part.pitches[beat_idx])
        high = [p for p in all_pitches if p >= 60]
        low = [p for p in all_pitches if p < 60]
        treble.append(sorted(set(high)) if high else [])
        bass.append(sorted(set(low)) if low else [])
    return treble, bass


def expand_lead_sheet(
    melody: list[int],
    chord_symbols: list[str],
    chord_voicings: dict[str, list[int]],
) -> tuple[list[int], list[list[int]]]:
    accompaniment = []
    for sym in chord_symbols:
        voicing = chord_voicings.get(sym, [])
        accompaniment.append(voicing)
    return melody, accompaniment


def part_extraction(
    full_score: dict[str, list[list[int]]],
    instrument: str,
) -> list[list[int]] | None:
    return full_score.get(instrument)


def transpose_part(pitches: list[list[int]], semitones: int) -> list[list[int]]:
    return [[p + semitones for p in chord] for chord in pitches]
