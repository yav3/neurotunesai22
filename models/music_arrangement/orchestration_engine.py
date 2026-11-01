"""Algorithmic orchestration: instrument range/timbre modelling, voice assignment,
voicing-rule scoring, and doubling analysis.  Pure numpy."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class InstrumentFamily(str, Enum):
    WOODWIND = "woodwind"
    BRASS = "brass"
    STRING = "string"
    KEYBOARD = "keyboard"
    PERCUSSION = "percussion"
    VOICE = "voice"


@dataclass
class InstrumentSpec:
    name: str
    family: InstrumentFamily
    low_midi: int
    high_midi: int
    transposition: int = 0
    dynamic_range_db: float = 40.0
    brightness: float = 0.5
    sustain_capability: bool = True


ORCHESTRA_INSTRUMENTS: Dict[str, InstrumentSpec] = {
    "piccolo": InstrumentSpec("Piccolo", InstrumentFamily.WOODWIND, 74, 108, 0, 35, 0.85, True),
    "flute": InstrumentSpec("Flute", InstrumentFamily.WOODWIND, 60, 96, 0, 40, 0.75, True),
    "oboe": InstrumentSpec("Oboe", InstrumentFamily.WOODWIND, 58, 91, 0, 35, 0.65, True),
    "english_horn": InstrumentSpec("English Horn", InstrumentFamily.WOODWIND, 52, 81, -7, 30, 0.55, True),
    "clarinet_bb": InstrumentSpec("Clarinet in Bb", InstrumentFamily.WOODWIND, 50, 94, -2, 50, 0.60, True),
    "bassoon": InstrumentSpec("Bassoon", InstrumentFamily.WOODWIND, 34, 75, 0, 40, 0.40, True),
    "french_horn": InstrumentSpec("French Horn", InstrumentFamily.BRASS, 34, 77, -7, 45, 0.45, True),
    "trumpet_bb": InstrumentSpec("Trumpet in Bb", InstrumentFamily.BRASS, 55, 82, -2, 50, 0.80, True),
    "trombone": InstrumentSpec("Trombone", InstrumentFamily.BRASS, 40, 72, 0, 50, 0.55, True),
    "tuba": InstrumentSpec("Tuba", InstrumentFamily.BRASS, 28, 58, 0, 45, 0.30, True),
    "violin": InstrumentSpec("Violin", InstrumentFamily.STRING, 55, 103, 0, 50, 0.70, True),
    "viola": InstrumentSpec("Viola", InstrumentFamily.STRING, 48, 93, 0, 45, 0.55, True),
    "cello": InstrumentSpec("Cello", InstrumentFamily.STRING, 36, 84, 0, 50, 0.50, True),
    "contrabass": InstrumentSpec("Contrabass", InstrumentFamily.STRING, 28, 67, 0, 40, 0.30, True),
    "harp": InstrumentSpec("Harp", InstrumentFamily.STRING, 24, 103, 0, 35, 0.60, False),
    "piano": InstrumentSpec("Piano", InstrumentFamily.KEYBOARD, 21, 108, 0, 60, 0.65, False),
    "organ": InstrumentSpec("Organ", InstrumentFamily.KEYBOARD, 24, 108, 0, 50, 0.50, True),
    "timpani": InstrumentSpec("Timpani", InstrumentFamily.PERCUSSION, 40, 55, 0, 55, 0.35, False),
    "snare": InstrumentSpec("Snare Drum", InstrumentFamily.PERCUSSION, 60, 60, 0, 45, 0.70, False),
    "soprano": InstrumentSpec("Soprano", InstrumentFamily.VOICE, 60, 84, 0, 45, 0.70, True),
    "alto_voice": InstrumentSpec("Alto", InstrumentFamily.VOICE, 53, 77, 0, 40, 0.55, True),
    "tenor": InstrumentSpec("Tenor", InstrumentFamily.VOICE, 48, 72, 0, 40, 0.60, True),
    "bass_voice": InstrumentSpec("Bass", InstrumentFamily.VOICE, 40, 64, 0, 40, 0.40, True),
}


def in_range(note_midi: int, spec: InstrumentSpec) -> bool:
    return spec.low_midi <= note_midi <= spec.high_midi


def range_utilization(notes_midi: np.ndarray, spec: InstrumentSpec) -> float:
    if len(notes_midi) == 0:
        return 0.0
    span = spec.high_midi - spec.low_midi
    if span <= 0:
        return 0.0
    lo, hi = float(notes_midi.min()), float(notes_midi.max())
    used = hi - lo
    return min(used / span, 1.0)


def tessitura_comfort(note_midi: int, spec: InstrumentSpec) -> float:
    center = (spec.low_midi + spec.high_midi) / 2.0
    half_range = (spec.high_midi - spec.low_midi) / 2.0
    if half_range <= 0:
        return 1.0
    sigma = half_range * 0.6
    return float(np.exp(-0.5 * ((note_midi - center) / sigma) ** 2))


def timbre_compatibility(a: InstrumentSpec, b: InstrumentSpec) -> float:
    family_bonus = 0.3 if a.family == b.family else 0.0
    brightness_penalty = abs(a.brightness - b.brightness) * 0.5
    return max(0.0, min(1.0, 0.5 + family_bonus - brightness_penalty))


def assign_voices(
    melody_midi: np.ndarray,
    available: Optional[List[InstrumentSpec]] = None,
) -> List[Tuple[InstrumentSpec, float]]:
    instruments = available or list(ORCHESTRA_INSTRUMENTS.values())
    scores: List[Tuple[InstrumentSpec, float]] = []
    for spec in instruments:
        playable = np.array([in_range(int(n), spec) for n in melody_midi])
        if not np.all(playable):
            continue
        comfort = float(np.mean([tessitura_comfort(int(n), spec) for n in melody_midi]))
        util = range_utilization(melody_midi, spec)
        score = 0.7 * comfort + 0.3 * util
        scores.append((spec, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def doubling_score(voices: List[InstrumentSpec]) -> float:
    if len(voices) <= 1:
        return 1.0
    families = [v.family for v in voices]
    unique = len(set(families))
    diversity = unique / len(families)
    same_count = len(families) - unique
    penalty = same_count * 0.15
    return max(0.0, min(1.0, diversity - penalty + 0.5))


def voice_crossing_count(parts: List[np.ndarray]) -> int:
    if len(parts) < 2:
        return 0
    min_len = min(len(p) for p in parts)
    crossings = 0
    for i in range(len(parts) - 1):
        for t in range(min_len):
            if parts[i][t] < parts[i + 1][t]:
                crossings += 1
    return crossings


def spacing_score(chord_midi: np.ndarray) -> float:
    if len(chord_midi) < 2:
        return 1.0
    s = np.sort(chord_midi)
    intervals = np.diff(s)
    n = len(intervals)
    ideal = np.linspace(12, 3, n)
    diffs = np.abs(intervals - ideal)
    return float(max(0.0, 1.0 - np.mean(diffs) / 12.0))


def orchestration_score(
    parts: List[np.ndarray],
    instruments: List[InstrumentSpec],
) -> float:
    if not parts or not instruments:
        return 0.0
    comfort_sum = 0.0
    for part, inst in zip(parts, instruments):
        comfort_sum += float(np.mean([tessitura_comfort(int(n), inst) for n in part]))
    comfort_avg = comfort_sum / len(parts)
    crossings = voice_crossing_count(parts)
    cross_penalty = min(crossings * 0.02, 0.3)
    min_len = min(len(p) for p in parts)
    spacing_sum = 0.0
    for t in range(min_len):
        chord = np.array([p[t] for p in parts])
        spacing_sum += spacing_score(chord)
    spacing_avg = spacing_sum / max(min_len, 1)
    blend = 0.0
    n_pairs = 0
    for i in range(len(instruments)):
        for j in range(i + 1, len(instruments)):
            blend += timbre_compatibility(instruments[i], instruments[j])
            n_pairs += 1
    blend_avg = blend / max(n_pairs, 1)
    return 0.4 * comfort_avg + 0.25 * spacing_avg + 0.2 * blend_avg - cross_penalty + 0.15
