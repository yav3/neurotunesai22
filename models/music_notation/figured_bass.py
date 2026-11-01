"""Figured bass notation parsing, realisation, and analysis.

Baroque-era figured bass shorthand: figures → chord voicings,
realisation rules (doubling, voice leading), and analysis of
harmonic rhythm from figured bass sequences.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class FigureQuality(Enum):
    NATURAL = ""
    SHARP = "#"
    FLAT = "b"
    NATURAL_SIGN = "n"


@dataclass(frozen=True)
class Figure:
    interval: int  # scale degree above bass (3, 5, 6, 7, etc.)
    quality: FigureQuality = FigureQuality.NATURAL


@dataclass
class FiguredBassSymbol:
    bass_note: int  # MIDI pitch or scale degree
    figures: List[Figure] = field(default_factory=list)
    duration_beats: float = 1.0

    def implied_intervals(self) -> List[int]:
        """Return intervals above bass implied by the figures."""
        if not self.figures:
            return [0, 4, 7]  # root position triad (unison, 3rd, 5th in semitones)
        return [0] + [_figure_to_semitones(f) for f in self.figures]


# Common shorthand expansions
SHORTHAND: Dict[str, List[int]] = {
    "":    [3, 5],      # 5/3 root position
    "6":   [3, 6],      # first inversion
    "64":  [4, 6],      # second inversion
    "7":   [3, 5, 7],   # root position seventh
    "65":  [3, 5, 6],   # first inversion seventh
    "43":  [3, 4, 6],   # second inversion seventh
    "42":  [2, 4, 6],   # third inversion seventh (4/2)
}


def _figure_to_semitones(f: Figure) -> int:
    """Convert a scale-degree figure to semitones (diatonic major assumed)."""
    diatonic_map = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
    base = diatonic_map.get(f.interval, (f.interval - 1) * 2)  # fallback
    if f.quality == FigureQuality.SHARP:
        return base + 1
    elif f.quality == FigureQuality.FLAT:
        return base - 1
    return base


def parse_figures(text: str) -> List[Figure]:
    """Parse a figured bass string like '6', '65', '7#' into Figure list."""
    figures = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isdigit():
            interval = int(ch)
            quality = FigureQuality.NATURAL
            if i + 1 < len(text) and text[i + 1] in ('#', 'b', 'n'):
                quality = FigureQuality(text[i + 1])
                i += 1
            figures.append(Figure(interval=interval, quality=quality))
        i += 1
    return figures


def realise_chord(
    bass_midi: int,
    figures: List[Figure],
    n_voices: int = 4,
) -> List[int]:
    """Realise a figured bass symbol as MIDI pitches (close spacing)."""
    if not figures:
        figures = [Figure(3), Figure(5)]
    intervals = [_figure_to_semitones(f) for f in figures]
    pitches = [bass_midi]
    for iv in intervals:
        p = bass_midi + iv
        while p < bass_midi:
            p += 12
        pitches.append(p)
    # fill to n_voices by doubling bass at octave
    while len(pitches) < n_voices:
        pitches.append(bass_midi + 12 * (len(pitches) - len(intervals)))
    pitches.sort()
    return pitches[:n_voices]


def harmonic_rhythm(symbols: List[FiguredBassSymbol]) -> np.ndarray:
    """Extract harmonic rhythm as durations between chord changes."""
    if len(symbols) < 2:
        return np.array([s.duration_beats for s in symbols]) if symbols else np.array([])
    durations = []
    for s in symbols:
        durations.append(s.duration_beats)
    return np.array(durations)


def harmonic_density(symbols: List[FiguredBassSymbol], total_beats: float) -> float:
    """Chord changes per beat."""
    if total_beats <= 0:
        return 0.0
    return len(symbols) / total_beats


def voice_leading_cost(chord_a: List[int], chord_b: List[int]) -> int:
    """Sum of absolute semitone motion between two chords (same voice count)."""
    n = min(len(chord_a), len(chord_b))
    return sum(abs(a - b) for a, b in zip(sorted(chord_a)[:n], sorted(chord_b)[:n]))


def realise_sequence(
    symbols: List[FiguredBassSymbol],
    n_voices: int = 4,
) -> List[List[int]]:
    """Realise an entire figured bass sequence."""
    return [realise_chord(s.bass_note, s.figures, n_voices) for s in symbols]


def total_voice_leading_cost(chords: List[List[int]]) -> int:
    """Sum of voice-leading cost across an entire realised sequence."""
    if len(chords) < 2:
        return 0
    return sum(voice_leading_cost(chords[i], chords[i + 1]) for i in range(len(chords) - 1))
