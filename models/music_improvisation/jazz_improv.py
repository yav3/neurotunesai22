"""Jazz improvisation: chord-scale theory, guide tones, motivic development.

Implements chord-scale mapping, available tensions per chord quality,
guide-tone voice leading, motivic development operators (sequence,
fragmentation, augmentation, inversion), and call-response patterns
for computational jazz improvisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHORD_SCALE_MAP: dict[str, list[int]] = {
    "maj7": [0, 2, 4, 5, 7, 9, 11],  # Ionian
    "dom7": [0, 2, 4, 5, 7, 9, 10],  # Mixolydian
    "min7": [0, 2, 3, 5, 7, 9, 10],  # Dorian
    "min7b5": [0, 1, 3, 5, 6, 8, 10],  # Locrian
    "dim7": [0, 1, 3, 4, 6, 7, 9, 10],  # diminished (HW)
    "aug7": [0, 2, 4, 6, 8, 10],  # whole tone
    "alt": [0, 1, 3, 4, 6, 8, 10],  # altered / super-Locrian
    "sus4": [0, 2, 5, 7, 9, 10],  # Mixolydian sus4
    "min_maj7": [0, 2, 3, 5, 7, 9, 11],  # melodic minor
}

AVAILABLE_TENSIONS: dict[str, list[int]] = {
    "maj7": [9, 11, 6],  # 9, #11, 13
    "dom7": [9, 6],  # 9, 13
    "min7": [9, 11],  # 9, 11
    "min7b5": [9, 11],  # b9, 11
    "alt": [1, 3, 6, 8],  # b9, #9, b5, #5
}

GUIDE_TONES: dict[str, list[int]] = {
    "maj7": [4, 11],  # 3, 7
    "dom7": [4, 10],  # 3, b7
    "min7": [3, 10],  # b3, b7
    "min7b5": [3, 10],  # b3, b7
    "dim7": [3, 9],  # b3, bb7
}


@dataclass
class ChordContext:
    root: int
    quality: str
    duration_beats: float = 4.0


def available_pitches(chord: ChordContext, octave: int = 4) -> list[int]:
    scale = CHORD_SCALE_MAP.get(chord.quality, CHORD_SCALE_MAP["dom7"])
    base = chord.root + octave * 12
    return [base + s for s in scale]


def guide_tone_line(chords: list[ChordContext], octave: int = 4) -> list[int]:
    """Voice-lead guide tones across a chord progression."""
    if not chords:
        return []
    line = []
    prev_pitch = None
    for chord in chords:
        tones = GUIDE_TONES.get(chord.quality, [4, 10])
        candidates = [chord.root + octave * 12 + t for t in tones]
        if prev_pitch is None:
            chosen = candidates[0]
        else:
            chosen = min(candidates, key=lambda p: abs(p - prev_pitch))
        line.append(chosen)
        prev_pitch = chosen
    return line


# ---------------------------------------------------------------------------
# Motivic development operators
# ---------------------------------------------------------------------------

def sequence(motive: list[int], interval: int, repetitions: int = 3) -> list[int]:
    result = list(motive)
    for i in range(1, repetitions):
        result.extend([p + interval * i for p in motive])
    return result


def fragment(motive: list[int], keep_fraction: float = 0.5) -> list[int]:
    n = max(1, int(len(motive) * keep_fraction))
    return motive[:n]


def augment_rhythm(durations: list[float], factor: float = 2.0) -> list[float]:
    return [d * factor for d in durations]


def diminish_rhythm(durations: list[float], factor: float = 0.5) -> list[float]:
    return [d * factor for d in durations]


def invert_motive(motive: list[int], axis: int | None = None) -> list[int]:
    if not motive:
        return []
    if axis is None:
        axis = motive[0]
    return [2 * axis - p for p in motive]


def retrograde(motive: list[int]) -> list[int]:
    return list(reversed(motive))


# ---------------------------------------------------------------------------
# Call-response generator
# ---------------------------------------------------------------------------

@dataclass
class CallResponse:
    call: list[int]
    response: list[int]
    similarity: float


def generate_response(call: list[int], chord: ChordContext,
                      variation: float = 0.3,
                      rng: np.random.RandomState | None = None) -> CallResponse:
    if rng is None:
        rng = np.random.RandomState()
    scale = available_pitches(chord)
    response = []
    for p in call:
        if rng.rand() < variation:
            response.append(scale[rng.randint(len(scale))])
        else:
            nearest = min(scale, key=lambda s: abs(s - p))
            response.append(nearest)
    similarity = 1.0 - sum(abs(a - b) for a, b in zip(call, response)) / (len(call) * 12 + 1e-12)
    return CallResponse(call, response, max(0.0, similarity))


# ---------------------------------------------------------------------------
# Tension-release phrasing
# ---------------------------------------------------------------------------

def tension_profile(melody: list[int], chord: ChordContext) -> np.ndarray:
    """Per-note tension: 0 = chord tone, 0.5 = scale tone, 1.0 = chromatic."""
    chord_tones = set((chord.root + i) % 12 for i in [0, 4, 7, 10])
    scale_tones = set((chord.root + s) % 12 for s in CHORD_SCALE_MAP.get(chord.quality, []))
    tensions = []
    for p in melody:
        pc = p % 12
        if pc in chord_tones:
            tensions.append(0.0)
        elif pc in scale_tones:
            tensions.append(0.5)
        else:
            tensions.append(1.0)
    return np.array(tensions)


def phrase_contour(melody: list[int]) -> dict[str, float]:
    if len(melody) < 2:
        return {"range": 0, "peak_position": 0.0, "direction": 0.0}
    pitches = np.array(melody, dtype=float)
    return {
        "range": float(np.max(pitches) - np.min(pitches)),
        "peak_position": float(np.argmax(pitches) / (len(pitches) - 1)),
        "direction": float(pitches[-1] - pitches[0]),
        "mean_interval": float(np.mean(np.abs(np.diff(pitches)))),
    }
