"""Camelot wheel: harmonic mixing keys, compatibility, transition suggestions.

Full 24-position wheel (1A–12A minor, 1B–12B major). Extends the simpler
lookup in audio_solver.py with scored compatibility and transition planning.

Pure music theory — no ML, no medical code.
"""
from __future__ import annotations

from dataclasses import dataclass

from models.music_theory.intervals import normalize_note

# ---------------------------------------------------------------------------
# Camelot ↔ musical key mapping (standard encoding)
# ---------------------------------------------------------------------------

_CAMELOT_TO_KEY: dict[tuple[int, str], tuple[str, str]] = {
    # (number, letter) → (key_name, scale_type)
    (1,  "A"): ("Ab",  "minor"),  (1,  "B"): ("B",   "major"),
    (2,  "A"): ("Eb",  "minor"),  (2,  "B"): ("F#",  "major"),
    (3,  "A"): ("Bb",  "minor"),  (3,  "B"): ("C#",  "major"),
    (4,  "A"): ("F",   "minor"),  (4,  "B"): ("G#",  "major"),
    (5,  "A"): ("C",   "minor"),  (5,  "B"): ("D#",  "major"),
    (6,  "A"): ("G",   "minor"),  (6,  "B"): ("A#",  "major"),
    (7,  "A"): ("D",   "minor"),  (7,  "B"): ("F",   "major"),
    (8,  "A"): ("A",   "minor"),  (8,  "B"): ("C",   "major"),
    (9,  "A"): ("E",   "minor"),  (9,  "B"): ("G",   "major"),
    (10, "A"): ("B",   "minor"),  (10, "B"): ("D",   "major"),
    (11, "A"): ("F#",  "minor"),  (11, "B"): ("A",   "major"),
    (12, "A"): ("C#",  "minor"),  (12, "B"): ("E",   "major"),
}

_KEY_TO_CAMELOT: dict[tuple[str, str], tuple[int, str]] = {}
for _k, _v in _CAMELOT_TO_KEY.items():
    _key_norm = normalize_note(_v[0])
    _KEY_TO_CAMELOT[(_key_norm, _v[1])] = _k


@dataclass(frozen=True)
class CamelotKey:
    number: int     # 1–12
    letter: str     # 'A' (minor) or 'B' (major)
    key_name: str
    scale_type: str


def key_to_camelot(key_name: str, scale_type: str = "major") -> CamelotKey:
    """Look up the Camelot position for a musical key."""
    norm = normalize_note(key_name)
    st = "minor" if "minor" in scale_type.lower() else "major"
    pair = _KEY_TO_CAMELOT.get((norm, st))
    if pair is None:
        raise ValueError(f"No Camelot mapping for {key_name} {scale_type}")
    return CamelotKey(number=pair[0], letter=pair[1], key_name=norm, scale_type=st)


def camelot_to_key(number: int, letter: str) -> str:
    """Return the musical key name for a Camelot position."""
    pair = _CAMELOT_TO_KEY.get((number, letter.upper()))
    if pair is None:
        raise ValueError(f"Invalid Camelot position: {number}{letter}")
    return pair[0]


def _camelot_distance(k1: CamelotKey, k2: CamelotKey) -> tuple[int, bool]:
    """Return (number_distance_on_wheel, same_letter)."""
    d = min(abs(k1.number - k2.number), 12 - abs(k1.number - k2.number))
    return d, k1.letter == k2.letter


def camelot_compatibility(key1: CamelotKey, key2: CamelotKey) -> float:
    """Compatibility score in [0, 1].

    1.0  = same key
    0.95 = adjacent on wheel (±1, same letter)
    0.90 = energy boost (+7 semitones, same letter — one position across)
    0.85 = parallel (same number, opposite letter: A↔B)
    0.70 = ±2 same letter
    0.50 = ±1 opposite letter
    0.0  = incompatible (distance > 2)
    """
    if key1.number == key2.number and key1.letter == key2.letter:
        return 1.0

    dist, same_letter = _camelot_distance(key1, key2)

    if same_letter and dist == 1:
        return 0.95  # adjacent
    if same_letter and dist == 7:
        return 0.90  # energy boost
    if not same_letter and dist == 0:
        return 0.85  # parallel (A↔B)
    if same_letter and dist == 2:
        return 0.70
    if not same_letter and dist == 1:
        return 0.50

    return 0.0


def harmonic_mixing_score(key1: CamelotKey, key2: CamelotKey) -> float:
    """Alias for camelot_compatibility — score in [0, 1]."""
    return camelot_compatibility(key1, key2)


def suggest_transitions(
    current_key: CamelotKey,
    n: int = 5,
) -> list[tuple[CamelotKey, float, str]]:
    """Return up to *n* compatible keys ranked by compatibility score.

    Each entry: (CamelotKey, score, transition_type_name).
    """
    candidates: list[tuple[CamelotKey, float, str]] = []

    for (num, let), (kn, st) in _CAMELOT_TO_KEY.items():
        ck = CamelotKey(number=num, letter=let, key_name=normalize_note(kn), scale_type=st)
        if ck.number == current_key.number and ck.letter == current_key.letter:
            continue
        score = camelot_compatibility(current_key, ck)
        if score <= 0.0:
            continue

        dist, same_letter = _camelot_distance(current_key, ck)
        if same_letter and dist == 1:
            label = "adjacent"
        elif same_letter and dist == 7:
            label = "energy_boost"
        elif not same_letter and dist == 0:
            label = "parallel"
        elif same_letter and dist == 2:
            label = "outer_compatible"
        elif not same_letter and dist == 1:
            label = "cross_adjacent"
        else:
            label = "compatible"

        candidates.append((ck, score, label))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]
