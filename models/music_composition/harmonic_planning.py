"""Tension-guided harmonic planning — Lerdahl (2001) tonal tension model.

Plans chord progressions by specifying a target tension curve, then
selecting chords that best match the desired tension profile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Tonal pitch space (Lerdahl 2001) ─────────────────────────────────────

PITCH_CLASSES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

PC_TO_IDX: Dict[str, int] = {pc: i for i, pc in enumerate(PITCH_CLASSES)}

DIATONIC_SETS: Dict[str, Tuple[int, ...]] = {
    "C": (0, 2, 4, 5, 7, 9, 11),
    "G": (7, 9, 11, 0, 2, 4, 6),
    "D": (2, 4, 6, 7, 9, 11, 1),
    "A": (9, 11, 1, 2, 4, 6, 8),
    "E": (4, 6, 8, 9, 11, 1, 3),
    "B": (11, 1, 3, 4, 6, 8, 10),
    "F#": (6, 8, 10, 11, 1, 3, 5),
    "F": (5, 7, 9, 10, 0, 2, 4),
    "Bb": (10, 0, 2, 3, 5, 7, 9),
    "Eb": (3, 5, 7, 8, 10, 0, 2),
    "Ab": (8, 10, 0, 1, 3, 5, 7),
    "Db": (1, 3, 5, 6, 8, 10, 0),
}

CHORD_TYPES: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
}


@dataclass(frozen=True)
class ChordSymbol:
    root: int       # pitch class 0-11
    quality: str    # key into CHORD_TYPES
    pcs: Tuple[int, ...]

    @staticmethod
    def from_name(root: str, quality: str = "maj") -> "ChordSymbol":
        r = PC_TO_IDX[root]
        intervals = CHORD_TYPES[quality]
        pcs = tuple((r + i) % 12 for i in intervals)
        return ChordSymbol(root=r, quality=quality, pcs=pcs)


@dataclass(frozen=True)
class TensionScore:
    surface_dissonance: float
    harmonic_distance: float
    tonal_distance: float
    total: float


# ── Tension computation ──────────────────────────────────────────────────

def _circle_of_fifths_distance(key1: int, key2: int) -> int:
    """Shortest distance on the circle of fifths."""
    d = (key2 - key1) % 12
    fifths_forward = 0
    pc = key1
    while pc != key2 and fifths_forward < 12:
        pc = (pc + 7) % 12
        fifths_forward += 1
    fifths_back = 12 - fifths_forward
    return min(fifths_forward, fifths_back)


def surface_dissonance(chord: ChordSymbol) -> float:
    """Interval-class based dissonance (Hutchinson-Knopoff model, simplified)."""
    ic_weights = {0: 0.0, 1: 1.0, 2: 0.8, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.7}
    pcs = chord.pcs
    total = 0.0
    count = 0
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            ic = min((pcs[j] - pcs[i]) % 12, (pcs[i] - pcs[j]) % 12)
            total += ic_weights.get(ic, 0.5)
            count += 1
    return total / max(count, 1)


def harmonic_distance(chord_a: ChordSymbol, chord_b: ChordSymbol) -> float:
    """Voice-leading + root-motion distance."""
    root_motion = min((chord_b.root - chord_a.root) % 12, (chord_a.root - chord_b.root) % 12)
    common = len(set(chord_a.pcs) & set(chord_b.pcs))
    max_common = min(len(chord_a.pcs), len(chord_b.pcs))
    retention = common / max(max_common, 1)
    root_cost = {0: 0, 5: 1, 7: 1, 4: 2, 3: 2, 2: 3, 1: 4, 6: 4}.get(root_motion, 3)
    return root_cost * (1.0 - 0.3 * retention)


def tonal_distance(chord: ChordSymbol, key_root: int) -> float:
    """Distance of chord from the tonal centre (circle of fifths)."""
    return float(_circle_of_fifths_distance(key_root, chord.root))


def compute_tension(
    chord: ChordSymbol,
    prev_chord: Optional[ChordSymbol],
    key_root: int = 0,
    w_surface: float = 0.3,
    w_harmonic: float = 0.4,
    w_tonal: float = 0.3,
) -> TensionScore:
    """Composite tension score for a chord in context."""
    sd = surface_dissonance(chord)
    hd = harmonic_distance(prev_chord, chord) if prev_chord else 0.0
    td = tonal_distance(chord, key_root)
    total = w_surface * sd + w_harmonic * hd + w_tonal * td
    return TensionScore(
        surface_dissonance=sd,
        harmonic_distance=hd,
        tonal_distance=td,
        total=total,
    )


# ── Tension curve targets ────────────────────────────────────────────────

def arch_tension(n_chords: int, peak: float = 1.0) -> np.ndarray:
    """Arch shape: low → peak at midpoint → low."""
    t = np.linspace(0, np.pi, n_chords)
    return peak * np.sin(t)


def ramp_tension(n_chords: int, start: float = 0.0, end: float = 1.0) -> np.ndarray:
    return np.linspace(start, end, n_chords)


def wave_tension(n_chords: int, periods: float = 2.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * periods, n_chords)
    return 0.5 + amplitude * np.sin(t)


# ── Harmonic planning ────────────────────────────────────────────────────

def _candidate_chords(key_root: int) -> List[ChordSymbol]:
    """Diatonic triads + dominant 7th for a major key."""
    diatonic = DIATONIC_SETS.get(PITCH_CLASSES[key_root], DIATONIC_SETS["C"])
    qualities = ["maj", "min", "min", "maj", "maj", "min", "dim"]
    chords = []
    for deg, qual in zip(diatonic[:7], qualities):
        chords.append(ChordSymbol.from_name(PITCH_CLASSES[deg % 12], qual))
    chords.append(ChordSymbol.from_name(PITCH_CLASSES[(key_root + 7) % 12], "dom7"))
    return chords


@dataclass
class HarmonicPlan:
    chords: List[ChordSymbol]
    tensions: List[TensionScore]
    target_curve: np.ndarray
    achieved_curve: np.ndarray
    fit_error: float


def plan_progression(
    target_tension: np.ndarray,
    key_root: int = 0,
    start_chord: Optional[ChordSymbol] = None,
) -> HarmonicPlan:
    """Greedy chord selection to match a target tension curve."""
    candidates = _candidate_chords(key_root)
    n = len(target_tension)
    result_chords: List[ChordSymbol] = []
    result_tensions: List[TensionScore] = []
    prev = start_chord or ChordSymbol.from_name(PITCH_CLASSES[key_root], "maj")

    for i in range(n):
        target = target_tension[i]
        best_chord = candidates[0]
        best_err = float("inf")
        best_ts = compute_tension(candidates[0], prev, key_root)

        for c in candidates:
            ts = compute_tension(c, prev, key_root)
            err = abs(ts.total - target)
            if err < best_err:
                best_err = err
                best_chord = c
                best_ts = ts

        result_chords.append(best_chord)
        result_tensions.append(best_ts)
        prev = best_chord

    achieved = np.array([t.total for t in result_tensions])
    fit_error = float(np.mean((achieved - target_tension) ** 2))

    return HarmonicPlan(
        chords=result_chords,
        tensions=result_tensions,
        target_curve=target_tension,
        achieved_curve=achieved,
        fit_error=fit_error,
    )


# ── Functional harmony constraints ──────────────────────────────────────

FUNCTIONAL_GRAMMAR: Dict[str, List[str]] = {
    "I": ["ii", "iii", "IV", "V", "vi"],
    "ii": ["V", "viio"],
    "iii": ["IV", "vi"],
    "IV": ["I", "V", "viio"],
    "V": ["I", "vi"],
    "vi": ["ii", "IV"],
    "viio": ["I"],
}

DEGREE_TO_ROOT: Dict[str, int] = {
    "I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "viio": 11,
}

DEGREE_QUALITY: Dict[str, str] = {
    "I": "maj", "ii": "min", "iii": "min", "IV": "maj",
    "V": "maj", "vi": "min", "viio": "dim",
}


def valid_next_chords(current_degree: str) -> List[str]:
    """Return valid next degrees from functional harmony grammar."""
    return FUNCTIONAL_GRAMMAR.get(current_degree, ["I", "IV", "V"])


def degree_to_chord(degree: str, key_root: int = 0) -> ChordSymbol:
    """Convert Roman numeral degree to ChordSymbol."""
    offset = DEGREE_TO_ROOT.get(degree, 0)
    quality = DEGREE_QUALITY.get(degree, "maj")
    root_pc = (key_root + offset) % 12
    return ChordSymbol.from_name(PITCH_CLASSES[root_pc], quality)
