"""Schenkerian prolongational analysis primitives.

Ursatz detection, Urlinie, Bassbrechung, prolongation types, structural levels.
Operates on pitch sequences as numpy arrays. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class ProlongationType(Enum):
    NEIGHBOR = "neighbor"          # upper/lower neighbor returning to original
    PASSING = "passing"            # stepwise motion between two structural tones
    ARPEGGIATION = "arpeggiation"  # consonant skip filling chord tones
    INITIAL_ASCENT = "initial_ascent"
    REACHING_OVER = "reaching_over"
    UNFOLDING = "unfolding"


class StructuralLevel(Enum):
    FOREGROUND = "foreground"
    MIDDLEGROUND = "middleground"
    BACKGROUND = "background"


@dataclass
class StructuralTone:
    position: int          # index in the original sequence
    pitch: int             # MIDI pitch
    scale_degree: int      # 1-7 (relative to key)
    level: StructuralLevel
    prolongation: ProlongationType | None = None


@dataclass
class Ursatz:
    """Fundamental structure: Urlinie (descent) + Bassbrechung (bass arpeggiation)."""
    urlinie_start: int       # scale degree (3, 5, or 8)
    urlinie_degrees: list[int]   # descending stepwise to 1
    bass_degrees: list[int]      # typically [1, 5, 1]
    key_root: int           # MIDI pitch class of the tonic


@dataclass
class ReductionResult:
    structural_tones: list[StructuralTone]
    removed_indices: list[int]
    prolongations: list[tuple[int, int, ProlongationType]]


def pitch_to_degree(pitch: int, key_root: int, mode: str = "major") -> int:
    """Convert MIDI pitch to scale degree (1-7) relative to key.

    Returns 0 if not a diatonic scale tone.
    """
    major_steps = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    minor_steps = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7}
    steps = major_steps if mode == "major" else minor_steps
    pc = (pitch - key_root) % 12
    return steps.get(pc, 0)


def degree_to_pc(degree: int, key_root: int, mode: str = "major") -> int:
    """Convert scale degree (1-7) to pitch class."""
    major_map = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
    minor_map = {1: 0, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 10}
    m = major_map if mode == "major" else minor_map
    return (key_root + m.get(degree, 0)) % 12


def detect_urlinie(pitches: np.ndarray, key_root: int,
                   mode: str = "major") -> list[int] | None:
    """Detect a fundamental line (Urlinie) — stepwise descent to ^1.

    Searches for descending stepwise patterns from ^3, ^5, or ^8 to ^1.
    Returns the scale degrees if found, else None.
    """
    degrees = np.array([pitch_to_degree(int(p), key_root, mode) for p in pitches])
    nonzero = degrees[degrees > 0]
    if len(nonzero) < 3:
        return None

    for start_deg in (8, 5, 3):
        effective_start = start_deg if start_deg != 8 else 8
        if start_deg == 8:
            target_seq = [8, 7, 6, 5, 4, 3, 2, 1]
        elif start_deg == 5:
            target_seq = [5, 4, 3, 2, 1]
        else:
            target_seq = [3, 2, 1]

        idx = 0
        found: list[int] = []
        for d in nonzero:
            d_eff = d
            if start_deg == 8 and d == 1 and idx == 0:
                d_eff = 8
            if d_eff == 8 and idx == 0 and start_deg == 8:
                d_eff = 8
            if idx < len(target_seq) and d_eff == target_seq[idx]:
                found.append(d_eff)
                idx += 1
                if idx == len(target_seq):
                    return found
    return None


def detect_bassbrechung(bass_pitches: np.ndarray, key_root: int,
                        mode: str = "major") -> list[int] | None:
    """Detect bass arpeggiation I-V-I."""
    degrees = [pitch_to_degree(int(p), key_root, mode) for p in bass_pitches]
    degs_nonzero = [d for d in degrees if d > 0]
    if len(degs_nonzero) < 3:
        return None
    if degs_nonzero[0] == 1:
        for i, d in enumerate(degs_nonzero[1:], 1):
            if d == 5:
                for d2 in degs_nonzero[i + 1:]:
                    if d2 == 1:
                        return [1, 5, 1]
    return None


def detect_neighbor(pitches: np.ndarray, idx: int) -> ProlongationType | None:
    """Check if pitch at *idx* is a neighbor tone (returns to origin)."""
    if idx == 0 or idx >= len(pitches) - 1:
        return None
    if pitches[idx - 1] == pitches[idx + 1]:
        step = abs(int(pitches[idx]) - int(pitches[idx - 1]))
        if step <= 2:
            return ProlongationType.NEIGHBOR
    return None


def detect_passing(pitches: np.ndarray, idx: int) -> ProlongationType | None:
    """Check if pitch at *idx* is a passing tone (stepwise between two others)."""
    if idx == 0 or idx >= len(pitches) - 1:
        return None
    p_prev, p_curr, p_next = int(pitches[idx - 1]), int(pitches[idx]), int(pitches[idx + 1])
    if (p_prev < p_curr < p_next or p_prev > p_curr > p_next):
        if abs(p_curr - p_prev) <= 2 and abs(p_next - p_curr) <= 2:
            return ProlongationType.PASSING
    return None


def detect_arpeggiation(pitches: np.ndarray, idx: int,
                        key_root: int, mode: str = "major") -> ProlongationType | None:
    """Check if pitch at *idx* is an arpeggiation (chord tone skip)."""
    if idx == 0 or idx >= len(pitches) - 1:
        return None
    intervals = {3, 4, 5, 7, 8, 9}
    step = abs(int(pitches[idx]) - int(pitches[idx - 1])) % 12
    if step in intervals:
        return ProlongationType.ARPEGGIATION
    return None


def reduce_foreground(pitches: np.ndarray, key_root: int,
                      mode: str = "major") -> ReductionResult:
    """Remove foreground embellishments, keeping structural tones.

    Identifies neighbor, passing, and arpeggiation tones and removes them.
    """
    n = len(pitches)
    structural: list[StructuralTone] = []
    removed: list[int] = []
    prolongations: list[tuple[int, int, ProlongationType]] = []

    keep = np.ones(n, dtype=bool)
    for i in range(1, n - 1):
        ptype = detect_neighbor(pitches, i)
        if ptype is None:
            ptype = detect_passing(pitches, i)
        if ptype is not None:
            keep[i] = False
            removed.append(i)
            prolongations.append((i - 1, i + 1, ptype))

    for i in range(n):
        if keep[i]:
            deg = pitch_to_degree(int(pitches[i]), key_root, mode)
            structural.append(StructuralTone(
                position=i,
                pitch=int(pitches[i]),
                scale_degree=deg,
                level=StructuralLevel.MIDDLEGROUND,
            ))

    return ReductionResult(structural, removed, prolongations)


def structural_weight(pitch: int, key_root: int, metric_position: float = 1.0,
                      duration: float = 1.0, mode: str = "major") -> float:
    """Heuristic structural weight combining scale degree, metric position, and duration.

    Higher weight = more likely to be structural.
    """
    deg = pitch_to_degree(pitch, key_root, mode)
    deg_weights = {1: 1.0, 3: 0.8, 5: 0.9, 2: 0.5, 4: 0.4, 6: 0.3, 7: 0.6, 0: 0.1}
    w_deg = deg_weights.get(deg, 0.1)
    return float(w_deg * metric_position * np.sqrt(duration))


def analyze_structure(pitches: np.ndarray, key_root: int,
                      durations: np.ndarray | None = None,
                      mode: str = "major") -> dict:
    """Full Schenkerian-style analysis: detect Urlinie, Bassbrechung, reduce foreground."""
    if durations is None:
        durations = np.ones(len(pitches))

    urlinie = detect_urlinie(pitches, key_root, mode)
    reduction = reduce_foreground(pitches, key_root, mode)
    weights = np.array([
        structural_weight(int(pitches[i]), key_root, 1.0, float(durations[i]), mode)
        for i in range(len(pitches))
    ])

    return {
        "urlinie": urlinie,
        "reduction": reduction,
        "structural_weights": weights,
        "n_structural_tones": len(reduction.structural_tones),
        "n_embellishments": len(reduction.removed_indices),
        "reduction_ratio": len(reduction.removed_indices) / max(len(pitches), 1),
    }
