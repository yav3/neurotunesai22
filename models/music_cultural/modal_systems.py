"""Modal systems: church modes, Greek modes, raga, maqam, pentatonic, and modal analysis.

Comprehensive modal analysis going beyond major/minor: mode detection via pitch
profiles, mode transposition, modal brightness ordering, characteristic intervals,
modal interchange, and cross-cultural mode comparison.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ModalFamily(str, Enum):
    CHURCH = "church"
    GREEK = "greek"
    MELODIC_MINOR = "melodic_minor"
    HARMONIC_MINOR = "harmonic_minor"
    PENTATONIC = "pentatonic"
    BLUES = "blues"
    WHOLE_TONE = "whole_tone"
    DIMINISHED = "diminished"
    RAGA = "raga"
    MAQAM = "maqam"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Mode:
    name: str
    family: ModalFamily
    intervals: Tuple[int, ...]  # semitones from root
    brightness: float = 0.0     # higher = brighter/more sharps
    characteristic_degrees: Tuple[int, ...] = ()


# Church modes ordered by brightness (Lydian brightest → Locrian darkest)
CHURCH_MODES: Dict[str, Mode] = {
    "lydian":     Mode("Lydian",     ModalFamily.CHURCH, (0,2,4,6,7,9,11), 3.0,  (3,)),
    "ionian":     Mode("Ionian",     ModalFamily.CHURCH, (0,2,4,5,7,9,11), 2.0,  (3,6)),
    "mixolydian": Mode("Mixolydian", ModalFamily.CHURCH, (0,2,4,5,7,9,10), 1.0,  (6,)),
    "dorian":     Mode("Dorian",     ModalFamily.CHURCH, (0,2,3,5,7,9,10), 0.0,  (5,)),
    "aeolian":    Mode("Aeolian",    ModalFamily.CHURCH, (0,2,3,5,7,8,10), -1.0, (5,7)),
    "phrygian":   Mode("Phrygian",   ModalFamily.CHURCH, (0,1,3,5,7,8,10), -2.0, (1,)),
    "locrian":    Mode("Locrian",    ModalFamily.CHURCH, (0,1,3,5,6,8,10), -3.0, (1,4)),
}

MELODIC_MINOR_MODES: Dict[str, Mode] = {
    "melodic_minor":   Mode("Melodic Minor",   ModalFamily.MELODIC_MINOR, (0,2,3,5,7,9,11), 1.5, (2,6)),
    "dorian_b2":       Mode("Dorian b2",       ModalFamily.MELODIC_MINOR, (0,1,3,5,7,9,10), -1.5, (1,)),
    "lydian_augmented": Mode("Lydian Aug",      ModalFamily.MELODIC_MINOR, (0,2,4,6,8,9,11), 3.5, (3,4)),
    "lydian_dominant":  Mode("Lydian Dom",      ModalFamily.MELODIC_MINOR, (0,2,4,6,7,9,10), 2.5, (3,6)),
    "mixolydian_b6":   Mode("Mixolydian b6",   ModalFamily.MELODIC_MINOR, (0,2,4,5,7,8,10), 0.5, (5,6)),
    "locrian_natural2": Mode("Locrian #2",     ModalFamily.MELODIC_MINOR, (0,2,3,5,6,8,10), -2.5, (4,)),
    "altered":          Mode("Altered",         ModalFamily.MELODIC_MINOR, (0,1,3,4,6,8,10), -3.5, (1,3,4,6)),
}

PENTATONIC_MODES: Dict[str, Mode] = {
    "major_pentatonic": Mode("Major Pent", ModalFamily.PENTATONIC, (0,2,4,7,9), 2.0, ()),
    "minor_pentatonic": Mode("Minor Pent", ModalFamily.PENTATONIC, (0,3,5,7,10), -1.0, ()),
    "egyptian":         Mode("Egyptian",   ModalFamily.PENTATONIC, (0,2,5,7,10), 0.0, ()),
    "man_gong":         Mode("Man Gong",   ModalFamily.PENTATONIC, (0,3,5,8,10), -2.0, ()),
    "ritusen":          Mode("Ritusen",    ModalFamily.PENTATONIC, (0,2,5,7,9), 1.0, ()),
}


def pitch_class_histogram(midi_notes: np.ndarray) -> np.ndarray:
    """12-bin pitch-class histogram, normalised to sum=1."""
    pcs = midi_notes.astype(int) % 12
    hist = np.zeros(12, dtype=float)
    for pc in pcs:
        hist[pc] += 1.0
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def mode_template(mode: Mode) -> np.ndarray:
    """Binary 12-element template for a mode (root = index 0)."""
    t = np.zeros(12, dtype=float)
    for iv in mode.intervals:
        t[iv % 12] = 1.0
    t /= t.sum()
    return t


def correlate_profile(histogram: np.ndarray, template: np.ndarray) -> float:
    """Pearson correlation between pitch histogram and mode template."""
    h = histogram - histogram.mean()
    t = template - template.mean()
    denom = np.sqrt((h @ h) * (t @ t))
    if denom < 1e-12:
        return 0.0
    return float(h @ t / denom)


def detect_mode(
    midi_notes: np.ndarray,
    modes: Optional[Dict[str, Mode]] = None,
) -> List[Tuple[str, int, float]]:
    """Detect mode by rotating histogram over 12 roots.

    Returns list of (mode_name, root_pc, correlation) sorted descending.
    """
    if modes is None:
        modes = {**CHURCH_MODES, **MELODIC_MINOR_MODES}
    hist = pitch_class_histogram(midi_notes)
    results: List[Tuple[str, int, float]] = []
    for name, mode in modes.items():
        tmpl = mode_template(mode)
        for root in range(12):
            rotated = np.roll(hist, -root)
            score = correlate_profile(rotated, tmpl)
            results.append((name, root, score))
    results.sort(key=lambda x: -x[2])
    return results


def transpose_mode(mode: Mode, semitones: int) -> Tuple[int, ...]:
    """Transpose mode intervals by semitones."""
    return tuple((iv + semitones) % 12 for iv in mode.intervals)


def modal_brightness(midi_notes: np.ndarray, root: int = 0) -> float:
    """Brightness score: sum of raised/lowered scale degrees vs. natural major."""
    major = set(CHURCH_MODES["ionian"].intervals)
    hist = pitch_class_histogram(midi_notes)
    rotated = np.roll(hist, -root)
    score = 0.0
    for pc in range(12):
        if rotated[pc] > 0.05:
            if pc in major:
                pass
            elif (pc - 1) % 12 in major:
                score -= rotated[pc]  # flattened degree
            elif (pc + 1) % 12 in major:
                score += rotated[pc]  # raised degree
    return float(score)


def modal_interchange_candidates(
    current_mode: Mode,
    target_family: Optional[ModalFamily] = None,
) -> List[Tuple[str, Mode, int]]:
    """Suggest modal interchange chords: modes sharing most notes with current.

    Returns (name, mode, shared_count) sorted by shared_count descending.
    """
    current_set = set(current_mode.intervals)
    pool = {**CHURCH_MODES, **MELODIC_MINOR_MODES}
    if target_family:
        pool = {k: v for k, v in pool.items() if v.family == target_family}
    results = []
    for name, mode in pool.items():
        if mode.name == current_mode.name:
            continue
        shared = len(current_set & set(mode.intervals))
        results.append((name, mode, shared))
    results.sort(key=lambda x: -x[2])
    return results


def characteristic_interval_strength(
    midi_notes: np.ndarray, mode: Mode, root: int = 0,
) -> float:
    """How strongly the characteristic degrees of a mode are represented."""
    if not mode.characteristic_degrees:
        return 0.0
    hist = pitch_class_histogram(midi_notes)
    rotated = np.roll(hist, -root)
    char_pcs = [mode.intervals[d] % 12 for d in mode.characteristic_degrees
                if d < len(mode.intervals)]
    if not char_pcs:
        return 0.0
    return float(np.mean([rotated[pc] for pc in char_pcs]))


def interval_vector(mode: Mode) -> np.ndarray:
    """6-element interval vector (ic1..ic6) for the mode's pitch-class set."""
    pcs = set(mode.intervals)
    vec = np.zeros(6, dtype=int)
    pcs_list = sorted(pcs)
    for i in range(len(pcs_list)):
        for j in range(i + 1, len(pcs_list)):
            diff = (pcs_list[j] - pcs_list[i]) % 12
            ic = min(diff, 12 - diff)
            if 1 <= ic <= 6:
                vec[ic - 1] += 1
    return vec


def mode_distance(a: Mode, b: Mode) -> float:
    """Euclidean distance between interval vectors."""
    va = interval_vector(a).astype(float)
    vb = interval_vector(b).astype(float)
    return float(np.linalg.norm(va - vb))


def brightness_ordering(modes: Dict[str, Mode]) -> List[Tuple[str, float]]:
    """Order modes by brightness score."""
    result = [(name, mode.brightness) for name, mode in modes.items()]
    result.sort(key=lambda x: -x[1])
    return result


def mode_subset_of(inner: Mode, outer: Mode) -> bool:
    """Check if inner's pitch-class set is a subset of outer's."""
    return set(inner.intervals).issubset(set(outer.intervals))


def common_tones(a: Mode, b: Mode) -> int:
    """Number of shared pitch classes between two modes (both rooted at 0)."""
    return len(set(a.intervals) & set(b.intervals))
