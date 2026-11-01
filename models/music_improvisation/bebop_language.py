"""Bebop language: chromatic approach tones, enclosures, digital patterns,
guide-tone lines, and bebop-scale vocabulary.

Based on Baker (1988) bebop language analysis, Coker (1964) patterns for
jazz, and Bergonzi (2000) inside/outside approach. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class ApproachType(Enum):
    CHROMATIC_BELOW = "chromatic_below"
    CHROMATIC_ABOVE = "chromatic_above"
    DIATONIC_BELOW = "diatonic_below"
    DIATONIC_ABOVE = "diatonic_above"
    DOUBLE_CHROMATIC = "double_chromatic"
    ENCLOSURE = "enclosure"


class DigitalPattern(Enum):
    P1235 = "1235"
    P1357 = "1357"
    P3579 = "3579"
    P5713 = "5713"
    P7135 = "7135"
    P1325 = "1325"


BEBOP_MAJOR = [0, 2, 4, 5, 7, 8, 9, 11]
BEBOP_DOMINANT = [0, 2, 4, 5, 7, 9, 10, 11]
BEBOP_DORIAN = [0, 2, 3, 4, 5, 7, 9, 10]
BEBOP_MELODIC_MINOR = [0, 2, 3, 5, 7, 8, 9, 11]


@dataclass
class BebopPhrase:
    pitches: np.ndarray
    durations: np.ndarray
    approach_tones: List[int] = field(default_factory=list)
    enclosures: List[Tuple[int, int]] = field(default_factory=list)
    guide_tones: List[int] = field(default_factory=list)


@dataclass
class GuidetoneLine:
    chord_roots: np.ndarray
    chord_types: List[str]
    guide_pitches: np.ndarray


def chromatic_approach(target: int, from_below: bool = True) -> List[int]:
    if from_below:
        return [target - 1, target]
    return [target + 1, target]


def diatonic_approach(target: int, scale: List[int],
                      from_below: bool = True) -> List[int]:
    root = target % 12
    scale_set = set(scale)
    if from_below:
        for i in range(1, 4):
            candidate = (root - i) % 12
            if candidate in scale_set:
                return [target - i, target]
        return [target - 2, target]
    for i in range(1, 4):
        candidate = (root + i) % 12
        if candidate in scale_set:
            return [target + i, target]
    return [target + 2, target]


def enclosure(target: int, chromatic: bool = True) -> List[int]:
    if chromatic:
        return [target + 1, target - 1, target]
    return [target + 2, target - 1, target]


def double_chromatic_approach(target: int,
                              from_below: bool = True) -> List[int]:
    if from_below:
        return [target - 2, target - 1, target]
    return [target + 2, target + 1, target]


def digital_pattern(root: int, chord_tones: List[int],
                    pattern: DigitalPattern) -> np.ndarray:
    degrees = [int(d) - 1 for d in pattern.value]
    pitches = []
    for d in degrees:
        idx = d % len(chord_tones)
        octave = d // len(chord_tones)
        pitches.append(root + chord_tones[idx] + 12 * octave)
    return np.array(pitches)


def chord_tones_from_type(chord_type: str) -> List[int]:
    types: Dict[str, List[int]] = {
        "maj7": [0, 4, 7, 11],
        "7": [0, 4, 7, 10],
        "min7": [0, 3, 7, 10],
        "m7b5": [0, 3, 6, 10],
        "dim7": [0, 3, 6, 9],
        "aug7": [0, 4, 8, 10],
        "min_maj7": [0, 3, 7, 11],
        "6": [0, 4, 7, 9],
        "min6": [0, 3, 7, 9],
    }
    return types.get(chord_type, [0, 4, 7, 10])


def guide_tone_line(roots: np.ndarray,
                    chord_types: List[str]) -> GuidetoneLine:
    n = len(roots)
    guide = np.zeros(n, dtype=int)
    for i in range(n):
        ct = chord_tones_from_type(chord_types[i])
        root = int(roots[i])
        if i == 0:
            guide[i] = root + ct[1]
        else:
            prev = int(guide[i - 1])
            best = root + ct[1]
            best_dist = abs(best - prev)
            for t in ct[1:]:
                for octave in [-12, 0, 12]:
                    candidate = root + t + octave
                    d = abs(candidate - prev)
                    if d < best_dist:
                        best = candidate
                        best_dist = d
            guide[i] = best
    return GuidetoneLine(roots, chord_types, guide)


def bebop_scale_run(root: int, scale_type: str = "dominant",
                    direction: int = -1, n_notes: int = 8) -> np.ndarray:
    scales = {
        "dominant": BEBOP_DOMINANT,
        "major": BEBOP_MAJOR,
        "dorian": BEBOP_DORIAN,
        "melodic_minor": BEBOP_MELODIC_MINOR,
    }
    scale = scales.get(scale_type, BEBOP_DOMINANT)
    pitches = []
    if direction < 0:
        idx = len(scale) - 1
        octave = 0
        for _ in range(n_notes):
            pitches.append(root + scale[idx] + 12 * octave)
            idx -= 1
            if idx < 0:
                idx = len(scale) - 1
                octave -= 1
    else:
        idx = 0
        octave = 0
        for _ in range(n_notes):
            pitches.append(root + scale[idx] + 12 * octave)
            idx += 1
            if idx >= len(scale):
                idx = 0
                octave += 1
    return np.array(pitches)


def detect_approach_tones(pitches: np.ndarray,
                          chord_tones_set: set) -> List[Tuple[int, ApproachType]]:
    approaches = []
    for i in range(len(pitches) - 1):
        p = int(pitches[i])
        target = int(pitches[i + 1])
        if p % 12 not in chord_tones_set and target % 12 in chord_tones_set:
            diff = target - p
            if diff == 1:
                approaches.append((i, ApproachType.CHROMATIC_BELOW))
            elif diff == -1:
                approaches.append((i, ApproachType.CHROMATIC_ABOVE))
            elif diff == 2:
                approaches.append((i, ApproachType.DIATONIC_BELOW))
            elif diff == -2:
                approaches.append((i, ApproachType.DIATONIC_ABOVE))
    return approaches


def detect_enclosures(pitches: np.ndarray,
                      chord_tones_set: set) -> List[Tuple[int, int]]:
    enclosures = []
    for i in range(len(pitches) - 2):
        p1 = int(pitches[i])
        p2 = int(pitches[i + 1])
        target = int(pitches[i + 2])
        if target % 12 in chord_tones_set:
            if p1 > target and p2 < target and (p1 - target) <= 3 and (target - p2) <= 3:
                enclosures.append((i, i + 2))
            elif p1 < target and p2 > target and (target - p1) <= 3 and (p2 - target) <= 3:
                enclosures.append((i, i + 2))
    return enclosures


def chromaticism_density(pitches: np.ndarray) -> float:
    if len(pitches) < 2:
        return 0.0
    intervals = np.abs(np.diff(pitches))
    chromatic_steps = np.sum(intervals == 1)
    return float(chromatic_steps / len(intervals))


def chord_tone_targeting(pitches: np.ndarray,
                         chord_tones_set: set) -> float:
    if len(pitches) == 0:
        return 0.0
    on_chord = sum(1 for p in pitches if int(p) % 12 in chord_tones_set)
    return on_chord / len(pitches)


def line_contour_variety(pitches: np.ndarray) -> float:
    if len(pitches) < 3:
        return 0.0
    directions = np.sign(np.diff(pitches.astype(float)))
    changes = np.sum(np.abs(np.diff(directions)) > 0)
    return float(changes / max(len(directions) - 1, 1))


def rhythmic_displacement(onsets: np.ndarray,
                          beat_positions: np.ndarray) -> float:
    if len(onsets) == 0 or len(beat_positions) == 0:
        return 0.0
    displaced = 0
    for onset in onsets:
        dists = np.abs(beat_positions - onset)
        if np.min(dists) > 0.1:
            displaced += 1
    return displaced / len(onsets)


def ii_v_i_lick(root: int, key: int = 0) -> np.ndarray:
    ii_root = key + 2
    v_root = key + 7
    i_root = key
    lick = []
    for ct in [0, 3, 7, 10]:
        lick.append(root + ii_root + ct)
    for ct in [11, 7, 4, 0]:
        lick.append(root + v_root + ct)
    lick.append(root + i_root + 7)
    lick.append(root + i_root + 4)
    lick.append(root + i_root)
    return np.array(lick)
