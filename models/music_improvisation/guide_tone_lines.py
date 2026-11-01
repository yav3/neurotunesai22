"""Guide-tone voice leading through chord changes.

Target-note approach patterns, bebop chromatic approach, guide-tone
contour analysis, and enclosure patterns for improvisation practice.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_CHORD_TONES: Dict[str, List[int]] = {
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "aug7": [0, 4, 8, 10],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "sus4": [0, 5, 7, 10],
    "sus2": [0, 2, 7, 10],
}


class ApproachType(Enum):
    STEP_ABOVE = "step_above"
    STEP_BELOW = "step_below"
    CHROMATIC_ABOVE = "chromatic_above"
    CHROMATIC_BELOW = "chromatic_below"
    ENCLOSURE_ABOVE_FIRST = "enclosure_above_first"
    ENCLOSURE_BELOW_FIRST = "enclosure_below_first"
    DOUBLE_CHROMATIC_ABOVE = "double_chromatic_above"
    DOUBLE_CHROMATIC_BELOW = "double_chromatic_below"
    DIRECT = "direct"


@dataclass
class GuideTone:
    pitch_class: int
    chord_degree: int
    chord_root: int
    chord_quality: str


@dataclass
class ApproachPattern:
    target: int
    approach_type: ApproachType
    notes: List[int]


@dataclass
class GuideToneLine:
    guide_tones: List[GuideTone]
    intervals: List[int]
    total_motion: int
    smoothness: float
    contour: List[int]


def chord_tones(root: int, quality: str) -> List[int]:
    """Pitch classes for a chord."""
    intervals = _CHORD_TONES.get(quality, [0, 4, 7])
    return [(root + i) % 12 for i in intervals]


def guide_tones_for_chord(root: int, quality: str) -> List[int]:
    """3rd and 7th (the guide tones) for a chord quality."""
    intervals = _CHORD_TONES.get(quality, [0, 4, 7])
    third = None
    seventh = None
    for i in intervals:
        if i in (3, 4):
            third = (root + i) % 12
        if i in (9, 10, 11):
            seventh = (root + i) % 12
    result = []
    if third is not None:
        result.append(third)
    if seventh is not None:
        result.append(seventh)
    if not result:
        result.append((root + intervals[1]) % 12 if len(intervals) > 1 else root)
    return result


def _voice_lead_interval(a: int, b: int) -> int:
    """Signed interval in semitones, shortest path."""
    diff = (b - a) % 12
    if diff > 6:
        return diff - 12
    return diff


def build_guide_tone_line(
    roots: List[int],
    qualities: List[str],
    prefer_seventh: bool = True,
) -> GuideToneLine:
    """Build a smooth guide-tone line through a chord progression.

    Selects the 3rd or 7th of each chord that minimises voice-leading
    distance from the previous guide tone.
    """
    if not roots:
        return GuideToneLine([], [], 0, 1.0, [])

    guide_tone_list: List[GuideTone] = []
    for idx, (r, q) in enumerate(zip(roots, qualities)):
        candidates = guide_tones_for_chord(r, q)
        if idx == 0:
            pick = candidates[-1] if prefer_seventh and len(candidates) > 1 else candidates[0]
        else:
            prev = guide_tone_list[-1].pitch_class
            pick = min(candidates, key=lambda c: abs(_voice_lead_interval(prev, c)))

        intervals_list = _CHORD_TONES.get(q, [0, 4, 7])
        degree = (pick - r) % 12
        gt = GuideTone(pitch_class=pick, chord_degree=degree, chord_root=r, chord_quality=q)
        guide_tone_list.append(gt)

    intervals = []
    for i in range(1, len(guide_tone_list)):
        intervals.append(_voice_lead_interval(
            guide_tone_list[i - 1].pitch_class,
            guide_tone_list[i].pitch_class,
        ))

    total = sum(abs(iv) for iv in intervals)
    smoothness = 1.0 / (1.0 + total / max(len(intervals), 1))
    contour = [0] + [1 if iv > 0 else (-1 if iv < 0 else 0) for iv in intervals]

    return GuideToneLine(
        guide_tones=guide_tone_list,
        intervals=intervals,
        total_motion=total,
        smoothness=smoothness,
        contour=contour,
    )


def approach_from_above(target: int, chromatic: bool = False) -> ApproachPattern:
    """Single note approach from above."""
    offset = 1 if chromatic else 2
    note = (target + offset) % 12
    atype = ApproachType.CHROMATIC_ABOVE if chromatic else ApproachType.STEP_ABOVE
    return ApproachPattern(target=target, approach_type=atype, notes=[note, target])


def approach_from_below(target: int, chromatic: bool = False) -> ApproachPattern:
    """Single note approach from below."""
    offset = 1 if chromatic else 2
    note = (target - offset) % 12
    atype = ApproachType.CHROMATIC_BELOW if chromatic else ApproachType.STEP_BELOW
    return ApproachPattern(target=target, approach_type=atype, notes=[note, target])


def enclosure(target: int, above_first: bool = True) -> ApproachPattern:
    """Enclosure pattern: chromatic above and below target."""
    above = (target + 1) % 12
    below = (target - 1) % 12
    if above_first:
        return ApproachPattern(
            target=target,
            approach_type=ApproachType.ENCLOSURE_ABOVE_FIRST,
            notes=[above, below, target],
        )
    return ApproachPattern(
        target=target,
        approach_type=ApproachType.ENCLOSURE_BELOW_FIRST,
        notes=[below, above, target],
    )


def double_chromatic_approach(target: int, from_above: bool = True) -> ApproachPattern:
    """Two chromatic notes leading to target."""
    if from_above:
        return ApproachPattern(
            target=target,
            approach_type=ApproachType.DOUBLE_CHROMATIC_ABOVE,
            notes=[(target + 2) % 12, (target + 1) % 12, target],
        )
    return ApproachPattern(
        target=target,
        approach_type=ApproachType.DOUBLE_CHROMATIC_BELOW,
        notes=[(target - 2) % 12, (target - 1) % 12, target],
    )


def bebop_approach(target: int, chord_root: int, chord_quality: str) -> ApproachPattern:
    """Bebop-style approach using chord tone + chromatic note + target."""
    ct = chord_tones(chord_root, chord_quality)
    above = (target + 1) % 12
    best_ct = min(ct, key=lambda c: abs(_voice_lead_interval(c, above)))
    return ApproachPattern(
        target=target,
        approach_type=ApproachType.ENCLOSURE_ABOVE_FIRST,
        notes=[best_ct, above, target],
    )


def target_note_line(
    roots: List[int],
    qualities: List[str],
    approach_types: Optional[List[ApproachType]] = None,
) -> List[ApproachPattern]:
    """Build a line of approach patterns to guide tones."""
    gtl = build_guide_tone_line(roots, qualities)
    patterns = []
    for i, gt in enumerate(gtl.guide_tones):
        if approach_types and i < len(approach_types):
            atype = approach_types[i]
        else:
            atype = ApproachType.CHROMATIC_ABOVE if i % 2 == 0 else ApproachType.CHROMATIC_BELOW

        if atype == ApproachType.CHROMATIC_ABOVE:
            patterns.append(approach_from_above(gt.pitch_class, chromatic=True))
        elif atype == ApproachType.CHROMATIC_BELOW:
            patterns.append(approach_from_below(gt.pitch_class, chromatic=True))
        elif atype == ApproachType.STEP_ABOVE:
            patterns.append(approach_from_above(gt.pitch_class, chromatic=False))
        elif atype == ApproachType.STEP_BELOW:
            patterns.append(approach_from_below(gt.pitch_class, chromatic=False))
        elif atype in (ApproachType.ENCLOSURE_ABOVE_FIRST, ApproachType.ENCLOSURE_BELOW_FIRST):
            patterns.append(enclosure(gt.pitch_class, atype == ApproachType.ENCLOSURE_ABOVE_FIRST))
        elif atype in (ApproachType.DOUBLE_CHROMATIC_ABOVE, ApproachType.DOUBLE_CHROMATIC_BELOW):
            patterns.append(double_chromatic_approach(gt.pitch_class, atype == ApproachType.DOUBLE_CHROMATIC_ABOVE))
        else:
            patterns.append(ApproachPattern(target=gt.pitch_class, approach_type=ApproachType.DIRECT, notes=[gt.pitch_class]))
    return patterns


def contour_tension(contour: List[int]) -> float:
    """Fraction of direction changes in a contour (0=monotone, 1=alternating)."""
    if len(contour) < 3:
        return 0.0
    changes = 0
    filtered = [c for c in contour if c != 0]
    for i in range(1, len(filtered)):
        if filtered[i] != filtered[i - 1]:
            changes += 1
    return changes / max(len(filtered) - 1, 1)


def line_range(guide_tone_line: GuideToneLine) -> int:
    """Semitone span of the guide-tone line (mod-12 aware)."""
    if not guide_tone_line.guide_tones:
        return 0
    pcs = [gt.pitch_class for gt in guide_tone_line.guide_tones]
    running = [pcs[0]]
    for iv in guide_tone_line.intervals:
        running.append(running[-1] + iv)
    return max(running) - min(running)


def common_tone_connections(
    roots: List[int], qualities: List[str],
) -> List[Optional[int]]:
    """For each adjacent chord pair, find a common tone (or None)."""
    result: List[Optional[int]] = []
    for i in range(len(roots) - 1):
        ct_a = set(chord_tones(roots[i], qualities[i]))
        ct_b = set(chord_tones(roots[i + 1], qualities[i + 1]))
        common = ct_a & ct_b
        result.append(min(common) if common else None)
    return result
