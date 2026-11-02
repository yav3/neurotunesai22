"""Narrative arc analysis for song lyrics.

Maps lyric sections to story structure (exposition, rising action, climax, etc.).
Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class NarrativePhase(Enum):
    EXPOSITION = "exposition"
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    REFLECTION = "reflection"


class SectionType(Enum):
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    PRE_CHORUS = "pre_chorus"
    OUTRO = "outro"
    INTRO = "intro"


@dataclass
class LyricSection:
    section_type: SectionType
    lines: list[str]
    index: int
    emotional_intensity: float = 0.0
    narrative_phase: NarrativePhase = NarrativePhase.EXPOSITION


@dataclass
class NarrativeArc:
    sections: list[LyricSection]
    intensity_curve: np.ndarray
    climax_index: int
    arc_type: str


INTENSITY_WORDS = {
    "high": {"scream", "fire", "burn", "explode", "thunder", "crash", "shout",
             "rage", "fury", "storm", "war", "fight", "blood", "destroy",
             "forever", "always", "never", "die", "kill", "hate", "pain"},
    "medium": {"run", "dance", "dream", "cry", "fall", "rise", "fly",
               "love", "heart", "soul", "feel", "hold", "break", "wish",
               "need", "want", "hope", "fear", "lost", "found"},
    "low": {"walk", "sit", "think", "know", "see", "look", "wait",
            "slow", "quiet", "soft", "gentle", "calm", "peace", "still",
            "home", "room", "door", "window", "morning", "evening"},
}


def _line_intensity(line: str) -> float:
    tokens = set(line.lower().split())
    high = len(tokens & INTENSITY_WORDS["high"])
    med = len(tokens & INTENSITY_WORDS["medium"])
    low = len(tokens & INTENSITY_WORDS["low"])
    n = max(len(tokens), 1)
    return float((high * 1.0 + med * 0.5 + low * 0.2) / n)


def section_intensity(lines: Sequence[str]) -> float:
    """Average emotional intensity of a section's lines."""
    if not lines:
        return 0.0
    return float(np.mean([_line_intensity(l) for l in lines]))


def detect_repetition_ratio(lines: Sequence[str]) -> float:
    """Fraction of lines that are repeated within the section."""
    if len(lines) <= 1:
        return 0.0
    normalized = [l.lower().strip() for l in lines]
    unique = set(normalized)
    return 1.0 - len(unique) / len(normalized)


def assign_narrative_phases(sections: list[LyricSection]) -> list[LyricSection]:
    """Assign narrative phases based on intensity curve shape."""
    if not sections:
        return sections

    intensities = np.array([s.emotional_intensity for s in sections])
    n = len(sections)

    if n == 0:
        return sections

    climax_idx = int(np.argmax(intensities))

    for i, sec in enumerate(sections):
        if i < climax_idx:
            progress = i / max(climax_idx, 1)
            if progress < 0.4:
                sec.narrative_phase = NarrativePhase.EXPOSITION
            else:
                sec.narrative_phase = NarrativePhase.RISING_ACTION
        elif i == climax_idx:
            sec.narrative_phase = NarrativePhase.CLIMAX
        else:
            remaining = (i - climax_idx) / max(n - climax_idx - 1, 1)
            if remaining < 0.5:
                sec.narrative_phase = NarrativePhase.FALLING_ACTION
            else:
                sec.narrative_phase = NarrativePhase.RESOLUTION

    return sections


def classify_arc_type(intensity_curve: np.ndarray) -> str:
    """Classify the overall arc shape: ramp, peak, valley, flat, double-peak."""
    if len(intensity_curve) < 3:
        return "flat"

    normalized = intensity_curve - np.mean(intensity_curve)
    peaks = []
    for i in range(1, len(normalized) - 1):
        if normalized[i] > normalized[i - 1] and normalized[i] > normalized[i + 1]:
            peaks.append(i)

    if len(peaks) >= 2:
        return "double_peak"

    trend = np.polyfit(np.arange(len(normalized)), normalized, 1)[0]
    if abs(trend) < 0.02:
        if len(peaks) == 1 and peaks[0] > len(normalized) * 0.3:
            return "peak"
        return "flat"
    elif trend > 0:
        return "ramp_up"
    else:
        return "ramp_down"


def build_narrative_arc(sections: list[LyricSection]) -> NarrativeArc:
    """Full narrative arc analysis from labeled sections."""
    for sec in sections:
        sec.emotional_intensity = section_intensity(sec.lines)

    sections = assign_narrative_phases(sections)
    intensities = np.array([s.emotional_intensity for s in sections])
    climax_idx = int(np.argmax(intensities)) if len(intensities) > 0 else 0
    arc_type = classify_arc_type(intensities)

    return NarrativeArc(
        sections=sections,
        intensity_curve=intensities,
        climax_index=climax_idx,
        arc_type=arc_type,
    )


def thematic_coherence(sections: list[LyricSection]) -> float:
    """Measure how much vocabulary overlaps across sections (0 = disjoint, 1 = identical)."""
    if len(sections) < 2:
        return 1.0
    vocab_sets = []
    for sec in sections:
        words = set()
        for line in sec.lines:
            words.update(line.lower().split())
        vocab_sets.append(words)

    overlaps = []
    for i in range(len(vocab_sets)):
        for j in range(i + 1, len(vocab_sets)):
            union = vocab_sets[i] | vocab_sets[j]
            inter = vocab_sets[i] & vocab_sets[j]
            if union:
                overlaps.append(len(inter) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.0
