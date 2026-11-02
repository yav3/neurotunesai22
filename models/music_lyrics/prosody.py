"""Lyric prosody analysis: stress alignment, syllabification, text-music fit.

Analyses how lyrical text aligns with musical rhythm, detecting mismatches
between linguistic stress and metric accent, measuring syllable-note
alignment quality, and computing prosodic fitness scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Syllable:
    text: str
    stressed: bool
    weight: float = 1.0


@dataclass(frozen=True)
class MetricPosition:
    beat: int
    subdivision: int
    strength: float


@dataclass
class ProsodyAlignment:
    syllable: Syllable
    position: MetricPosition
    fit_score: float

    @property
    def is_mismatch(self) -> bool:
        return (self.syllable.stressed and self.position.strength < 0.5) or \
               (not self.syllable.stressed and self.position.strength > 0.8)


def syllabify(word: str) -> List[str]:
    """Split a word into approximate syllables using vowel-cluster heuristic."""
    vowels = set("aeiouyAEIOUY")
    syllables = []
    current = []
    prev_vowel = False

    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            if current and prev_vowel is False and len(current) > 1:
                syllables.append("".join(current[:-1]))
                current = [current[-1]]
        current.append(ch)
        prev_vowel = is_v

    if current:
        if syllables:
            syllables.append("".join(current))
        else:
            syllables.append("".join(current))

    return syllables if syllables else [word]


def estimate_stress(syllables: List[str]) -> List[bool]:
    """Estimate stress pattern for a list of syllables (heuristic)."""
    n = len(syllables)
    if n == 0:
        return []
    if n == 1:
        return [True]
    if n == 2:
        return [True, False]
    stress = [False] * n
    stress[0] = True
    if n >= 3:
        stress[-2] = True
    return stress


def metric_strength_profile(beats_per_bar: int,
                            subdivisions: int = 2) -> List[float]:
    """Generate metric strength values for each subdivision in a bar."""
    total = beats_per_bar * subdivisions
    strengths = []
    for i in range(total):
        if i == 0:
            s = 1.0
        elif i % subdivisions == 0:
            s = 0.7
        elif i % (subdivisions // 2) == 0 and subdivisions > 1:
            s = 0.4
        else:
            s = 0.2
        strengths.append(s)
    return strengths


def align_syllables_to_beats(
    syllables: List[Syllable],
    positions: List[MetricPosition],
) -> List[ProsodyAlignment]:
    """Align syllables to metric positions and score each pair."""
    alignments = []
    n = min(len(syllables), len(positions))
    for i in range(n):
        syl = syllables[i]
        pos = positions[i]
        if syl.stressed and pos.strength >= 0.5:
            fit = 1.0
        elif not syl.stressed and pos.strength < 0.5:
            fit = 0.8
        elif syl.stressed and pos.strength < 0.3:
            fit = 0.1
        else:
            fit = 0.5
        alignments.append(ProsodyAlignment(syl, pos, fit))
    return alignments


def prosody_score(alignments: List[ProsodyAlignment]) -> float:
    """Overall prosodic fitness in [0, 1]."""
    if not alignments:
        return 0.0
    return float(np.mean([a.fit_score for a in alignments]))


def mismatch_count(alignments: List[ProsodyAlignment]) -> int:
    """Count stress-metric mismatches."""
    return sum(1 for a in alignments if a.is_mismatch)


def mismatch_ratio(alignments: List[ProsodyAlignment]) -> float:
    """Fraction of alignments that are mismatches."""
    if not alignments:
        return 0.0
    return mismatch_count(alignments) / len(alignments)


def syllable_density(text: str, duration_beats: float) -> float:
    """Syllables per beat for a text passage."""
    words = text.split()
    total_syls = sum(len(syllabify(w)) for w in words)
    if duration_beats <= 0:
        return 0.0
    return total_syls / duration_beats


def melisma_ratio(n_syllables: int, n_notes: int) -> float:
    """Ratio of notes to syllables (>1 = melismatic, =1 = syllabic)."""
    if n_syllables == 0:
        return 0.0
    return n_notes / n_syllables


def classify_text_setting(n_syllables: int, n_notes: int) -> str:
    """Classify as syllabic, neumatic, or melismatic."""
    ratio = melisma_ratio(n_syllables, n_notes)
    if ratio <= 1.1:
        return "syllabic"
    elif ratio <= 3.0:
        return "neumatic"
    else:
        return "melismatic"


def stress_contour(text: str) -> np.ndarray:
    """Binary stress contour for all syllables in a text."""
    words = text.split()
    contour = []
    for word in words:
        syls = syllabify(word)
        stresses = estimate_stress(syls)
        contour.extend(stresses)
    return np.array(contour, dtype=float)


def contour_correlation(stress: np.ndarray, melody_contour: np.ndarray) -> float:
    """Pearson correlation between stress pattern and melodic contour."""
    n = min(len(stress), len(melody_contour))
    if n < 2:
        return 0.0
    s = stress[:n]
    m = melody_contour[:n]
    s_std = np.std(s)
    m_std = np.std(m)
    if s_std < 1e-12 or m_std < 1e-12:
        return 0.0
    return float(np.corrcoef(s, m)[0, 1])
