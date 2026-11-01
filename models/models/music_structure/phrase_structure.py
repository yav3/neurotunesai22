"""Musical phrase grouping and hierarchy.

Models the nested structure of musical phrases (period → sentence →
phrase → subphrase) with boundary detection and regularity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Phrase:
    start_beat: float
    end_beat: float
    level: int = 0  # 0=subphrase, 1=phrase, 2=sentence, 3=period
    children: List["Phrase"] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.end_beat - self.start_beat


class PhraseTree:
    def __init__(self, root: Phrase):
        self.root = root

    def depth(self) -> int:
        def _depth(p: Phrase) -> int:
            if not p.children:
                return 0
            return 1 + max(_depth(c) for c in p.children)
        return _depth(self.root)

    def leaf_phrases(self) -> List[Phrase]:
        def _leaves(p: Phrase) -> List[Phrase]:
            if not p.children:
                return [p]
            result: list[Phrase] = []
            for c in p.children:
                result.extend(_leaves(c))
            return result
        return _leaves(self.root)

    def all_phrases(self) -> List[Phrase]:
        def _all(p: Phrase) -> List[Phrase]:
            result = [p]
            for c in p.children:
                result.extend(_all(c))
            return result
        return _all(self.root)


def detect_phrase_boundaries(
    melody_contour: np.ndarray,
    rhythm_pattern: np.ndarray,
    gap_threshold: float = 1.5,
) -> List[int]:
    """Heuristic phrase parsing using melodic arch + rhythmic gaps.
    A boundary is declared where: (a) a large IOI gap occurs, or
    (b) the melody reaches a local minimum after a peak."""
    boundaries: list[int] = [0]
    n = len(melody_contour)

    for i in range(1, n - 1):
        is_gap = False
        if i < len(rhythm_pattern) - 1:
            is_gap = rhythm_pattern[i] > gap_threshold * np.median(rhythm_pattern)
        is_valley = melody_contour[i] < melody_contour[i - 1] and melody_contour[i] <= melody_contour[i + 1]
        if is_gap or (is_valley and i > boundaries[-1] + 3):
            boundaries.append(i)

    if boundaries[-1] != n - 1:
        boundaries.append(n - 1)
    return boundaries


def group_into_periods(phrases: List[Phrase]) -> List[Tuple[Phrase, Phrase]]:
    """Pair consecutive phrases into antecedent-consequent periods."""
    pairs: list[tuple[Phrase, Phrase]] = []
    for i in range(0, len(phrases) - 1, 2):
        pairs.append((phrases[i], phrases[i + 1]))
    return pairs


def phrase_length_regularity(phrases: List[Phrase]) -> float:
    """Measure how regular phrase lengths are (0=irregular, 1=perfectly regular)."""
    if len(phrases) < 2:
        return 1.0
    lengths = [p.length for p in phrases]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 1.0
    cv = float(np.std(lengths) / mean_len)
    return max(0.0, 1.0 - cv)


def melodic_arch_score(contour: np.ndarray) -> float:
    """How well the contour fits a rise-peak-fall arch [0,1]."""
    if len(contour) < 3:
        return 0.0
    n = len(contour)
    peak_idx = np.argmax(contour)
    ideal_peak = n * 0.55
    peak_score = 1.0 - abs(peak_idx - ideal_peak) / n
    rising = contour[:peak_idx + 1]
    falling = contour[peak_idx:]
    rise_mono = 0.0
    if len(rising) > 1:
        diffs = np.diff(rising)
        rise_mono = float(np.mean(diffs >= 0))
    fall_mono = 0.0
    if len(falling) > 1:
        diffs = np.diff(falling)
        fall_mono = float(np.mean(diffs <= 0))
    return float(np.clip((peak_score + rise_mono + fall_mono) / 3.0, 0, 1))


def build_phrase_tree(
    boundaries: List[int],
    levels: Optional[List[int]] = None,
) -> PhraseTree:
    """Construct a hierarchical phrase grouping from boundary positions."""
    if levels is None:
        levels = [0] * (len(boundaries) - 1)
    phrases: list[Phrase] = []
    for i in range(len(boundaries) - 1):
        lvl = levels[i] if i < len(levels) else 0
        phrases.append(Phrase(
            start_beat=float(boundaries[i]),
            end_beat=float(boundaries[i + 1]),
            level=lvl,
        ))
    if not phrases:
        return PhraseTree(Phrase(start_beat=0, end_beat=0, level=0))

    max_level = max(p.level for p in phrases)
    root = Phrase(
        start_beat=phrases[0].start_beat,
        end_beat=phrases[-1].end_beat,
        level=max_level + 1,
        children=phrases,
    )
    return PhraseTree(root)


def sentence_structure(phrases: List[Phrase]) -> str:
    """Classify phrase grouping as 'sentence' (a a' b) or 'period' (a b) or 'bar form' (a a b)."""
    if len(phrases) < 2:
        return "fragment"
    lengths = [p.length for p in phrases]
    if len(phrases) == 2:
        return "period"
    if len(phrases) == 3:
        if abs(lengths[0] - lengths[1]) < 0.5:
            return "sentence"
    return "through-composed"


def hypermeter_level(beat: float, beats_per_bar: int = 4, bars_per_phrase: int = 4) -> int:
    """Return the hypermetric level of a given beat position.
    Level 0=beat, 1=downbeat, 2=phrase-start, 3=section-start."""
    phrase_length = beats_per_bar * bars_per_phrase
    if beat % phrase_length == 0:
        return 2
    if beat % beats_per_bar == 0:
        return 1
    return 0
