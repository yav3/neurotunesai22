"""Lick / vocabulary pattern matching for jazz and blues improvisation.

A lick is a short melodic fragment described by its interval sequence and
rhythmic profile.  This module provides a library of common patterns and
functions to detect, classify, and score how well a passage matches known
vocabulary.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LickTemplate:
    """A named melodic vocabulary pattern."""
    name: str
    intervals: list[int]
    style: str = "jazz"
    tags: list[str] = field(default_factory=list)


# ---- Built-in vocabulary ----

BEBOP_LICKS: list[LickTemplate] = [
    LickTemplate("bebop_enclosure_above", [1, -2, 1], "bebop", ["enclosure"]),
    LickTemplate("bebop_enclosure_below", [-1, 2, -1], "bebop", ["enclosure"]),
    LickTemplate("bebop_1235", [2, 2, 1, 2], "bebop", ["scale_run"]),
    LickTemplate("bebop_chromatic_approach", [1, 1, 1, -3], "bebop", ["chromatic"]),
    LickTemplate("bebop_turn", [2, -1, -2, 1], "bebop", ["ornament"]),
    LickTemplate("honeysuckle_rose", [0, 1, 0, -1], "bebop", ["classic"]),
    LickTemplate("cry_me_a_river_descent", [-1, -2, -2, -1], "bebop", ["descending"]),
]

BLUES_LICKS: list[LickTemplate] = [
    LickTemplate("blues_bend_release", [3, -1, -2], "blues", ["bend"]),
    LickTemplate("blues_minor_pent_run", [3, 2, 2, 3], "blues", ["pentatonic"]),
    LickTemplate("blues_b5_chromatic", [1, 1, -2], "blues", ["chromatic", "blue_note"]),
    LickTemplate("blues_call", [3, 2, -2, -3], "blues", ["call"]),
    LickTemplate("turnaround_IV_I", [-5, 7, -2, -3], "blues", ["turnaround"]),
]

COMMON_PATTERNS: list[LickTemplate] = [
    LickTemplate("arpeggio_up_maj", [4, 3, 5], "general", ["arpeggio"]),
    LickTemplate("arpeggio_up_min", [3, 4, 5], "general", ["arpeggio"]),
    LickTemplate("arpeggio_dom7", [4, 3, 3], "general", ["arpeggio"]),
    LickTemplate("scalar_ascending", [2, 2, 1, 2, 2], "general", ["scale_run"]),
    LickTemplate("scalar_descending", [-2, -2, -1, -2, -2], "general", ["scale_run"]),
    LickTemplate("chromatic_run_up", [1, 1, 1, 1], "general", ["chromatic"]),
    LickTemplate("chromatic_run_down", [-1, -1, -1, -1], "general", ["chromatic"]),
    LickTemplate("tritone_sub_approach", [-1, 6, -1], "general", ["tritone"]),
    LickTemplate("pentatonic_box", [2, 3, 2, 3], "general", ["pentatonic"]),
    LickTemplate("whole_tone_fragment", [2, 2, 2], "general", ["whole_tone"]),
]

ALL_TEMPLATES = BEBOP_LICKS + BLUES_LICKS + COMMON_PATTERNS


def _intervals(pitches: list[int]) -> list[int]:
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def match_score(intervals_a: list[int], intervals_b: list[int]) -> float:
    """Normalised similarity between two interval sequences (0-1).

    Uses a weighted combination of exact-match ratio and magnitude distance.
    """
    if not intervals_a or not intervals_b:
        return 0.0
    n = min(len(intervals_a), len(intervals_b))
    exact = sum(1 for i in range(n) if intervals_a[i] == intervals_b[i])
    mag = sum(abs(intervals_a[i] - intervals_b[i]) for i in range(n))
    max_mag = n * 12
    exact_ratio = exact / n
    mag_ratio = 1.0 - mag / max(max_mag, 1)
    length_penalty = n / max(len(intervals_a), len(intervals_b))
    return 0.5 * exact_ratio + 0.3 * mag_ratio + 0.2 * length_penalty


@dataclass
class LickMatch:
    """Result of scanning a passage against the lick library."""
    template: LickTemplate
    position: int
    score: float


def find_licks(
    pitches: list[int],
    templates: list[LickTemplate] | None = None,
    threshold: float = 0.6,
) -> list[LickMatch]:
    """Scan a pitch sequence for occurrences of known lick patterns.

    Slides each template across the passage and records positions where the
    match score exceeds *threshold*.
    """
    if templates is None:
        templates = ALL_TEMPLATES
    if len(pitches) < 2:
        return []
    iv = _intervals(pitches)
    matches: list[LickMatch] = []
    for tpl in templates:
        tpl_len = len(tpl.intervals)
        if tpl_len > len(iv):
            continue
        for start in range(len(iv) - tpl_len + 1):
            sub = iv[start: start + tpl_len]
            s = match_score(sub, tpl.intervals)
            if s >= threshold:
                matches.append(LickMatch(template=tpl, position=start, score=s))
    matches.sort(key=lambda m: -m.score)
    return matches


def vocabulary_richness(
    pitches: list[int],
    templates: list[LickTemplate] | None = None,
    threshold: float = 0.6,
) -> dict[str, float]:
    """Summarise how much of a passage is covered by known vocabulary.

    Returns
    -------
    dict
        ``coverage``: fraction of intervals inside at least one matched lick.
        ``unique_patterns``: number of distinct templates matched.
        ``style_distribution``: {style: count}.
    """
    if len(pitches) < 2:
        return {"coverage": 0.0, "unique_patterns": 0, "style_distribution": {}}
    matches = find_licks(pitches, templates, threshold)
    total_iv = len(pitches) - 1
    covered = set()
    style_counts: dict[str, int] = {}
    unique_names: set[str] = set()
    for m in matches:
        tpl_len = len(m.template.intervals)
        for k in range(m.position, m.position + tpl_len):
            covered.add(k)
        unique_names.add(m.template.name)
        style_counts[m.template.style] = style_counts.get(m.template.style, 0) + 1
    return {
        "coverage": len(covered) / max(total_iv, 1),
        "unique_patterns": len(unique_names),
        "style_distribution": style_counts,
    }


def suggest_next_lick(
    recent_pitches: list[int],
    templates: list[LickTemplate] | None = None,
    top_n: int = 3,
) -> list[LickTemplate]:
    """Given a recent pitch context, suggest vocabulary that could follow.

    Ranks templates by how well their first interval matches the current
    melodic direction and interval magnitude.
    """
    if templates is None:
        templates = ALL_TEMPLATES
    if len(recent_pitches) < 2:
        return templates[:top_n]
    last_iv = recent_pitches[-1] - recent_pitches[-2]

    def affinity(tpl: LickTemplate) -> float:
        if not tpl.intervals:
            return 0.0
        first = tpl.intervals[0]
        direction_match = 1.0 if (first > 0) == (last_iv > 0) else 0.0
        mag_sim = 1.0 - abs(abs(first) - abs(last_iv)) / 12
        return 0.6 * direction_match + 0.4 * mag_sim

    ranked = sorted(templates, key=affinity, reverse=True)
    return ranked[:top_n]
