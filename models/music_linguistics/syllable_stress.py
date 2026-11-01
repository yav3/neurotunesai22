"""Syllable stress analysis for text-setting in vocal music.

Maps English syllable stress patterns onto metric grids.
Pure numpy / regex.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import re

import numpy as np


class StressLevel(Enum):
    UNSTRESSED = 0
    SECONDARY = 1
    PRIMARY = 2


@dataclass
class SyllableInfo:
    text: str
    stress: StressLevel
    position: int


@dataclass
class StressPattern:
    syllables: list[SyllableInfo]
    pattern_string: str


COMMON_STRESS: dict[str, list[int]] = {
    "music": [2, 0], "melody": [2, 0, 0], "rhythm": [2, 0],
    "harmony": [2, 0, 0], "love": [2], "baby": [2, 0],
    "forever": [0, 2, 0], "together": [0, 2, 0],
    "beautiful": [2, 0, 0], "remember": [0, 2, 0],
    "tonight": [0, 2], "again": [0, 2], "away": [0, 2],
    "never": [2, 0], "always": [2, 1], "everything": [2, 0, 0],
    "nothing": [2, 0], "dancing": [2, 0], "running": [2, 0],
    "believe": [0, 2], "desire": [0, 2], "emotion": [0, 2, 0],
    "heaven": [2, 0], "shadow": [2, 0], "fire": [2, 0],
    "water": [2, 0], "river": [2, 0], "thunder": [2, 0],
    "wonder": [2, 0], "freedom": [2, 0],
}


def estimate_syllable_count(word: str) -> int:
    """Heuristic syllable count from vowel clusters."""
    word = word.lower().strip()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def lookup_stress(word: str) -> list[StressLevel]:
    """Look up stress pattern; fall back to heuristic."""
    w = word.lower().strip(".,!?;:'\"()-")
    if w in COMMON_STRESS:
        return [StressLevel(s) for s in COMMON_STRESS[w]]
    n = estimate_syllable_count(w)
    if n == 1:
        return [StressLevel.PRIMARY]
    pattern = []
    for i in range(n):
        if i == 0:
            pattern.append(StressLevel.PRIMARY)
        elif i % 2 == 0:
            pattern.append(StressLevel.SECONDARY)
        else:
            pattern.append(StressLevel.UNSTRESSED)
    return pattern


def line_stress_pattern(text: str) -> StressPattern:
    """Build a full stress pattern for a line of text."""
    words = re.findall(r"[a-zA-Z']+", text)
    syllables: list[SyllableInfo] = []
    pos = 0
    for w in words:
        stresses = lookup_stress(w)
        for i, s in enumerate(stresses):
            syllables.append(SyllableInfo(text=w if i == 0 else "", stress=s, position=pos))
            pos += 1
    pattern_str = "".join(str(s.stress.value) for s in syllables)
    return StressPattern(syllables=syllables, pattern_string=pattern_str)


def metric_alignment_score(
    stress_pattern: str, metric_grid: str,
) -> float:
    """Score alignment between syllable stress and a metric grid.

    Both are strings of 0/1/2; score = proportion of matching positions.
    """
    n = min(len(stress_pattern), len(metric_grid))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if stress_pattern[i] == metric_grid[i])
    return matches / n


def generate_metric_grid(
    beats_per_bar: int, subdivisions: int = 2, n_bars: int = 2,
) -> str:
    """Generate a metric stress grid: 2=downbeat, 1=beat, 0=weak."""
    grid = []
    for bar in range(n_bars):
        for beat in range(beats_per_bar):
            for sub in range(subdivisions):
                if sub == 0 and beat == 0:
                    grid.append("2")
                elif sub == 0:
                    grid.append("1")
                else:
                    grid.append("0")
    return "".join(grid)


def stress_mismatch_positions(
    stress_pattern: str, metric_grid: str,
) -> list[int]:
    """Find positions where strong syllable lands on weak beat or vice versa."""
    mismatches = []
    n = min(len(stress_pattern), len(metric_grid))
    for i in range(n):
        s = int(stress_pattern[i])
        m = int(metric_grid[i])
        if (s == 2 and m == 0) or (s == 0 and m == 2):
            mismatches.append(i)
    return mismatches


def prosodic_naturalness(lines: Sequence[str], time_sig: tuple[int, int] = (4, 4)) -> float:
    """Average metric alignment across multiple lines."""
    if not lines:
        return 0.0
    beats, _ = time_sig
    grid = generate_metric_grid(beats, subdivisions=2, n_bars=2)
    scores = []
    for line in lines:
        pat = line_stress_pattern(line)
        score = metric_alignment_score(pat.pattern_string, grid)
        scores.append(score)
    return float(np.mean(scores))
