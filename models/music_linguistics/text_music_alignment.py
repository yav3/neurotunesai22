"""Text-music alignment: syllable-to-note mapping, stress matching, melismatic detection.

Aligns linguistic structure (syllables, stress patterns, phrase boundaries)
to musical structure (notes, beats, phrase arcs) for analysis of text setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AlignmentType(str, Enum):
    SYLLABIC = "syllabic"
    MELISMATIC = "melismatic"
    RECITATIVE = "recitative"


@dataclass
class SyllableNote:
    syllable: str
    note_pitches: list[int]
    onset: float
    duration: float
    stressed: bool = False

    @property
    def alignment_type(self) -> AlignmentType:
        if len(self.note_pitches) == 1:
            return AlignmentType.SYLLABIC
        elif len(self.note_pitches) > 2:
            return AlignmentType.MELISMATIC
        return AlignmentType.SYLLABIC


@dataclass
class AlignmentScore:
    stress_match: float
    melismatic_ratio: float
    phrase_alignment: float
    overall: float


def syllable_count(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    vowels = set("aeiouy")
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def simple_stress_pattern(word: str) -> list[bool]:
    n = syllable_count(word)
    if n <= 1:
        return [True]
    if n == 2:
        return [True, False]
    pattern = [False] * n
    pattern[0] = True
    if n >= 3:
        pattern[2] = True
    return pattern


def stress_match_score(syllable_stresses: list[bool],
                       beat_strengths: np.ndarray) -> float:
    n = min(len(syllable_stresses), len(beat_strengths))
    if n == 0:
        return 0.0
    matches = 0
    for i in range(n):
        is_strong_beat = beat_strengths[i] > np.median(beat_strengths)
        if syllable_stresses[i] == is_strong_beat:
            matches += 1
    return matches / n


def melismatic_ratio(syllable_notes: list[SyllableNote]) -> float:
    if not syllable_notes:
        return 0.0
    melismatic = sum(1 for sn in syllable_notes if len(sn.note_pitches) > 1)
    return melismatic / len(syllable_notes)


def detect_melismas(pitches: np.ndarray, syllable_boundaries: list[int]) -> list[tuple[int, int, int]]:
    melismas = []
    for i in range(len(syllable_boundaries) - 1):
        start = syllable_boundaries[i]
        end = syllable_boundaries[i + 1]
        n_notes = end - start
        if n_notes > 2:
            melismas.append((i, start, n_notes))
    return melismas


def phrase_boundary_alignment(text_boundaries: np.ndarray,
                               music_boundaries: np.ndarray,
                               tolerance: float = 0.5) -> float:
    if len(text_boundaries) == 0 or len(music_boundaries) == 0:
        return 0.0
    aligned = 0
    for tb in text_boundaries:
        dists = np.abs(music_boundaries - tb)
        if np.min(dists) <= tolerance:
            aligned += 1
    return aligned / len(text_boundaries)


def compute_alignment_score(syllable_notes: list[SyllableNote],
                             beat_strengths: np.ndarray,
                             text_phrase_bounds: np.ndarray | None = None,
                             music_phrase_bounds: np.ndarray | None = None) -> AlignmentScore:
    stresses = [sn.stressed for sn in syllable_notes]
    n_beats = min(len(stresses), len(beat_strengths))
    sm = stress_match_score(stresses[:n_beats], beat_strengths[:n_beats])
    mr = melismatic_ratio(syllable_notes)
    pa = 0.5
    if text_phrase_bounds is not None and music_phrase_bounds is not None:
        pa = phrase_boundary_alignment(text_phrase_bounds, music_phrase_bounds)
    overall = 0.4 * sm + 0.3 * (1.0 - abs(mr - 0.15)) + 0.3 * pa
    return AlignmentScore(
        stress_match=sm,
        melismatic_ratio=mr,
        phrase_alignment=pa,
        overall=float(np.clip(overall, 0, 1)),
    )


def word_painting_candidates(words: list[str], pitches: np.ndarray,
                              word_onsets: list[int]) -> list[dict]:
    ascending_words = {"rise", "up", "high", "soar", "ascend", "climb", "heaven", "sky"}
    descending_words = {"fall", "down", "low", "descend", "drop", "deep", "earth"}

    candidates = []
    for i, word in enumerate(words):
        w = word.lower().strip(".,!?;:")
        start = word_onsets[i] if i < len(word_onsets) else 0
        end = word_onsets[i + 1] if i + 1 < len(word_onsets) else len(pitches)
        segment = pitches[start:end]
        if len(segment) < 2:
            continue

        direction = segment[-1] - segment[0]
        if w in ascending_words and direction > 2:
            candidates.append({"word": w, "type": "ascending", "interval": float(direction)})
        elif w in descending_words and direction < -2:
            candidates.append({"word": w, "type": "descending", "interval": float(direction)})
    return candidates
