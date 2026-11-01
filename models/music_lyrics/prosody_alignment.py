"""Prosody-melody alignment: matching lyrical stress to musical rhythm.

Aligns syllable stress patterns with metrical accent structure,
matches speech intonation contours to melodic contour, and optimises
text underlay via dynamic programming.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


VOWELS = set("aeiouyAEIOUY")
STRESSED_SUFFIXES = ["tion", "sion", "ment", "ness", "ful", "less", "ing", "ous", "ive", "ble"]
UNSTRESSED_SUFFIXES = ["a", "the", "an", "of", "to", "in", "is", "it", "at", "by"]


@dataclass(frozen=True)
class Syllable:
    text: str
    stress_level: int = 0
    duration_weight: float = 1.0


@dataclass
class ProsodyAlignment:
    syllables: list[Syllable]
    note_onsets: np.ndarray
    alignment_map: list[int]
    stress_violations: int = 0
    melisma_count: int = 0


@dataclass
class MelodyContour:
    pitches: np.ndarray
    times: np.ndarray


def syllabify(word: str) -> list[str]:
    w = word.lower().strip()
    if not w:
        return []
    syllables: list[str] = []
    current = ""
    prev_vowel = False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_vowel:
            if current and prev_vowel is False and syllables:
                pass
            current += ch
            prev_vowel = True
        elif not is_v and prev_vowel:
            current += ch
            prev_vowel = False
        elif is_v and prev_vowel:
            syllables.append(current)
            current = ch
            prev_vowel = True
        else:
            current += ch
    if current:
        if syllables and not any(c in VOWELS for c in current):
            syllables[-1] += current
        else:
            syllables.append(current)
    if not syllables:
        syllables = [w]
    return syllables


def stress_pattern(word: str) -> list[int]:
    syls = syllabify(word)
    if not syls:
        return []
    if word.lower() in UNSTRESSED_SUFFIXES:
        return [0] * len(syls)
    if len(syls) == 1:
        return [2]
    stresses = [1] * len(syls)
    stresses[0] = 2
    for i, s in enumerate(syls):
        for suf in STRESSED_SUFFIXES:
            if s.endswith(suf):
                stresses[i] = 2
    if len(stresses) > 1:
        for i in range(1, len(stresses)):
            if stresses[i] == 2 and stresses[i - 1] == 2:
                stresses[i - 1] = 1
    return stresses


def _metrical_weight(onset: float, beats_per_bar: float = 4.0) -> float:
    beat = onset % beats_per_bar
    if abs(beat) < 0.01:
        return 1.0
    if abs(beat - beats_per_bar / 2) < 0.01:
        return 0.7
    if abs(beat % 1.0) < 0.01:
        return 0.5
    return 0.2


def align_syllables_to_beats(
    syllables: Sequence[Syllable],
    note_onsets: np.ndarray,
    beats_per_bar: float = 4.0,
) -> ProsodyAlignment:
    onsets = np.asarray(note_onsets, dtype=np.float64)
    n_syl = len(syllables)
    n_notes = len(onsets)
    if n_syl == 0 or n_notes == 0:
        return ProsodyAlignment(list(syllables), onsets, [], 0, 0)
    cost = np.full((n_syl + 1, n_notes + 1), np.inf)
    cost[0, 0] = 0.0
    back = np.zeros((n_syl + 1, n_notes + 1), dtype=int)
    for i in range(1, n_syl + 1):
        for j in range(1, n_notes + 1):
            mw = _metrical_weight(onsets[j - 1], beats_per_bar)
            stress = syllables[i - 1].stress_level / 2.0
            match_cost = abs(stress - mw)
            skip_cost = 0.5
            assign = cost[i - 1, j - 1] + match_cost
            skip_note = cost[i, j - 1] + skip_cost
            if assign <= skip_note:
                cost[i, j] = assign
                back[i, j] = 1
            else:
                cost[i, j] = skip_note
                back[i, j] = 2
    alignment: list[int] = []
    i, j = n_syl, n_notes
    while i > 0 and j > 0:
        if back[i, j] == 1:
            alignment.append(j - 1)
            i -= 1
            j -= 1
        else:
            j -= 1
    alignment.reverse()
    while len(alignment) < n_syl:
        alignment.append(min(len(alignment), n_notes - 1))
    violations = 0
    for idx, syl in enumerate(syllables):
        if idx < len(alignment):
            mw = _metrical_weight(onsets[alignment[idx]], beats_per_bar)
            if syl.stress_level == 2 and mw < 0.5:
                violations += 1
    melisma = n_notes - n_syl if n_notes > n_syl else 0
    return ProsodyAlignment(
        syllables=list(syllables),
        note_onsets=onsets,
        alignment_map=alignment,
        stress_violations=violations,
        melisma_count=max(0, melisma),
    )


def prosody_score(alignment: ProsodyAlignment, beats_per_bar: float = 4.0) -> float:
    if not alignment.syllables or len(alignment.note_onsets) == 0:
        return 0.0
    n = len(alignment.syllables)
    stress_match = 0.0
    for i, syl in enumerate(alignment.syllables):
        if i < len(alignment.alignment_map):
            note_idx = alignment.alignment_map[i]
            mw = _metrical_weight(alignment.note_onsets[note_idx], beats_per_bar)
            stress_match += 1.0 - abs(syl.stress_level / 2.0 - mw)
    stress_score = stress_match / max(n, 1)
    violation_penalty = alignment.stress_violations / max(n, 1)
    return float(np.clip(0.7 * stress_score + 0.3 * (1.0 - violation_penalty), 0.0, 1.0))


def contour_similarity(speech_f0: np.ndarray, melody_pitches: np.ndarray) -> float:
    f0 = np.asarray(speech_f0, dtype=np.float64)
    mp = np.asarray(melody_pitches, dtype=np.float64)
    if len(f0) < 2 or len(mp) < 2:
        return 0.0
    n = min(len(f0), len(mp))
    f0_norm = (f0[:n] - np.mean(f0[:n])) / max(np.std(f0[:n]), 1e-9)
    mp_norm = (mp[:n] - np.mean(mp[:n])) / max(np.std(mp[:n]), 1e-9)
    corr = float(np.corrcoef(f0_norm, mp_norm)[0, 1])
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def melisma_ratio(n_syllables: int, n_notes: int) -> float:
    if n_notes == 0:
        return 0.0
    return float(max(0, n_notes - n_syllables) / n_notes)


def text_setting_quality(
    alignment: ProsodyAlignment,
    speech_f0: np.ndarray | None = None,
    melody_pitches: np.ndarray | None = None,
) -> float:
    ps = prosody_score(alignment)
    mr = melisma_ratio(len(alignment.syllables), len(alignment.note_onsets))
    melisma_penalty = max(0.0, mr - 0.3) * 2.0
    cs = 0.5
    if speech_f0 is not None and melody_pitches is not None:
        cs = contour_similarity(speech_f0, melody_pitches)
    return float(np.clip(0.5 * ps + 0.3 * cs + 0.2 * (1.0 - melisma_penalty), 0.0, 1.0))


def optimal_text_underlay(
    syllables: Sequence[Syllable],
    note_durations: np.ndarray,
    beats_per_bar: float = 4.0,
) -> list[int]:
    durs = np.asarray(note_durations, dtype=np.float64)
    n_syl = len(syllables)
    n_notes = len(durs)
    if n_syl == 0 or n_notes == 0:
        return []
    dp = np.full((n_syl + 1, n_notes + 1), np.inf)
    dp[0, 0] = 0.0
    parent = np.full((n_syl + 1, n_notes + 1), -1, dtype=int)
    for i in range(1, n_syl + 1):
        for j in range(i, n_notes + 1):
            dur_match = abs(syllables[i - 1].duration_weight - durs[j - 1])
            assign = dp[i - 1, j - 1] + dur_match
            if j > i and dp[i, j - 1] + 0.3 < assign:
                dp[i, j] = dp[i, j - 1] + 0.3
                parent[i, j] = 0
            else:
                dp[i, j] = assign
                parent[i, j] = 1
    j = n_notes
    best_j = n_notes
    for jj in range(n_syl, n_notes + 1):
        if dp[n_syl, jj] < dp[n_syl, best_j]:
            best_j = jj
    j = best_j
    result: list[int] = []
    i = n_syl
    while i > 0 and j > 0:
        if parent[i, j] == 1:
            result.append(j - 1)
            i -= 1
            j -= 1
        else:
            j -= 1
    result.reverse()
    return result
