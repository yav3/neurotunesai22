"""Lyric prosody analysis: stress-beat alignment, melismatic classification, prosodic fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SyllableStress", "ProsodicFit",
    "extract_stress_pattern", "stress_beat_alignment",
    "melismatic_classification", "vowel_openness_map",
    "text_rhythm_ioi", "prosodic_fit_score",
]


VOWEL_OPENNESS: Dict[str, float] = {
    "AA": 1.0, "AE": 0.9, "AH": 0.7, "AO": 0.9, "AW": 0.8,
    "AY": 0.8, "EH": 0.6, "ER": 0.5, "EY": 0.5, "IH": 0.3,
    "IY": 0.2, "OW": 0.6, "OY": 0.5, "UH": 0.4, "UW": 0.2,
}

STRESS_RULES: Dict[str, int] = {
    "the": 0, "a": 0, "an": 0, "and": 0, "or": 0, "but": 0,
    "in": 0, "on": 0, "at": 0, "to": 0, "for": 0, "of": 0,
    "is": 0, "am": 0, "are": 0, "was": 0, "were": 0,
    "i": 1, "you": 1, "he": 1, "she": 1, "we": 1, "they": 1,
}


@dataclass
class SyllableStress:
    text: str
    stress_level: int
    vowel_openness: float


@dataclass
class ProsodicFit:
    stress_score: float
    vowel_score: float
    rhythm_score: float
    composite: float


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    vowels = set("aeiouy")
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


def _primary_vowel(word: str) -> str:
    word = word.lower()
    vowels = "aeiouy"
    for ch in word:
        if ch in vowels:
            return ch.upper() + "H"
    return "AH"


def extract_stress_pattern(text: str) -> List[SyllableStress]:
    words = text.split()
    result = []
    for word in words:
        clean = word.strip(".,!?;:'\"()-").lower()
        if not clean:
            continue
        n_syl = _count_syllables(clean)
        base_stress = STRESS_RULES.get(clean, 1)
        vowel_ph = _primary_vowel(clean)
        openness = VOWEL_OPENNESS.get(vowel_ph, 0.5)
        if n_syl == 1:
            result.append(SyllableStress(clean, base_stress, openness))
        else:
            for i in range(n_syl):
                s = 1 if i == 0 else 0
                result.append(SyllableStress(f"{clean}[{i}]", s, openness))
    return result


def stress_beat_alignment(
    stress_pattern: List[int], beat_positions: np.ndarray
) -> float:
    if len(stress_pattern) == 0 or len(beat_positions) == 0:
        return 0.0
    n = min(len(stress_pattern), len(beat_positions))
    aligned = 0
    total_stressed = 0
    for i in range(n):
        if stress_pattern[i] > 0:
            total_stressed += 1
            if beat_positions[i] > 0.5:
                aligned += 1
    if total_stressed == 0:
        return 1.0
    return aligned / total_stressed


def melismatic_classification(
    notes_per_syllable: np.ndarray, threshold: float = 1.5
) -> np.ndarray:
    return (notes_per_syllable > threshold).astype(np.float64)


def vowel_openness_map(phonemes: List[str]) -> np.ndarray:
    return np.array([VOWEL_OPENNESS.get(p.upper(), 0.5) for p in phonemes])


def text_rhythm_ioi(syllable_durations: np.ndarray) -> np.ndarray:
    if len(syllable_durations) < 2:
        return np.array([])
    onsets = np.cumsum(syllable_durations)
    return np.diff(onsets)


def prosodic_fit_score(
    stress_pattern: List[int],
    beat_positions: np.ndarray,
    vowel_openness: np.ndarray,
    pitch_contour: np.ndarray,
    syllable_durations: Optional[np.ndarray] = None,
    beat_ioi: Optional[np.ndarray] = None,
) -> ProsodicFit:
    s_score = stress_beat_alignment(stress_pattern, beat_positions)

    n = min(len(vowel_openness), len(pitch_contour))
    if n >= 2:
        vo = vowel_openness[:n]
        pc = pitch_contour[:n]
        vo_norm = (vo - vo.mean()) / max(vo.std(), 1e-9)
        pc_norm = (pc - pc.mean()) / max(pc.std(), 1e-9)
        v_score = float(np.clip(np.corrcoef(vo_norm, pc_norm)[0, 1], -1, 1))
        v_score = (v_score + 1.0) / 2.0
    else:
        v_score = 0.5

    r_score = 0.5
    if syllable_durations is not None and beat_ioi is not None:
        text_ioi = text_rhythm_ioi(syllable_durations)
        m = min(len(text_ioi), len(beat_ioi))
        if m >= 2:
            t = text_ioi[:m]
            b = beat_ioi[:m]
            t_norm = t / max(t.mean(), 1e-9)
            b_norm = b / max(b.mean(), 1e-9)
            r_score = float(1.0 - np.mean(np.abs(t_norm - b_norm)))
            r_score = max(0.0, min(1.0, r_score))

    composite = 0.4 * s_score + 0.3 * v_score + 0.3 * r_score
    return ProsodicFit(s_score, v_score, r_score, composite)
