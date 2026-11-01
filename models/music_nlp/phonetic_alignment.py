"""Phonetic alignment for lyrics-to-melody fitting.

Implements syllable-to-note alignment, vowel-consonant classification for
singability analysis, and melismatic detection — pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


VOWELS = set("aeiouAEIOU")
CONSONANT_CLUSTERS = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr",
                      "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st",
                      "str", "sw", "tr", "tw", "wr", "th", "sh", "ch", "wh"}


@dataclass
class Syllable:
    text: str
    onset: str = ""
    nucleus: str = ""
    coda: str = ""
    stress: int = 0
    duration: float = 0.0

    @property
    def is_open(self) -> bool:
        return len(self.coda) == 0

    @property
    def singability(self) -> float:
        v_count = sum(1 for c in self.nucleus if c in VOWELS)
        c_count = len(self.onset) + len(self.coda)
        if v_count == 0:
            return 0.1
        return float(np.clip(v_count / (v_count + c_count * 0.5), 0, 1))


@dataclass
class PhoneticWord:
    text: str
    syllables: List[Syllable] = field(default_factory=list)

    @property
    def n_syllables(self) -> int:
        return len(self.syllables)


def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)


def syllabify(word: str) -> List[Syllable]:
    word = word.lower().strip()
    if not word:
        return []
    n = count_syllables(word)
    if n == 1:
        onset, nucleus, coda = "", "", ""
        state = "onset"
        for ch in word:
            if ch in VOWELS:
                state = "nucleus"
                nucleus += ch
            elif state == "onset":
                onset += ch
            else:
                state = "coda"
                coda += ch
        return [Syllable(text=word, onset=onset, nucleus=nucleus, coda=coda)]
    chunk_size = max(1, len(word) // n)
    syllables = []
    pos = 0
    for i in range(n):
        end = pos + chunk_size if i < n - 1 else len(word)
        text = word[pos:end]
        onset = "".join(c for c in text if c not in VOWELS and text.index(c) < len(text) // 2) if text else ""
        nucleus = "".join(c for c in text if c in VOWELS)
        coda = ""
        syllables.append(Syllable(text=text, onset=onset[:2], nucleus=nucleus, coda=coda))
        pos = end
    return syllables


@dataclass
class NoteSlot:
    pitch: int
    start: float
    duration: float
    is_rest: bool = False


@dataclass
class AlignmentResult:
    pairs: List[Tuple[Syllable, NoteSlot]] = field(default_factory=list)
    melismas: List[Tuple[Syllable, List[NoteSlot]]] = field(default_factory=list)
    unmatched_syllables: int = 0
    unmatched_notes: int = 0

    @property
    def alignment_score(self) -> float:
        total = len(self.pairs) + len(self.melismas) + self.unmatched_syllables + self.unmatched_notes
        if total == 0:
            return 1.0
        matched = len(self.pairs) + len(self.melismas)
        return matched / total


def align_syllables_to_notes(syllables: List[Syllable], notes: List[NoteSlot]) -> AlignmentResult:
    result = AlignmentResult()
    non_rest = [n for n in notes if not n.is_rest]
    si, ni = 0, 0
    while si < len(syllables) and ni < len(non_rest):
        syl = syllables[si]
        note = non_rest[ni]
        if ni + 1 < len(non_rest) and non_rest[ni + 1].start - note.start < 0.1:
            mel_notes = [note]
            j = ni + 1
            while j < len(non_rest) and non_rest[j].start - non_rest[j - 1].start < 0.15:
                mel_notes.append(non_rest[j])
                j += 1
            if len(mel_notes) > 1:
                result.melismas.append((syl, mel_notes))
                si += 1
                ni = j
                continue
        result.pairs.append((syl, note))
        si += 1
        ni += 1
    result.unmatched_syllables = len(syllables) - si
    result.unmatched_notes = len(non_rest) - ni
    return result


def stress_alignment_score(syllables: List[Syllable], notes: List[NoteSlot]) -> float:
    alignment = align_syllables_to_notes(syllables, notes)
    if not alignment.pairs:
        return 0.0
    scores = []
    for syl, note in alignment.pairs:
        if syl.stress > 0 and note.duration > np.mean([n.duration for _, n in alignment.pairs]):
            scores.append(1.0)
        elif syl.stress == 0:
            scores.append(0.7)
        else:
            scores.append(0.3)
    return float(np.mean(scores))


def vowel_on_high_note(syllables: List[Syllable], notes: List[NoteSlot]) -> float:
    alignment = align_syllables_to_notes(syllables, notes)
    if not alignment.pairs:
        return 0.0
    pitches = [n.pitch for _, n in alignment.pairs]
    if not pitches:
        return 0.0
    median_pitch = np.median(pitches)
    good = 0
    high_count = 0
    for syl, note in alignment.pairs:
        if note.pitch > median_pitch:
            high_count += 1
            if syl.singability > 0.5:
                good += 1
    return good / max(high_count, 1)


def lyric_density(syllables: List[Syllable], total_duration: float) -> float:
    if total_duration <= 0:
        return 0.0
    return len(syllables) / total_duration


def phrasing_breath_points(notes: List[NoteSlot], min_gap: float = 0.3) -> List[float]:
    points = []
    for i in range(1, len(notes)):
        gap = notes[i].start - (notes[i - 1].start + notes[i - 1].duration)
        if gap >= min_gap:
            points.append(notes[i].start)
    return points
