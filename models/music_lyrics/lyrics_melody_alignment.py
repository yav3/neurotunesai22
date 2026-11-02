"""Lyrics-melody alignment: syllable-to-note mapping and prosodic scoring.

Maps lyric syllables to melodic notes via dynamic-programming alignment,
scores prosodic fit (lexical stress vs metric accent), and handles
melisma (one syllable → multiple notes) and elision (multiple syllables → one note).

References:
  - Harte (2010) on text-setting constraints
  - Temperley (2004) "A Bayesian approach to key-finding"
  - Palmer & Kelly (1992) on linguistic stress and musical meter
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class AlignmentType(Enum):
    SYLLABIC = "syllabic"
    MELISMATIC = "melismatic"
    ELIDED = "elided"


@dataclass
class SyllableInfo:
    text: str
    stress: int  # 0 = unstressed, 1 = primary, 2 = secondary
    word_position: int  # 0-indexed position within word
    word_length: int  # total syllables in parent word
    is_word_start: bool
    is_word_end: bool


@dataclass
class NoteInfo:
    midi_pitch: int  # MIDI note number
    duration: float  # in beats
    onset_beat: float
    is_downbeat: bool
    metric_weight: float  # 0..1, higher = stronger beat


@dataclass
class AlignmentPair:
    syllable: SyllableInfo
    notes: List[NoteInfo]
    alignment_type: AlignmentType
    prosodic_score: float  # how well stress matches metric weight


@dataclass
class AlignmentResult:
    pairs: List[AlignmentPair]
    total_prosodic_score: float
    melisma_ratio: float  # fraction of syllables with >1 note
    stress_correlation: float  # Pearson r between stress and metric weight


# ── Syllable extraction ──

_UNSTRESSED_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "it", "its", "if", "so", "no", "not", "he", "she", "we", "me",
    "my", "his", "her", "our", "your", "up", "do", "did", "has", "had",
})

_VOWELS = set("aeiouyAEIOUY")


def _count_syllables(word: str) -> int:
    w = word.lower().strip()
    if not w:
        return 0
    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ie", "ee", "ye")):
        count -= 1
    return max(1, count)


def _assign_stress(word: str) -> List[int]:
    w = word.lower().strip()
    n = _count_syllables(w)
    if n == 0:
        return []
    if n == 1:
        return [0] if w in _UNSTRESSED_WORDS else [1]
    pattern = [0] * n
    if n == 2:
        suffixes_first = ("ing", "tion", "ment", "ness", "ful", "less", "ous", "ive")
        if w.endswith(suffixes_first):
            pattern[0] = 1
        elif w.startswith(("un", "re", "de", "pre", "dis")):
            pattern[1] = 1
        else:
            pattern[0] = 1
    elif n == 3:
        pattern[0] = 2
        pattern[1] = 1
    else:
        for i in range(n):
            if i % 2 == 0:
                pattern[i] = 2 if i > 0 else 1
            else:
                pattern[i] = 1 if i == 1 else 0
        pattern[1] = 1
        pattern[0] = 2
    return pattern


def extract_syllables(text: str) -> List[SyllableInfo]:
    import re
    words = re.findall(r"[a-zA-Z']+", text)
    syllables: List[SyllableInfo] = []
    for word in words:
        n = _count_syllables(word)
        if n == 0:
            continue
        stress_pat = _assign_stress(word)
        w_lower = word.lower()
        syl_len = max(1, len(w_lower) // n)
        for i in range(n):
            start = i * syl_len
            end = (i + 1) * syl_len if i < n - 1 else len(w_lower)
            syl_text = w_lower[start:end]
            syllables.append(SyllableInfo(
                text=syl_text,
                stress=stress_pat[i] if i < len(stress_pat) else 0,
                word_position=i,
                word_length=n,
                is_word_start=(i == 0),
                is_word_end=(i == n - 1),
            ))
    return syllables


def extract_notes(pitches: np.ndarray, durations: np.ndarray,
                  time_signature: Tuple[int, int] = (4, 4)) -> List[NoteInfo]:
    beats_per_measure, beat_unit = time_signature
    notes: List[NoteInfo] = []
    onset = 0.0
    for i in range(len(pitches)):
        beat_in_measure = onset % beats_per_measure
        is_downbeat = beat_in_measure < 0.01
        if beats_per_measure == 4:
            if is_downbeat:
                mw = 1.0
            elif abs(beat_in_measure - 2.0) < 0.01:
                mw = 0.75
            elif abs(beat_in_measure % 1.0) < 0.01:
                mw = 0.5
            else:
                mw = 0.25
        elif beats_per_measure == 3:
            if is_downbeat:
                mw = 1.0
            elif abs(beat_in_measure % 1.0) < 0.01:
                mw = 0.5
            else:
                mw = 0.25
        else:
            mw = 1.0 if is_downbeat else 0.5

        dur = float(durations[i]) if i < len(durations) else 1.0
        notes.append(NoteInfo(
            midi_pitch=int(pitches[i]),
            duration=dur,
            onset_beat=onset,
            is_downbeat=is_downbeat,
            metric_weight=mw,
        ))
        onset += dur
    return notes


# ── Dynamic-programming alignment ──

def _substitution_cost(syl: SyllableInfo, note: NoteInfo) -> float:
    stress_match = 1.0 - abs(syl.stress / max(1, 2) - note.metric_weight)
    word_start_bonus = 0.0
    if syl.is_word_start and note.is_downbeat:
        word_start_bonus = -0.3
    elif syl.is_word_start and note.metric_weight >= 0.75:
        word_start_bonus = -0.15
    return (1.0 - stress_match) + word_start_bonus


def _melisma_cost(n_notes: int) -> float:
    if n_notes <= 1:
        return 0.0
    return 0.2 * (n_notes - 1)


def _elision_cost(n_syllables: int) -> float:
    if n_syllables <= 1:
        return 0.0
    return 0.5 * (n_syllables - 1)


def align_syllables_to_notes(syllables: List[SyllableInfo],
                              notes: List[NoteInfo],
                              melisma_limit: int = 4) -> AlignmentResult:
    n_syl = len(syllables)
    n_note = len(notes)
    if n_syl == 0 or n_note == 0:
        return AlignmentResult(pairs=[], total_prosodic_score=0.0,
                               melisma_ratio=0.0, stress_correlation=0.0)

    INF = float('inf')
    dp = np.full((n_syl + 1, n_note + 1), INF)
    back = np.zeros((n_syl + 1, n_note + 1, 2), dtype=int)
    dp[0, 0] = 0.0

    for i in range(1, n_syl + 1):
        for j in range(1, n_note + 1):
            best = INF
            best_back = (0, 0)
            sub_cost = _substitution_cost(syllables[i - 1], notes[j - 1])
            if dp[i - 1, j - 1] + sub_cost < best:
                best = dp[i - 1, j - 1] + sub_cost
                best_back = (i - 1, j - 1)
            for k in range(2, min(melisma_limit + 1, j + 1)):
                if i - 1 >= 0 and j - k >= 0:
                    mc = _melisma_cost(k) + sub_cost
                    if dp[i - 1, j - k] + mc < best:
                        best = dp[i - 1, j - k] + mc
                        best_back = (i - 1, j - k)
            if i >= 2 and j >= 1:
                ec = _elision_cost(2) + _substitution_cost(syllables[i - 1], notes[j - 1])
                if dp[i - 2, j - 1] + ec < best:
                    best = dp[i - 2, j - 1] + ec
                    best_back = (i - 2, j - 1)

            dp[i, j] = best
            back[i, j] = best_back

    i, j = n_syl, n_note
    if dp[i, j] == INF:
        min_j = np.argmin(dp[i, :])
        j = int(min_j)

    raw_pairs: List[Tuple[int, int, int]] = []
    while i > 0 and j > 0:
        pi, pj = int(back[i, j, 0]), int(back[i, j, 1])
        n_syls_used = i - pi
        n_notes_used = j - pj
        for s in range(pi, i):
            note_start = pj
            note_end = j
            raw_pairs.append((s, note_start, note_end))
        i, j = pi, pj

    raw_pairs.reverse()
    seen_syls = set()
    pairs: List[AlignmentPair] = []
    for (s_idx, n_start, n_end) in raw_pairs:
        if s_idx in seen_syls or s_idx >= n_syl:
            continue
        seen_syls.add(s_idx)
        assigned_notes = notes[n_start:n_end]
        if not assigned_notes:
            assigned_notes = [notes[min(n_start, n_note - 1)]]
        if len(assigned_notes) > 1:
            atype = AlignmentType.MELISMATIC
        else:
            atype = AlignmentType.SYLLABIC
        prosodic = 1.0 - _substitution_cost(syllables[s_idx], assigned_notes[0])
        pairs.append(AlignmentPair(
            syllable=syllables[s_idx],
            notes=assigned_notes,
            alignment_type=atype,
            prosodic_score=max(0.0, min(1.0, prosodic)),
        ))

    total_score = np.mean([p.prosodic_score for p in pairs]) if pairs else 0.0
    melisma_count = sum(1 for p in pairs if p.alignment_type == AlignmentType.MELISMATIC)
    melisma_ratio = melisma_count / max(1, len(pairs))

    stresses = np.array([p.syllable.stress for p in pairs], dtype=float)
    weights = np.array([p.notes[0].metric_weight for p in pairs], dtype=float)
    if len(stresses) > 1 and np.std(stresses) > 0 and np.std(weights) > 0:
        stress_corr = float(np.corrcoef(stresses, weights)[0, 1])
    else:
        stress_corr = 0.0

    return AlignmentResult(
        pairs=pairs,
        total_prosodic_score=float(total_score),
        melisma_ratio=float(melisma_ratio),
        stress_correlation=stress_corr,
    )


# ── High-level alignment from text + MIDI-like arrays ──

def align_lyrics_to_melody(text: str,
                           pitches: np.ndarray,
                           durations: np.ndarray,
                           time_signature: Tuple[int, int] = (4, 4),
                           melisma_limit: int = 4) -> AlignmentResult:
    syllables = extract_syllables(text)
    notes = extract_notes(pitches, durations, time_signature)
    return align_syllables_to_notes(syllables, notes, melisma_limit)


# ── Prosodic fitness metrics ──

def prosodic_fitness(text: str, pitches: np.ndarray,
                     durations: np.ndarray,
                     time_signature: Tuple[int, int] = (4, 4)) -> dict:
    result = align_lyrics_to_melody(text, pitches, durations, time_signature)
    stressed_on_strong = 0
    unstressed_on_weak = 0
    total = len(result.pairs)
    for p in result.pairs:
        if p.syllable.stress >= 1 and p.notes[0].metric_weight >= 0.75:
            stressed_on_strong += 1
        elif p.syllable.stress == 0 and p.notes[0].metric_weight < 0.5:
            unstressed_on_weak += 1
    return {
        "prosodic_score": result.total_prosodic_score,
        "stress_correlation": result.stress_correlation,
        "melisma_ratio": result.melisma_ratio,
        "stressed_on_strong_beats": stressed_on_strong / max(1, total),
        "unstressed_on_weak_beats": unstressed_on_weak / max(1, total),
        "n_syllables": len(result.pairs),
    }


def phrase_boundary_score(text: str, pitches: np.ndarray,
                          durations: np.ndarray,
                          time_signature: Tuple[int, int] = (4, 4)) -> float:
    import re
    clauses = re.split(r'[,;:.!?\n]+', text)
    clause_syl_counts: List[int] = []
    for clause in clauses:
        clause = clause.strip()
        if clause:
            clause_syl_counts.append(len(extract_syllables(clause)))

    notes = extract_notes(pitches, durations, time_signature)
    beats_per_measure = time_signature[0]
    measure_boundaries: List[int] = []
    for i, note in enumerate(notes):
        if note.is_downbeat and i > 0:
            measure_boundaries.append(i)

    if not clause_syl_counts or not measure_boundaries:
        return 0.0

    cum_syls = np.cumsum(clause_syl_counts)
    clause_ends = set(int(c) for c in cum_syls[:-1])

    note_idx = 0
    matches = 0
    for end_syl in clause_ends:
        closest = min(measure_boundaries, key=lambda b: abs(b - end_syl),
                      default=None)
        if closest is not None and abs(closest - end_syl) <= 2:
            matches += 1

    return matches / max(1, len(clause_ends))


def pitch_contour_text_affect(text: str, pitches: np.ndarray,
                              durations: np.ndarray) -> dict:
    from models.music_lyrics.sentiment import line_vad
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines or len(pitches) == 0:
        return {"contour_valence_correlation": 0.0, "n_lines": 0}

    notes_per_line = max(1, len(pitches) // len(lines))
    line_valences: List[float] = []
    line_pitches: List[float] = []
    for i, line in enumerate(lines):
        v, a, d = line_vad(line)
        line_valences.append(v)
        start = i * notes_per_line
        end = min((i + 1) * notes_per_line, len(pitches))
        segment = pitches[start:end]
        if len(segment) > 0:
            line_pitches.append(float(np.mean(segment)))
        else:
            line_pitches.append(60.0)

    v_arr = np.array(line_valences)
    p_arr = np.array(line_pitches)
    if len(v_arr) > 1 and np.std(v_arr) > 0 and np.std(p_arr) > 0:
        corr = float(np.corrcoef(v_arr, p_arr)[0, 1])
    else:
        corr = 0.0

    return {
        "contour_valence_correlation": corr,
        "n_lines": len(lines),
        "mean_valence": float(np.mean(v_arr)),
        "mean_pitch": float(np.mean(p_arr)),
        "pitch_range": float(np.ptp(p_arr)),
    }
