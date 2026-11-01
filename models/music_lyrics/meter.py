"""Poetic meter detection: stress patterns, meter classification, scansion regularity."""
from __future__ import annotations

import re
from models.music_lyrics.tokenizer import word_tokenize, syllable_estimate

_UNSTRESSED_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "it", "its", "if", "so", "no", "not", "he", "she", "we", "me",
    "my", "his", "her", "our", "your", "up", "do", "did", "has", "had",
    "than", "that", "this", "then", "when", "who", "how", "all",
}


def stress_pattern(word: str) -> list[int]:
    w = word.lower().strip()
    if not w or not any(c.isalpha() for c in w):
        return []
    n_syl = syllable_estimate(w)
    if n_syl == 0:
        return []
    if n_syl == 1:
        return [0] if w in _UNSTRESSED_WORDS else [1]
    pattern = [0] * n_syl
    if n_syl == 2:
        if w.endswith(("ing", "tion", "ment", "ness", "ful", "less", "ous", "ive", "al")):
            pattern[0] = 1
        elif w.startswith(("un", "re", "de", "pre", "dis", "mis", "out", "over")):
            pattern[1] = 1
        else:
            pattern[0] = 1
    elif n_syl == 3:
        pattern[1] = 1
    else:
        for i in range(0, n_syl, 2):
            pattern[min(i + 1, n_syl - 1)] = 1
        if pattern.count(1) == 0:
            pattern[1 if n_syl > 1 else 0] = 1
    return pattern


def _line_stress(line: str) -> list[int]:
    words = word_tokenize(line)
    pattern: list[int] = []
    for w in words:
        if any(c.isalpha() for c in w):
            pattern.extend(stress_pattern(w))
    return pattern


_METER_PATTERNS: dict[str, list[int]] = {
    "iambic": [0, 1],
    "trochaic": [1, 0],
    "anapestic": [0, 0, 1],
    "dactylic": [1, 0, 0],
    "spondaic": [1, 1],
    "pyrrhic": [0, 0],
    "amphibrachic": [0, 1, 0],
}


def _pattern_match_score(stress: list[int], template: list[int]) -> float:
    if not stress or not template:
        return 0.0
    matches = 0
    total = len(stress)
    for i, s in enumerate(stress):
        expected = template[i % len(template)]
        if s == expected:
            matches += 1
    return matches / total if total > 0 else 0.0


def detect_meter(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return "free verse"

    avg_scores: dict[str, float] = {name: 0.0 for name in _METER_PATTERNS}
    n_lines = 0
    for line in lines:
        stress = _line_stress(line)
        if len(stress) < 3:
            continue
        n_lines += 1
        for name, tmpl in _METER_PATTERNS.items():
            avg_scores[name] += _pattern_match_score(stress, tmpl)

    if n_lines == 0:
        return "free verse"

    for name in avg_scores:
        avg_scores[name] /= n_lines

    best = max(avg_scores, key=lambda k: avg_scores[k])
    if avg_scores[best] < 0.55:
        return "free verse"
    return best


def scansion_regularity(text: str) -> float:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0

    syllable_counts = []
    for line in lines:
        words = word_tokenize(line)
        count = sum(syllable_estimate(w) for w in words if any(c.isalpha() for c in w))
        if count > 0:
            syllable_counts.append(count)

    if len(syllable_counts) < 2:
        return 0.0

    mean = sum(syllable_counts) / len(syllable_counts)
    if mean == 0:
        return 0.0
    variance = sum((c - mean) ** 2 for c in syllable_counts) / len(syllable_counts)
    cv = (variance ** 0.5) / mean
    return max(0.0, 1.0 - cv)
