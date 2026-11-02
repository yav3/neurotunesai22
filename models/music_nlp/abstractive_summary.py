"""Song summarization: extractive and abstractive lyric summaries.

Sentence scoring, section importance, thematic condensation. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import re

import numpy as np


@dataclass
class ScoredLine:
    text: str
    score: float
    line_index: int
    section: str = ""


@dataclass
class SongSummary:
    key_lines: list[ScoredLine]
    themes: list[str]
    narrative_summary: str
    compression_ratio: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _sentence_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union)


def textrank_sentences(lines: Sequence[str], top_k: int = 5) -> list[ScoredLine]:
    """TextRank sentence scoring for extractive summarization."""
    n = len(lines)
    if n == 0:
        return []

    tokenized = [_tokenize(l) for l in lines]

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = _sentence_similarity(tokenized[i], tokenized[j])
            sim[i, j] = s
            sim[j, i] = s

    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = sim / row_sums

    d = 0.85
    scores = np.ones(n) / n
    for _ in range(30):
        scores = (1 - d) / n + d * transition.T @ scores

    scored = [ScoredLine(text=lines[i], score=float(scores[i]), line_index=i) for i in range(n)]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]


def position_weighted_score(lines: Sequence[str]) -> list[ScoredLine]:
    """Score lines by position heuristic: first/last lines and chorus-like repeats score higher."""
    n = len(lines)
    if n == 0:
        return []

    line_counts: dict[str, int] = {}
    for l in lines:
        key = l.lower().strip()
        line_counts[key] = line_counts.get(key, 0) + 1

    scored: list[ScoredLine] = []
    for i, line in enumerate(lines):
        key = line.lower().strip()
        position_score = 1.0
        if i == 0 or i == n - 1:
            position_score = 1.5
        elif i < n * 0.1 or i > n * 0.9:
            position_score = 1.2

        repeat_score = min(line_counts[key] / 2, 2.0)

        length_score = min(len(line.split()) / 8, 1.5)

        total = position_score * 0.3 + repeat_score * 0.4 + length_score * 0.3
        scored.append(ScoredLine(text=line, score=total, line_index=i))

    return scored


def extract_themes(lines: Sequence[str], n_themes: int = 5) -> list[str]:
    """Extract dominant themes as most frequent non-stopword terms."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "but", "or", "not", "so", "i", "me", "my", "you",
        "your", "he", "she", "it", "we", "they", "them", "this",
        "that", "what", "when", "where", "how", "who", "all",
        "just", "like", "know", "got", "get", "go", "come",
        "can", "could", "would", "should", "up", "out", "no",
        "yeah", "oh", "ooh", "ah", "la", "na", "da",
    }
    freq: dict[str, int] = {}
    for line in lines:
        for word in _tokenize(line):
            if word not in stopwords and len(word) > 2:
                freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:n_themes]]


def summarize_song(
    lines: Sequence[str], n_lines: int = 4, n_themes: int = 5,
) -> SongSummary:
    """Produce a song summary: key lines, themes, compression ratio."""
    key_lines = textrank_sentences(lines, top_k=n_lines)
    themes = extract_themes(lines, n_themes=n_themes)

    key_lines_sorted = sorted(key_lines, key=lambda s: s.line_index)

    narrative = " / ".join(s.text.strip() for s in key_lines_sorted)

    compression = n_lines / max(len(lines), 1)

    return SongSummary(
        key_lines=key_lines_sorted,
        themes=themes,
        narrative_summary=narrative,
        compression_ratio=compression,
    )


def chorus_detection(lines: Sequence[str], threshold: int = 2) -> list[str]:
    """Detect likely chorus lines by repetition count."""
    counts: dict[str, int] = {}
    for l in lines:
        key = l.lower().strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return [text for text, count in counts.items() if count >= threshold]


def information_density(line: str) -> float:
    """Estimate information density: unique-word-ratio * avg-word-length / 10."""
    tokens = _tokenize(line)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    avg_len = np.mean([len(t) for t in tokens])
    return float(unique_ratio * avg_len / 10)
