"""Originality scoring: melodic, harmonic, rhythmic novelty against corpora."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def n_gram_novelty(sequence: list, corpus: list[list], n: int = 3) -> float:
    if len(sequence) < n:
        return 1.0
    query_grams = set()
    for i in range(len(sequence) - n + 1):
        query_grams.add(tuple(sequence[i:i + n]))
    corpus_grams: set[tuple] = set()
    for item in corpus:
        for i in range(len(item) - n + 1):
            corpus_grams.add(tuple(item[i:i + n]))
    if not query_grams:
        return 1.0
    novel = query_grams - corpus_grams
    return len(novel) / len(query_grams)


class OriginalityScorer:
    def melodic_originality(self, melody: list[int], corpus: list[list[int]]) -> float:
        intervals = [melody[i + 1] - melody[i] for i in range(len(melody) - 1)]
        corpus_intervals = [
            [m[i + 1] - m[i] for i in range(len(m) - 1)]
            for m in corpus if len(m) >= 2
        ]
        if not corpus_intervals or len(intervals) < 3:
            return 1.0
        return n_gram_novelty(intervals, corpus_intervals, n=3)

    def harmonic_originality(self, progression: list[str], corpus: list[list[str]]) -> float:
        if len(progression) < 2:
            return 1.0
        return n_gram_novelty(progression, corpus, n=2)

    def rhythmic_originality(self, pattern: list[float], corpus: list[list[float]]) -> float:
        quantized = [round(d * 4) for d in pattern]
        corpus_q = [[round(d * 4) for d in p] for p in corpus]
        if len(quantized) < 3:
            return 1.0
        return n_gram_novelty(quantized, corpus_q, n=3)

    def composite_originality(
        self, melody: list[int], progression: list[str], rhythm: list[float],
        melody_corpus: list[list[int]], harmony_corpus: list[list[str]],
        rhythm_corpus: list[list[float]],
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> float:
        m = self.melodic_originality(melody, melody_corpus)
        h = self.harmonic_originality(progression, harmony_corpus)
        r = self.rhythmic_originality(rhythm, rhythm_corpus)
        total_w = sum(weights)
        return (weights[0] * m + weights[1] * h + weights[2] * r) / total_w
