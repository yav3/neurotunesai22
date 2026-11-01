"""Plagiarism detection: sliding-window melody comparison with transposition."""
from __future__ import annotations
from dataclasses import dataclass
from models.music_copyright.melody_comparator import transposition_invariant_similarity


@dataclass
class PlagiarismReport:
    similarity_score: float
    matching_segments: list[tuple[int, int, float]]
    transposition_offset: int
    confidence: float


class PlagiarismDetector:
    def __init__(self, window_size: int = 8, threshold: float = 0.85):
        self.window_size = window_size
        self.threshold = threshold

    def compare_melodies(self, melody_a: list[int], melody_b: list[int]) -> PlagiarismReport:
        if len(melody_a) < self.window_size or len(melody_b) < self.window_size:
            sim = transposition_invariant_similarity(melody_a, melody_b)
            offset = self._best_transposition(melody_a, melody_b)
            return PlagiarismReport(sim, [], offset, sim)

        segments: list[tuple[int, int, float]] = []
        best_scores: list[float] = []

        for i in range(len(melody_a) - self.window_size + 1):
            window_a = melody_a[i:i + self.window_size]
            best_sim = 0.0
            best_j = 0
            for j in range(len(melody_b) - self.window_size + 1):
                window_b = melody_b[j:j + self.window_size]
                sim = transposition_invariant_similarity(window_a, window_b)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            best_scores.append(best_sim)
            if best_sim >= self.threshold:
                segments.append((i, best_j, best_sim))

        overall = sum(best_scores) / len(best_scores) if best_scores else 0.0
        offset = self._best_transposition(melody_a, melody_b)
        confidence = len(segments) / len(best_scores) if best_scores else 0.0
        return PlagiarismReport(overall, segments, offset, confidence)

    def scan_corpus(self, query: list[int], corpus: list[list[int]]) -> list[tuple[int, PlagiarismReport]]:
        results = []
        for idx, melody in enumerate(corpus):
            report = self.compare_melodies(query, melody)
            if report.similarity_score >= self.threshold:
                results.append((idx, report))
        results.sort(key=lambda x: x[1].similarity_score, reverse=True)
        return results

    @staticmethod
    def _best_transposition(a: list[int], b: list[int]) -> int:
        if not a or not b:
            return 0
        offsets: dict[int, int] = {}
        min_len = min(len(a), len(b))
        for i in range(min_len):
            d = (b[i] - a[i]) % 12
            offsets[d] = offsets.get(d, 0) + 1
        return max(offsets, key=offsets.get) if offsets else 0
