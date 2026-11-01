"""Fingerprint matching and temporal alignment via histogram voting."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    best_offset: int
    confidence: float
    n_aligned: int
    histogram: dict[int, int]


@dataclass
class MatchResult:
    track_id: str
    confidence: float
    time_offset: int
    n_matching_hashes: int
    alignment: AlignmentResult


def align_matches(candidate_offsets: list[int]) -> AlignmentResult:
    if not candidate_offsets:
        return AlignmentResult(best_offset=0, confidence=0.0, n_aligned=0, histogram={})
    counts = Counter(candidate_offsets)
    best_offset, best_count = counts.most_common(1)[0]
    total = len(candidate_offsets)
    confidence = best_count / total if total > 0 else 0.0
    return AlignmentResult(
        best_offset=best_offset,
        confidence=confidence,
        n_aligned=best_count,
        histogram=dict(counts),
    )


def score_match(alignment: AlignmentResult) -> float:
    if alignment.n_aligned == 0:
        return 0.0
    base = alignment.confidence
    count_bonus = min(alignment.n_aligned / 50.0, 1.0)
    n_bins = len(alignment.histogram)
    peakedness = 1.0 / max(n_bins, 1)
    return 0.5 * base + 0.3 * count_bonus + 0.2 * peakedness


class FingerprintMatcher:
    """End-to-end fingerprint matching pipeline.

    Takes a query fingerprint (list of (hash, offset) pairs) and a database
    (dict mapping track_id to list of (hash, offset) pairs), returns ranked matches.
    """

    def __init__(self, min_matches: int = 3, min_confidence: float = 0.05) -> None:
        self.min_matches = min_matches
        self.min_confidence = min_confidence

    def match(
        self,
        query_fp: list[tuple[int, int]],
        database: dict[str, list[tuple[int, int]]],
    ) -> list[MatchResult]:
        query_hashes = {h: offset for h, offset in query_fp}
        results = []

        for track_id, db_fp in database.items():
            offsets = []
            for h, db_offset in db_fp:
                if h in query_hashes:
                    offsets.append(db_offset - query_hashes[h])

            if len(offsets) < self.min_matches:
                continue

            alignment = align_matches(offsets)
            confidence = score_match(alignment)

            if confidence < self.min_confidence:
                continue

            results.append(
                MatchResult(
                    track_id=track_id,
                    confidence=confidence,
                    time_offset=alignment.best_offset,
                    n_matching_hashes=alignment.n_aligned,
                    alignment=alignment,
                )
            )

        return sorted(results, key=lambda m: -m.confidence)

    def match_partial(
        self,
        query_fp: list[tuple[int, int]],
        database: dict[str, list[tuple[int, int]]],
        query_start: int = 0,
        query_end: int | None = None,
    ) -> list[MatchResult]:
        if query_end is None:
            query_end = max((off for _, off in query_fp), default=0) + 1
        filtered = [(h, off) for h, off in query_fp if query_start <= off < query_end]
        return self.match(filtered, database)
