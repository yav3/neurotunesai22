"""Pattern discovery: suffix arrays, repeated motif extraction, compression similarity."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence

import numpy as np


@dataclass
class Pattern:
    tokens: Tuple[int, ...]
    positions: List[int]
    length: int
    frequency: int


@dataclass
class PatternStats:
    total_patterns: int
    coverage: float
    avg_length: float
    max_length: int
    compression_ratio: float


def build_suffix_array(seq: Sequence[int]) -> np.ndarray:
    n = len(seq)
    sa = np.arange(n)
    rank = np.array(list(seq), dtype=np.int64)
    tmp = np.empty(n, dtype=np.int64)
    k = 1
    while k < n:
        def sort_key(i: int) -> Tuple[int, int]:
            r2 = rank[i + k] if i + k < n else -1
            return (int(rank[i]), int(r2))
        sa = np.array(sorted(range(n), key=sort_key))
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i - 1]]
            if sort_key(int(sa[i])) != sort_key(int(sa[i - 1])):
                tmp[sa[i]] += 1
        rank = tmp.copy()
        if int(rank[sa[-1]]) == n - 1:
            break
        k *= 2
    return sa


def lcp_array(seq: Sequence[int], sa: np.ndarray) -> np.ndarray:
    n = len(seq)
    rank = np.empty(n, dtype=np.int64)
    for i in range(n):
        rank[sa[i]] = i
    lcp = np.zeros(n, dtype=np.int64)
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and seq[i + h] == seq[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
        else:
            h = 0
    return lcp


def find_repeated_substrings(seq: Sequence[int],
                             min_len: int = 2,
                             min_freq: int = 2) -> List[Pattern]:
    if len(seq) < min_len:
        return []
    sa = build_suffix_array(seq)
    lcp = lcp_array(seq, sa)
    patterns: Dict[Tuple[int, ...], List[int]] = {}
    for i in range(1, len(sa)):
        common = int(lcp[i])
        for length in range(min_len, common + 1):
            substr = tuple(seq[sa[i]:sa[i] + length])
            if substr not in patterns:
                patterns[substr] = []
            if sa[i] not in patterns[substr]:
                patterns[substr].append(int(sa[i]))
            if sa[i - 1] not in patterns[substr]:
                patterns[substr].append(int(sa[i - 1]))
    result = []
    for tokens, positions in patterns.items():
        if len(positions) >= min_freq:
            result.append(Pattern(
                tokens=tokens,
                positions=sorted(positions),
                length=len(tokens),
                frequency=len(positions),
            ))
    result.sort(key=lambda p: p.length * p.frequency, reverse=True)
    return result


def maximum_repeating_pattern(seq: Sequence[int]) -> Optional[Pattern]:
    patterns = find_repeated_substrings(seq, min_len=2, min_freq=2)
    if not patterns:
        return None
    return max(patterns, key=lambda p: p.length * p.frequency)


def coverage_ratio(seq: Sequence[int], patterns: List[Pattern]) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    covered = np.zeros(n, dtype=bool)
    for p in patterns:
        for pos in p.positions:
            covered[pos:pos + p.length] = True
    return float(np.mean(covered))


def compression_distance(seq_a: Sequence[int],
                         seq_b: Sequence[int]) -> float:
    def _compressed_len(s: Sequence[int]) -> int:
        patterns = find_repeated_substrings(s, min_len=2, min_freq=2)
        saved = sum(p.length * (p.frequency - 1) for p in patterns)
        return max(1, len(s) - saved)
    ca = _compressed_len(seq_a)
    cb = _compressed_len(seq_b)
    cab = _compressed_len(list(seq_a) + list(seq_b))
    return (cab - min(ca, cb)) / max(ca, cb, 1)


def normalized_compression_distance(seq_a: Sequence[int],
                                    seq_b: Sequence[int]) -> float:
    return compression_distance(seq_a, seq_b)


def tandem_repeat_finder(seq: Sequence[int],
                         min_period: int = 1,
                         max_period: int = 16) -> List[Tuple[int, int, int]]:
    n = len(seq)
    repeats: List[Tuple[int, int, int]] = []
    for period in range(min_period, min(max_period + 1, n)):
        i = 0
        while i + period <= n:
            count = 0
            while i + (count + 1) * period <= n:
                chunk = seq[i + count * period:i + (count + 1) * period]
                ref = seq[i:i + period]
                if tuple(chunk) != tuple(ref):
                    break
                count += 1
            if count >= 2:
                repeats.append((i, period, count))
                i += count * period
            else:
                i += 1
    return repeats


def string_match_positions(text: Sequence[int],
                           pattern: Sequence[int]) -> List[int]:
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    positions: List[int] = []
    pat_tuple = tuple(pattern)
    for i in range(n - m + 1):
        if tuple(text[i:i + m]) == pat_tuple:
            positions.append(i)
    return positions


def interval_sequence(pitches: Sequence[int]) -> List[int]:
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def contour_sequence(pitches: Sequence[int]) -> List[int]:
    contour = []
    for i in range(len(pitches) - 1):
        if pitches[i + 1] > pitches[i]:
            contour.append(1)
        elif pitches[i + 1] < pitches[i]:
            contour.append(-1)
        else:
            contour.append(0)
    return contour


def pattern_stats(seq: Sequence[int], patterns: List[Pattern]) -> PatternStats:
    if not patterns:
        return PatternStats(0, 0.0, 0.0, 0, 1.0)
    total_saved = sum(p.length * (p.frequency - 1) for p in patterns)
    compressed = max(1, len(seq) - total_saved)
    return PatternStats(
        total_patterns=len(patterns),
        coverage=coverage_ratio(seq, patterns),
        avg_length=float(np.mean([p.length for p in patterns])),
        max_length=max(p.length for p in patterns),
        compression_ratio=compressed / max(len(seq), 1),
    )


def discover_patterns(pitches: Sequence[int],
                      use_intervals: bool = True,
                      min_len: int = 3,
                      min_freq: int = 2) -> Tuple[List[Pattern], PatternStats]:
    seq = interval_sequence(pitches) if use_intervals else list(pitches)
    patterns = find_repeated_substrings(seq, min_len=min_len, min_freq=min_freq)
    stats = pattern_stats(seq, patterns)
    return patterns, stats
