"""Optimal voice leading via minimum-cost matching.

Implements Tymoczko's geometric theory of voice leading:
- Minimum total voice-leading distance between chords
- Efficient voice leading via Hungarian algorithm (O(n^3))
- Parsimonious voice leading (neo-Riemannian)
- Voice-leading zones (Cohn's hexatonic systems)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


def pitch_class_distance(a: int, b: int) -> int:
    """Shortest distance in pitch-class space (mod 12)."""
    d = (b - a) % 12
    return min(d, 12 - d)


def voice_leading_distance(chord_a: np.ndarray, chord_b: np.ndarray,
                           metric: str = "taxicab") -> float:
    """Total voice-leading distance between two chords.

    Assumes voices are matched by index. For optimal matching
    use `optimal_voice_leading`.

    Metrics:
      taxicab: sum of |intervals| (L1)
      euclidean: sqrt(sum of intervals^2) (L2)
      max: max individual voice motion (L∞)
    """
    intervals = np.array([(b - a) % 12 for a, b in zip(chord_a, chord_b)])
    intervals = np.minimum(intervals, 12 - intervals)
    if metric == "taxicab":
        return float(np.sum(intervals))
    elif metric == "euclidean":
        return float(np.sqrt(np.sum(intervals ** 2)))
    elif metric == "max":
        return float(np.max(intervals))
    return float(np.sum(intervals))


def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, float]:
    """Hungarian algorithm for minimum-cost assignment.

    O(n^3) implementation for the assignment problem.
    Returns (assignment, total_cost).
    """
    n = cost.shape[0]
    m = cost.shape[1]
    size = max(n, m)
    padded = np.full((size, size), 0.0)
    padded[:n, :m] = cost
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if j1 < 0:
                break
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    assignment = np.zeros(n, dtype=int)
    total = 0.0
    for j in range(1, size + 1):
        if p[j] > 0 and p[j] <= n and j <= m:
            assignment[p[j] - 1] = j - 1
            total += cost[p[j] - 1, j - 1]
    return assignment, total


def optimal_voice_leading(chord_a: np.ndarray,
                          chord_b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find minimum-cost voice assignment between two chords.

    Uses the Hungarian algorithm to find the assignment of voices
    in chord_a to voices in chord_b that minimizes total semitone motion.
    Returns (assignment indices, total distance).
    """
    n = len(chord_a)
    m = len(chord_b)
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = pitch_class_distance(int(chord_a[i]) % 12,
                                               int(chord_b[j]) % 12)
    assignment, total = _hungarian(cost)
    return assignment, total


class NeoRiemannianOp(Enum):
    """Neo-Riemannian transformations."""
    P = "P"   # parallel: C major ↔ C minor
    R = "R"   # relative: C major ↔ A minor
    L = "L"   # leading-tone: C major ↔ E minor


def _triad_quality(pitches: np.ndarray) -> str:
    """Determine if a triad (3 pitch classes) is major or minor."""
    pcs = sorted(set(p % 12 for p in pitches))
    if len(pcs) < 3:
        return "unknown"
    intervals = [(pcs[1] - pcs[0]) % 12, (pcs[2] - pcs[1]) % 12]
    if intervals == [4, 3]:
        return "major"
    elif intervals == [3, 4]:
        return "minor"
    return "unknown"


def neo_riemannian_transform(triad: np.ndarray,
                              op: NeoRiemannianOp) -> np.ndarray:
    """Apply a neo-Riemannian transformation to a triad.

    P: moves the third by a semitone (major ↔ minor, same root)
    R: moves the fifth (major) or root (minor) by a whole step
    L: moves the root (major) or fifth (minor) by a semitone
    """
    pcs = np.array([p % 12 for p in triad])
    quality = _triad_quality(pcs)
    root = pcs[0]
    result = pcs.copy()
    if op == NeoRiemannianOp.P:
        if quality == "major":
            result[1] = (root + 3) % 12  # lower third
        elif quality == "minor":
            result[1] = (root + 4) % 12  # raise third
    elif op == NeoRiemannianOp.R:
        if quality == "major":
            result[2] = (root + 9) % 12  # lower fifth to get relative minor
        elif quality == "minor":
            result[0] = (pcs[2] - 7) % 12  # raise root
    elif op == NeoRiemannianOp.L:
        if quality == "major":
            result[0] = (root - 1) % 12  # lower root
        elif quality == "minor":
            result[2] = (pcs[2] + 1) % 12  # raise fifth
    return result


def neo_riemannian_distance(chord_a: np.ndarray,
                            chord_b: np.ndarray,
                            max_depth: int = 4) -> int:
    """Minimum number of PLR operations to transform chord_a → chord_b.

    BFS over the Tonnetz, bounded by max_depth.
    Returns max_depth + 1 if unreachable.
    """
    target = frozenset(int(p) % 12 for p in chord_b)
    start = frozenset(int(p) % 12 for p in chord_a)
    if start == target:
        return 0
    queue = [(np.array(sorted(start)), 0)]
    visited = {start}
    ops = [NeoRiemannianOp.P, NeoRiemannianOp.R, NeoRiemannianOp.L]
    for current, depth in queue:
        if depth >= max_depth:
            continue
        for op in ops:
            transformed = neo_riemannian_transform(current, op)
            key = frozenset(int(p) % 12 for p in transformed)
            if key == target:
                return depth + 1
            if key not in visited:
                visited.add(key)
                queue.append((np.array(sorted(key)), depth + 1))
    return max_depth + 1


@dataclass
class VoiceLeadingPath:
    """A sequence of chords with voice-leading distances."""
    chords: List[np.ndarray]
    distances: List[float]
    total_distance: float
    smoothness: float  # average distance per transition


def analyze_voice_leading_path(chords: List[np.ndarray]) -> VoiceLeadingPath:
    """Analyze the voice-leading efficiency of a chord progression."""
    distances = []
    for i in range(len(chords) - 1):
        _, dist = optimal_voice_leading(chords[i], chords[i + 1])
        distances.append(dist)
    total = sum(distances)
    smoothness = total / max(1, len(distances))
    return VoiceLeadingPath(
        chords=chords,
        distances=distances,
        total_distance=total,
        smoothness=smoothness,
    )


def parsimonious_realization(chord_pcs: List[np.ndarray],
                             starting_voicing: np.ndarray) -> List[np.ndarray]:
    """Realize a pitch-class progression with minimal voice motion.

    Given a sequence of pitch-class sets and a starting voicing
    (with specific octave placements), find the voicing of each
    subsequent chord that minimizes total motion from the previous.
    """
    n_voices = len(starting_voicing)
    result = [starting_voicing.copy()]
    current = starting_voicing.copy()
    for pcs in chord_pcs[1:]:
        if len(pcs) != n_voices:
            result.append(pcs.copy())
            current = pcs.copy()
            continue
        candidates = []
        for pc in pcs:
            octave_choices = []
            for oct in range(-1, 9):
                pitch = int(pc) + 12 * oct
                if 21 <= pitch <= 108:
                    octave_choices.append(pitch)
            candidates.append(octave_choices)
        best_voicing = None
        best_cost = float('inf')
        def _search(voice_idx, partial, cost):
            nonlocal best_voicing, best_cost
            if voice_idx == n_voices:
                if cost < best_cost:
                    best_cost = cost
                    best_voicing = partial.copy()
                return
            for pitch in candidates[voice_idx]:
                new_cost = cost + abs(pitch - current[voice_idx])
                if new_cost < best_cost:
                    partial.append(pitch)
                    _search(voice_idx + 1, partial, new_cost)
                    partial.pop()
        _search(0, [], 0.0)
        if best_voicing is not None:
            voicing = np.array(best_voicing)
        else:
            voicing = pcs.copy()
        result.append(voicing)
        current = voicing
    return result
