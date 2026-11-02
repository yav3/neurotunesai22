"""Motivic development: thematic transformation, fragmentation, augmentation, inversion.

Implements classical techniques of motivic development: exact/tonal transposition,
inversion, retrograde, augmentation/diminution, fragmentation, sequence,
and motivic similarity via edit distance on interval sequences.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class TransformType(str, Enum):
    TRANSPOSE = "transpose"
    INVERT = "invert"
    RETROGRADE = "retrograde"
    AUGMENT = "augment"
    DIMINISH = "diminish"
    FRAGMENT = "fragment"
    SEQUENCE = "sequence"
    RETROGRADE_INVERSION = "retrograde_inversion"


@dataclass(frozen=True)
class Motive:
    pitches: Tuple[int, ...]      # MIDI pitches
    durations: Tuple[float, ...]  # in beats
    label: str = ""

    @property
    def intervals(self) -> Tuple[int, ...]:
        return tuple(self.pitches[i + 1] - self.pitches[i]
                     for i in range(len(self.pitches) - 1))

    @property
    def contour(self) -> Tuple[int, ...]:
        return tuple(
            1 if d > 0 else (-1 if d < 0 else 0)
            for d in self.intervals
        )


def transpose(motive: Motive, semitones: int) -> Motive:
    """Exact transposition by semitones."""
    return Motive(
        pitches=tuple(p + semitones for p in motive.pitches),
        durations=motive.durations,
        label=f"{motive.label}_T{semitones}",
    )


def invert(motive: Motive, axis: Optional[int] = None) -> Motive:
    """Interval inversion around axis (default: first pitch)."""
    ax = axis if axis is not None else motive.pitches[0]
    return Motive(
        pitches=tuple(2 * ax - p for p in motive.pitches),
        durations=motive.durations,
        label=f"{motive.label}_I",
    )


def retrograde(motive: Motive) -> Motive:
    """Reverse pitch and duration order."""
    return Motive(
        pitches=motive.pitches[::-1],
        durations=motive.durations[::-1],
        label=f"{motive.label}_R",
    )


def retrograde_inversion(motive: Motive, axis: Optional[int] = None) -> Motive:
    """Retrograde of the inversion."""
    inv = invert(motive, axis)
    return Motive(
        pitches=inv.pitches[::-1],
        durations=inv.durations[::-1],
        label=f"{motive.label}_RI",
    )


def augment(motive: Motive, factor: float = 2.0) -> Motive:
    """Rhythmic augmentation: multiply durations."""
    return Motive(
        pitches=motive.pitches,
        durations=tuple(d * factor for d in motive.durations),
        label=f"{motive.label}_aug{factor}",
    )


def diminish(motive: Motive, factor: float = 2.0) -> Motive:
    """Rhythmic diminution: divide durations."""
    return Motive(
        pitches=motive.pitches,
        durations=tuple(d / factor for d in motive.durations),
        label=f"{motive.label}_dim{factor}",
    )


def fragment(motive: Motive, start: int, length: int) -> Motive:
    """Extract a fragment from the motive."""
    end = min(start + length, len(motive.pitches))
    return Motive(
        pitches=motive.pitches[start:end],
        durations=motive.durations[start:end],
        label=f"{motive.label}_frag{start}_{length}",
    )


def sequence(motive: Motive, steps: List[int]) -> List[Motive]:
    """Generate a melodic sequence: repeat motive at each transposition level."""
    return [transpose(motive, s) for s in steps]


def tonal_transpose(motive: Motive, steps: int,
                    scale: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)) -> Motive:
    """Transpose within a diatonic scale by scale steps."""
    scale_arr = np.array(sorted(set(s % 12 for s in scale)))
    new_pitches = []
    for p in motive.pitches:
        pc = p % 12
        octave = p // 12
        idx = np.argmin(np.abs(scale_arr - pc))
        new_idx = (idx + steps) % len(scale_arr)
        oct_shift = (idx + steps) // len(scale_arr)
        new_pitches.append(int(scale_arr[new_idx] + (octave + oct_shift) * 12))
    return Motive(
        pitches=tuple(new_pitches),
        durations=motive.durations,
        label=f"{motive.label}_Td{steps}",
    )


def interval_edit_distance(a: Motive, b: Motive) -> int:
    """Edit distance on interval sequences (Levenshtein)."""
    sa, sb = a.intervals, b.intervals
    m, n = len(sa), len(sb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if sa[i - 1] == sb[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[m][n]


def contour_similarity(a: Motive, b: Motive) -> float:
    """Fraction of matching contour directions."""
    ca, cb = a.contour, b.contour
    if not ca or not cb:
        return 0.0
    min_len = min(len(ca), len(cb))
    matches = sum(1 for i in range(min_len) if ca[i] == cb[i])
    return matches / min_len


def develop(motive: Motive, transforms: List[TransformType],
            **kwargs) -> List[Motive]:
    """Apply a chain of transformations to a motive."""
    result = [motive]
    current = motive
    for t in transforms:
        if t == TransformType.TRANSPOSE:
            current = transpose(current, kwargs.get("semitones", 2))
        elif t == TransformType.INVERT:
            current = invert(current)
        elif t == TransformType.RETROGRADE:
            current = retrograde(current)
        elif t == TransformType.RETROGRADE_INVERSION:
            current = retrograde_inversion(current)
        elif t == TransformType.AUGMENT:
            current = augment(current, kwargs.get("factor", 2.0))
        elif t == TransformType.DIMINISH:
            current = diminish(current, kwargs.get("factor", 2.0))
        elif t == TransformType.FRAGMENT:
            current = fragment(current, kwargs.get("start", 0),
                               kwargs.get("length", 3))
        elif t == TransformType.SEQUENCE:
            seq = sequence(current, kwargs.get("steps", [0, 2, 4]))
            result.extend(seq[1:])
            current = seq[-1] if seq else current
            continue
        result.append(current)
    return result


def find_motive_occurrences(melody_pitches: np.ndarray,
                            motive: Motive,
                            max_transposition: int = 12) -> List[Tuple[int, int]]:
    """Find all transposed occurrences of motive in melody.

    Returns list of (start_index, transposition_semitones).
    """
    m_intervals = np.array(motive.intervals)
    if len(m_intervals) == 0:
        return []
    mel = melody_pitches.astype(int)
    mel_intervals = np.diff(mel)
    results = []
    for i in range(len(mel_intervals) - len(m_intervals) + 1):
        segment = mel_intervals[i:i + len(m_intervals)]
        if np.array_equal(segment, m_intervals):
            transposition = int(mel[i] - motive.pitches[0])
            if abs(transposition) <= max_transposition:
                results.append((i, transposition))
    return results


def _rhythm_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    va = np.array(a)
    vb = np.array(b)
    if len(va) != len(vb):
        maxlen = max(len(va), len(vb))
        va = np.pad(va, (0, maxlen - len(va)))
        vb = np.pad(vb, (0, maxlen - len(vb)))
    return float(np.sqrt(np.sum((va - vb) ** 2)))


def motive_similarity(a: Motive, b: Motive,
                      pitch_weight: float = 0.6,
                      rhythm_weight: float = 0.4) -> float:
    """Weighted combination of pitch-class edit distance and rhythm distance."""
    pca = [p % 12 for p in a.pitches]
    pcb = [p % 12 for p in b.pitches]
    max_len = max(len(pca), len(pcb), 1)
    pc_dist = interval_edit_distance(
        Motive(tuple(pca), a.durations), Motive(tuple(pcb), b.durations))
    pc_sim = 1.0 - pc_dist / max_len
    max_dur = max(sum(a.durations), sum(b.durations), 1e-10)
    r_dist = _rhythm_distance(a.durations, b.durations)
    r_sim = max(0.0, 1.0 - r_dist / max_dur)
    return float(np.clip(pitch_weight * pc_sim + rhythm_weight * r_sim, 0, 1))


def _contour_correlation(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    minlen = min(len(va), len(vb))
    va, vb = va[:minlen], vb[:minlen]
    sa, sb = float(np.std(va)), float(np.std(vb))
    if sa < 1e-10 or sb < 1e-10:
        return 1.0 if np.allclose(va, vb) else 0.0
    return float(np.corrcoef(va, vb)[0, 1])


def detect_motive_recurrence(
    pitches: np.ndarray,
    motive: Motive,
    threshold: float = 0.7,
) -> List[Tuple[int, float]]:
    """Sliding-window contour correlation to find approximate motive recurrences."""
    n = len(pitches)
    m = len(motive.pitches)
    if m < 2 or n < m:
        return []
    mc = motive.contour
    hits: List[Tuple[int, float]] = []
    for i in range(n - m + 1):
        window = pitches[i:i + m]
        wc = tuple(
            1 if window[j + 1] > window[j]
            else (-1 if window[j + 1] < window[j] else 0)
            for j in range(m - 1)
        )
        corr = _contour_correlation(mc, wc)
        if corr >= threshold:
            hits.append((i, corr))
    return hits


def development_density(
    motives: List[Motive],
    total_duration: float,
) -> float:
    """Count of motivic statements per unit time (Schoenberg's developing-variation density)."""
    if total_duration <= 0:
        return 0.0
    return len(motives) / total_duration


def motive_network(motives: List[Motive]) -> np.ndarray:
    """Build adjacency matrix of pairwise motive similarity scores."""
    n = len(motives)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = motive_similarity(motives[i], motives[j])
            mat[i, j] = mat[j, i] = s
    np.fill_diagonal(mat, 1.0)
    return mat


def contour_reduction(pitches: np.ndarray, depth: int = 1) -> List[int]:
    """Marvin-Laprade contour reduction: retain only local extrema."""
    pts = list(int(p) for p in pitches)
    for _ in range(depth):
        if len(pts) <= 2:
            break
        reduced = [pts[0]]
        for i in range(1, len(pts) - 1):
            if (pts[i] > pts[i - 1] and pts[i] > pts[i + 1]) or \
               (pts[i] < pts[i - 1] and pts[i] < pts[i + 1]):
                reduced.append(pts[i])
        reduced.append(pts[-1])
        pts = reduced
    return pts


def interval_class_vector(m: Motive) -> np.ndarray:
    """Forte-style IC vector from the motive's intervals."""
    vec = np.zeros(7, dtype=int)
    for iv in m.intervals:
        ic = abs(iv) % 12
        if ic > 6:
            ic = 12 - ic
        vec[ic] += 1
    return vec


def motive_complexity(m: Motive) -> float:
    """Composite complexity: range × direction-changes × pitch-class variety."""
    if len(m.pitches) < 2:
        return 0.0
    ivs = np.array(m.intervals)
    pitch_range = float(np.ptp(np.array(m.pitches)))
    direction_changes = sum(
        1 for i in range(len(ivs) - 1) if ivs[i] * ivs[i + 1] < 0)
    unique_pcs = len(set(p % 12 for p in m.pitches))
    n = len(m.pitches)
    range_norm = pitch_range / max(n * 2, 1)
    dir_norm = direction_changes / max(n - 2, 1)
    pc_norm = unique_pcs / min(n, 12)
    return float(np.clip(0.4 * range_norm + 0.3 * dir_norm + 0.3 * pc_norm, 0, 1))


def find_latent_motives(
    pitches: np.ndarray,
    min_length: int = 3,
    max_length: int = 8,
    similarity_threshold: float = 0.8,
) -> List[Motive]:
    """Discover recurring pitch-class patterns (Reti latent-motive detection)."""
    n = len(pitches)
    from typing import Dict
    candidates: Dict[str, Motive] = {}
    for length in range(min_length, min(max_length + 1, n + 1)):
        for start in range(n - length + 1):
            seg = tuple(int(p) for p in pitches[start:start + length])
            pcs = tuple(p % 12 for p in seg)
            key = str(pcs)
            if key not in candidates:
                candidates[key] = Motive(seg, tuple(1.0 for _ in seg),
                                         f"latent_{start}_{length}")
    motives = list(candidates.values())
    if len(motives) <= 1:
        return motives
    keep = []
    for m in motives:
        is_dup = any(motive_similarity(m, k) >= similarity_threshold for k in keep)
        if not is_dup:
            keep.append(m)
    return keep


def motive_catalogue(motives: List[Motive]) -> dict:
    """Group motives by contour shape."""
    groups: dict = {}
    for m in motives:
        key = str(m.contour)
        groups.setdefault(key, []).append(m)
    return groups
