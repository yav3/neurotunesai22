"""Sequence alignment algorithms for melodic similarity."""
import numpy as np


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray,
                 metric: str = "euclidean") -> tuple[float, np.ndarray]:
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0, np.empty((0, 2), dtype=int)
    a = seq_a.reshape(n, -1) if seq_a.ndim == 1 else seq_a
    b = seq_b.reshape(m, -1) if seq_b.ndim == 1 else seq_b
    cost_mat = np.full((n + 1, m + 1), np.inf)
    cost_mat[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if metric == "euclidean":
                d = float(np.sqrt(np.sum((a[i - 1] - b[j - 1]) ** 2)))
            else:
                d = float(np.abs(a[i - 1] - b[j - 1]).sum())
            cost_mat[i, j] = d + min(cost_mat[i - 1, j],
                                      cost_mat[i, j - 1],
                                      cost_mat[i - 1, j - 1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1]]
        idx = int(np.argmin(choices))
        if idx == 0:
            i -= 1; j -= 1
        elif idx == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(cost_mat[n, m]), np.array(path, dtype=int)


def edit_distance_melodic(melody_a: np.ndarray, melody_b: np.ndarray,
                          sub_cost: float = 1.0) -> int:
    n, m = len(melody_a), len(melody_b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if melody_a[i - 1] == melody_b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(dp[i - 1, j] + 1,
                               dp[i, j - 1] + 1,
                               dp[i - 1, j - 1] + int(sub_cost))
    return int(dp[n, m])


def longest_common_subsequence(seq_a: np.ndarray, seq_b: np.ndarray,
                               tolerance: float = 0.5) -> int:
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(float(seq_a[i - 1] - seq_b[j - 1])) <= tolerance:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


def smith_waterman_alignment(seq_a: np.ndarray, seq_b: np.ndarray,
                             match_score: float = 2.0,
                             gap_penalty: float = -1.0) -> tuple[float, list]:
    n, m = len(seq_a), len(seq_b)
    H = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = match_score if abs(float(seq_a[i-1] - seq_b[j-1])) < 1.0 else -1.0
            H[i, j] = max(0,
                          H[i-1, j-1] + match,
                          H[i-1, j] + gap_penalty,
                          H[i, j-1] + gap_penalty)
    best = float(np.max(H))
    pos = np.unravel_index(np.argmax(H), H.shape)
    path = []
    i, j = int(pos[0]), int(pos[1])
    while i > 0 and j > 0 and H[i, j] > 0:
        path.append((i - 1, j - 1))
        if H[i-1, j-1] >= H[i-1, j] and H[i-1, j-1] >= H[i, j-1]:
            i -= 1; j -= 1
        elif H[i-1, j] >= H[i, j-1]:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return best, path


def needleman_wunsch(seq_a: np.ndarray, seq_b: np.ndarray,
                     match_score: float = 1.0,
                     mismatch_penalty: float = -1.0,
                     gap_penalty: float = -2.0) -> tuple[float, list]:
    n, m = len(seq_a), len(seq_b)
    F = np.zeros((n + 1, m + 1))
    F[:, 0] = np.arange(n + 1) * gap_penalty
    F[0, :] = np.arange(m + 1) * gap_penalty
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match_score if abs(float(seq_a[i-1] - seq_b[j-1])) < 1.0 else mismatch_penalty
            F[i, j] = max(F[i-1, j-1] + s,
                          F[i-1, j] + gap_penalty,
                          F[i, j-1] + gap_penalty)
    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        s = match_score if abs(float(seq_a[i-1] - seq_b[j-1])) < 1.0 else mismatch_penalty
        if F[i, j] == F[i-1, j-1] + s:
            alignment.append((i-1, j-1))
            i -= 1; j -= 1
        elif F[i, j] == F[i-1, j] + gap_penalty:
            alignment.append((i-1, -1))
            i -= 1
        else:
            alignment.append((-1, j-1))
            j -= 1
    alignment.reverse()
    return float(F[n, m]), alignment


from dataclasses import dataclass
from typing import Optional


@dataclass
class SequenceAlignment:
    """Unified interface for sequence alignment with band constraints."""

    def dtw(
        self, a: np.ndarray, b: np.ndarray, band: Optional[int] = None
    ) -> tuple[float, np.ndarray]:
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0.0, np.empty((0, 2), dtype=int)
        x = a.reshape(n, -1) if a.ndim == 1 else a
        y = b.reshape(m, -1) if b.ndim == 1 else b
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            j_lo = max(1, i - band) if band else 1
            j_hi = min(m, i + band) if band else m
            for j in range(j_lo, j_hi + 1):
                d = float(np.sqrt(np.sum((x[i - 1] - y[j - 1]) ** 2)))
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            choices = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
            idx = int(np.argmin(choices))
            if idx == 0:
                i -= 1
                j -= 1
            elif idx == 1:
                i -= 1
            else:
                j -= 1
        path.reverse()
        return float(cost[n, m]), np.array(path, dtype=int)

    def edit_distance(self, a: np.ndarray, b: np.ndarray, sub_cost: float = 1.0) -> int:
        return edit_distance_melodic(a, b, sub_cost)

    def smith_waterman(
        self, a: np.ndarray, b: np.ndarray, match: float = 2.0, gap: float = -1.0
    ) -> tuple[float, list]:
        return smith_waterman_alignment(a, b, match, gap)

    def normalized_dtw(self, a: np.ndarray, b: np.ndarray) -> float:
        dist, path = self.dtw(a, b)
        return dist / max(len(path), 1)


@dataclass
class ChromaAlignment:
    """Cover-song similarity via chroma sequence alignment."""

    n_chroma: int = 12

    def oti(self, a: np.ndarray, b: np.ndarray) -> int:
        best_shift = 0
        best_sim = -np.inf
        a_mean = np.mean(a, axis=0) if a.ndim == 2 else a
        b_mean = np.mean(b, axis=0) if b.ndim == 2 else b
        for s in range(self.n_chroma):
            shifted = np.roll(b_mean, s)
            sim = float(np.dot(a_mean, shifted))
            if sim > best_sim:
                best_sim = sim
                best_shift = s
        return best_shift

    def cover_song_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        shift = self.oti(a, b)
        b_shifted = np.roll(b, shift, axis=-1)
        aligner = SequenceAlignment()
        dist, _ = aligner.dtw(a, b_shifted)
        max_len = max(a.shape[0], b_shifted.shape[0])
        return 1.0 / (1.0 + dist / max(max_len, 1))


@dataclass
class MelodyAlignment:
    """Pitch-interval and contour similarity for melodic comparison."""

    def interval_sequence(self, pitches: np.ndarray) -> np.ndarray:
        if len(pitches) < 2:
            return np.array([])
        return np.diff(pitches)

    def interval_edit_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        ia = self.interval_sequence(a)
        ib = self.interval_sequence(b)
        if len(ia) == 0 or len(ib) == 0:
            return max(len(ia), len(ib))
        return edit_distance_melodic(ia, ib)

    def contour_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        ca = np.sign(np.diff(a)) if len(a) > 1 else np.array([])
        cb = np.sign(np.diff(b)) if len(b) > 1 else np.array([])
        if len(ca) == 0 or len(cb) == 0:
            return 0.0
        lcs = longest_common_subsequence(ca, cb, tolerance=0.5)
        return lcs / max(len(ca), len(cb))

    def pitch_class_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        pa = np.mod(a, 12)
        pb = np.mod(b, 12)
        hist_a = np.bincount(pa.astype(int) % 12, minlength=12).astype(float)
        hist_b = np.bincount(pb.astype(int) % 12, minlength=12).astype(float)
        ha = hist_a / max(np.sum(hist_a), 1)
        hb = hist_b / max(np.sum(hist_b), 1)
        return float(np.sum(np.minimum(ha, hb)))


@dataclass
class RhythmAlignment:
    """Rhythmic pattern similarity via onset timing."""

    def onset_pattern_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return float(max(len(a), len(b)))
        aligner = SequenceAlignment()
        dist, _ = aligner.dtw(a.reshape(-1, 1), b.reshape(-1, 1))
        return dist

    def ioi_sequence_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        ioi_a = np.diff(a) if len(a) > 1 else np.array([])
        ioi_b = np.diff(b) if len(b) > 1 else np.array([])
        if len(ioi_a) == 0 or len(ioi_b) == 0:
            return float(max(len(ioi_a), len(ioi_b)))
        ratio_a = ioi_a / max(np.mean(ioi_a), 1e-10)
        ratio_b = ioi_b / max(np.mean(ioi_b), 1e-10)
        aligner = SequenceAlignment()
        dist, _ = aligner.dtw(ratio_a.reshape(-1, 1), ratio_b.reshape(-1, 1))
        return dist

    def groove_similarity(self, a: np.ndarray, b: np.ndarray, grid: float = 0.125) -> float:
        if len(a) == 0 or len(b) == 0:
            return 0.0
        qa = np.round(a / grid) * grid
        qb = np.round(b / grid) * grid
        max_t = max(qa[-1], qb[-1]) + grid
        n_bins = int(max_t / grid) + 1
        ha = np.zeros(n_bins)
        hb = np.zeros(n_bins)
        for t in qa:
            idx = int(round(t / grid))
            if 0 <= idx < n_bins:
                ha[idx] = 1
        for t in qb:
            idx = int(round(t / grid))
            if 0 <= idx < n_bins:
                hb[idx] = 1
        intersection = np.sum(ha * hb)
        union = np.sum(np.clip(ha + hb, 0, 1))
        return float(intersection / max(union, 1))
