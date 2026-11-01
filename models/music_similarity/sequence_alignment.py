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
