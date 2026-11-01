"""Playlist transition quality scoring and optimization."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class TransitionFeatures:
    tempo_diff: float
    key_compatibility: float
    energy_diff: float
    timbre_distance: float
    mood_distance: float


CAMELOT_WHEEL = {
    (0, 0): 8,  (1, 0): 3, (2, 0): 10, (3, 0): 5,  (4, 0): 12,
    (5, 0): 7,  (6, 0): 2, (7, 0): 9,  (8, 0): 4,  (9, 0): 11,
    (10, 0): 6, (11, 0): 1,
    (0, 1): 5,  (1, 1): 12, (2, 1): 7,  (3, 1): 2,  (4, 1): 9,
    (5, 1): 4,  (6, 1): 11, (7, 1): 6,  (8, 1): 1,  (9, 1): 8,
    (10, 1): 3, (11, 1): 10,
}


def key_compatibility_score(key1: int, key2: int, mode1: int = 0, mode2: int = 0) -> float:
    c1 = CAMELOT_WHEEL.get((key1 % 12, mode1 % 2), 1)
    c2 = CAMELOT_WHEEL.get((key2 % 12, mode2 % 2), 1)
    dist = min(abs(c1 - c2), 12 - abs(c1 - c2))
    if dist == 0:
        score = 1.0
    elif dist == 1:
        score = 0.85
    elif dist == 2:
        score = 0.5
    else:
        score = max(0.0, 1.0 - dist * 0.2)
    if mode1 != mode2 and dist <= 1:
        score = max(score, 0.8)
    return score


def tempo_compatibility(bpm1: float, bpm2: float) -> float:
    if bpm1 <= 0 or bpm2 <= 0:
        return 0.0
    ratio = bpm1 / bpm2
    candidates = [ratio, ratio * 2, ratio / 2]
    best = min(candidates, key=lambda r: abs(r - 1.0))
    return float(np.exp(-((best - 1.0) ** 2) / (2 * 0.05**2)))


def transition_score(feat: TransitionFeatures) -> float:
    w = [0.25, 0.25, 0.2, 0.15, 0.15]
    tempo_s = float(np.exp(-(feat.tempo_diff**2) / (2 * 20**2)))
    key_s = feat.key_compatibility
    energy_s = float(1.0 / (1.0 + np.exp(5 * (abs(feat.energy_diff) - 0.3))))
    timbre_s = float(1.0 / (1.0 + feat.timbre_distance))
    mood_s = float(1.0 / (1.0 + feat.mood_distance))
    return float(np.dot(w, [tempo_s, key_s, energy_s, timbre_s, mood_s]))


ENERGY_TEMPLATES = {
    "ramp_up": lambda n: np.linspace(0, 1, n),
    "chill_down": lambda n: np.linspace(1, 0, n),
    "wave": lambda n: 0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, n)),
    "peak_valley": lambda n: np.concatenate([np.linspace(0, 1, n // 2), np.linspace(1, 0, n - n // 2)]),
}


def energy_curve_fitness(energies: list[float], target_curve: str = "ramp_up") -> float:
    e = np.asarray(energies, dtype=float)
    n = len(e)
    if n < 2:
        return 0.0
    template_fn = ENERGY_TEMPLATES.get(target_curve)
    if template_fn is None:
        return 0.0
    tmpl = template_fn(n)
    e_std = np.std(e)
    t_std = np.std(tmpl)
    if e_std < 1e-10 or t_std < 1e-10:
        return 0.0
    corr = float(np.corrcoef(e, tmpl)[0, 1])
    return max(0.0, corr)


def crossfade_duration(bpm1: float, bpm2: float, energy1: float = 0.5, energy2: float = 0.5) -> float:
    tempo_match = tempo_compatibility(bpm1, bpm2)
    avg_energy = (energy1 + energy2) / 2.0
    base = 4.0
    if tempo_match > 0.9:
        base = 8.0
    elif tempo_match < 0.3:
        base = 2.0
    energy_factor = 1.0 - 0.5 * avg_energy
    return float(np.clip(base * energy_factor, 1.0, 16.0))


class PlaylistOptimizer:
    def __init__(self, transition_scores: np.ndarray):
        self.scores = np.asarray(transition_scores, dtype=float)
        self.n = self.scores.shape[0]

    def nearest_neighbor_order(self, start: int = 0) -> list[int]:
        visited = {start}
        order = [start]
        current = start
        for _ in range(self.n - 1):
            row = self.scores[current].copy()
            for v in visited:
                row[v] = -np.inf
            nxt = int(np.argmax(row))
            order.append(nxt)
            visited.add(nxt)
            current = nxt
        return order

    def two_opt_improve(self, order: list[int], max_iters: int = 100) -> list[int]:
        best = list(order)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best)):
                    old_cost = self.scores[best[i - 1], best[i]] + (
                        self.scores[best[j], best[j + 1]] if j + 1 < len(best) else 0
                    )
                    new_cost = self.scores[best[i - 1], best[j]] + (
                        self.scores[best[i], best[j + 1]] if j + 1 < len(best) else 0
                    )
                    if new_cost > old_cost:
                        best[i:j + 1] = best[i:j + 1][::-1]
                        improved = True
        return best

    def optimize(self, start: int = 0) -> list[int]:
        nn = self.nearest_neighbor_order(start)
        return self.two_opt_improve(nn)

    def total_score(self, order: list[int]) -> float:
        return float(sum(self.scores[order[i], order[i + 1]] for i in range(len(order) - 1)))
