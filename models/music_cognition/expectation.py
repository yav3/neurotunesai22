"""Musical expectation and surprise: Markov prediction, melodic principles."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


class MarkovExpectation:
    def __init__(self, order: int = 1):
        self.order = order
        self._counts: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._totals: dict[tuple, int] = defaultdict(int)

    def train(self, sequences: list[list[int]]) -> None:
        for seq in sequences:
            for i in range(self.order, len(seq)):
                context = tuple(seq[i - self.order:i])
                self._counts[context][seq[i]] += 1
                self._totals[context] += 1

    def predict(self, context: list[int]) -> dict[int, float]:
        key = tuple(context[-self.order:])
        total = self._totals.get(key, 0)
        if total == 0:
            return {}
        return {p: c / total for p, c in self._counts[key].items()}

    def probability(self, context: list[int], next_pitch: int) -> float:
        dist = self.predict(context)
        return dist.get(next_pitch, 0.0)

    def information_content(self, context: list[int], next_pitch: int) -> float:
        p = self.probability(context, next_pitch)
        return _information_content(p)

    def sequence_entropy(self, sequence: list[int]) -> float:
        if len(sequence) <= self.order:
            return 0.0
        ic_values = []
        for i in range(self.order, len(sequence)):
            context = sequence[i - self.order:i]
            ic_values.append(self.information_content(context, sequence[i]))
        return float(np.mean(ic_values)) if ic_values else 0.0


class MelodicExpectation:
    def surprise(self, interval: int) -> float:
        abs_int = abs(interval)
        if abs_int == 0:
            return 0.1
        if abs_int == 12:
            return 0.3
        return float(min(1.0, abs_int / 12.0))

    def proximity_score(self, interval: int) -> float:
        return 1.0 - self.surprise(interval)

    def registral_return(self, pitches: list[int]) -> float:
        if len(pitches) < 3:
            return 0.0
        mean_pitch = np.mean(pitches)
        returns = 0
        for i in range(2, len(pitches)):
            prev_dir = pitches[i - 1] - pitches[i - 2]
            curr_dir = pitches[i] - pitches[i - 1]
            if prev_dir != 0 and curr_dir != 0:
                if np.sign(prev_dir) != np.sign(curr_dir):
                    if abs(pitches[i] - mean_pitch) < abs(pitches[i - 1] - mean_pitch):
                        returns += 1
        return float(returns / (len(pitches) - 2))


def information_content(prob: float) -> float:
    return _information_content(prob)


def melodic_surprise(interval: int) -> float:
    return MelodicExpectation().surprise(interval)


def _information_content(prob: float) -> float:
    if prob <= 0:
        return float("inf")
    return float(-np.log2(prob))
