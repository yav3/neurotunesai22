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


# ---------------------------------------------------------------------------
# Narmour implication-realization model (1990)
# ---------------------------------------------------------------------------

class NarmourExpectancy:
    """Narmour IR model: proximity, direction, registral return, closure."""

    @staticmethod
    def proximity(realized_interval: int) -> float:
        return max(0.0, 1.0 - abs(realized_interval) / 12.0)

    @staticmethod
    def direction(implicative: int, realized: int) -> float:
        same_dir = (implicative > 0 and realized > 0) or (implicative < 0 and realized < 0)
        if abs(implicative) <= 5:
            return 1.0 if same_dir else 0.0
        return 0.0 if same_dir else 1.0

    @staticmethod
    def registral_return(implicative: int, realized: int) -> float:
        same_dir = (implicative > 0 and realized > 0) or (implicative < 0 and realized < 0)
        if abs(implicative) > 5 and not same_dir:
            return max(0.0, 1.0 - abs(abs(implicative) - abs(realized)) / 6.0)
        return 0.0

    @staticmethod
    def closure(realized: int, implicative: int) -> float:
        same_dir = (implicative > 0 and realized > 0) or (implicative < 0 and realized < 0)
        return 1.0 if abs(realized) <= 2 and not same_dir else 0.0

    def combined(self, implicative: int, realized: int,
                 w: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)) -> float:
        return (w[0] * self.proximity(realized)
                + w[1] * self.direction(implicative, realized)
                + w[2] * self.registral_return(implicative, realized)
                + w[3] * self.closure(realized, implicative))


def narmour_curve(melody: list[int]) -> np.ndarray:
    if len(melody) < 3:
        return np.array([])
    intervals = [melody[i + 1] - melody[i] for i in range(len(melody) - 1)]
    model = NarmourExpectancy()
    return np.array([model.combined(intervals[i], intervals[i + 1])
                     for i in range(len(intervals) - 1)])


# ---------------------------------------------------------------------------
# Statistical learning (exposure-based familiarity)
# ---------------------------------------------------------------------------

class StatisticalLearner:
    """Exposure-based distributional learning of melodic patterns."""

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self._counts: dict[tuple[int, ...], int] = defaultdict(int)
        self._total = 0

    def expose(self, sequence: list[int]) -> None:
        for i in range(len(sequence) - self.window + 1):
            self._counts[tuple(sequence[i : i + self.window])] += 1
            self._total += 1

    def familiarity(self, pattern: tuple[int, ...]) -> float:
        if self._total == 0:
            return 0.0
        return self._counts.get(pattern, 0) / self._total

    def novelty(self, sequence: list[int]) -> float:
        if len(sequence) < self.window or self._total == 0:
            return 1.0
        scores = []
        for i in range(len(sequence) - self.window + 1):
            scores.append(1.0 - self.familiarity(tuple(sequence[i : i + self.window])))
        return float(np.mean(scores))


def surprise_curve(sequence: list[int], model: MarkovExpectation) -> np.ndarray:
    """Per-note information content over a melody given a trained model."""
    ic = np.zeros(len(sequence))
    for i in range(model.order, len(sequence)):
        ctx = sequence[i - model.order : i]
        ic[i] = model.information_content(ctx, sequence[i])
    return ic
