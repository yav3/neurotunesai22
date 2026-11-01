"""Musical tension modeling: harmonic distance, voice leading, tension curves."""
from __future__ import annotations
import numpy as np

FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]


class TensionCurve:
    def __init__(self, w_harmonic: float = 0.4, w_dissonance: float = 0.3, w_voice_leading: float = 0.3):
        self._w = (w_harmonic, w_dissonance, w_voice_leading)

    def tension(self, chord_a: list[int], chord_b: list[int]) -> float:
        h = harmonic_tension(chord_a[0] % 12 if chord_a else 0, chord_b[0] % 12 if chord_b else 0)
        d = _dissonance_score(chord_b)
        v = voice_leading_cost(chord_a, chord_b) / max(len(chord_a), len(chord_b), 1) / 12.0
        return float(self._w[0] * h + self._w[1] * d + self._w[2] * min(1.0, v))

    def curve(self, chord_sequence: list[list[int]]) -> list[float]:
        if len(chord_sequence) < 2:
            return [0.0] * len(chord_sequence)
        tensions = [0.0]
        for i in range(1, len(chord_sequence)):
            tensions.append(self.tension(chord_sequence[i - 1], chord_sequence[i]))
        return tensions


class SurpriseTracker:
    def __init__(self):
        from models.music_cognition.expectation import MarkovExpectation
        self._model = MarkovExpectation(order=1)
        self._trained = False

    def train(self, sequences: list[list[int]]) -> None:
        self._model.train(sequences)
        self._trained = True

    def track(self, sequence: list[int]) -> list[float]:
        if not self._trained or len(sequence) < 2:
            return [0.0] * len(sequence)
        surprises = [0.0]
        for i in range(1, len(sequence)):
            ic = self._model.information_content(sequence[:i], sequence[i])
            surprises.append(min(ic, 20.0))
        return surprises


def harmonic_tension(root_a: int, root_b: int) -> float:
    pos_a = FIFTHS_ORDER.index(root_a % 12)
    pos_b = FIFTHS_ORDER.index(root_b % 12)
    dist = min(abs(pos_a - pos_b), 12 - abs(pos_a - pos_b))
    return float(dist / 6.0)


def voice_leading_cost(chord_a: list[int], chord_b: list[int]) -> int:
    if not chord_a or not chord_b:
        return 0
    pairs = minimal_voice_leading(chord_a, chord_b)
    return sum(abs(b - a) for a, b in pairs)


def minimal_voice_leading(chord_a: list[int], chord_b: list[int]) -> list[tuple[int, int]]:
    a = sorted(chord_a)
    b = sorted(chord_b)
    if len(a) <= len(b):
        short, long = a, b
    else:
        short, long = b, a
    pairs = []
    used = set()
    for note in short:
        best_idx, best_cost = 0, 999
        for j, target in enumerate(long):
            if j not in used:
                cost = abs(target - note)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = j
        used.add(best_idx)
        if len(a) <= len(b):
            pairs.append((note, long[best_idx]))
        else:
            pairs.append((long[best_idx], note))
    return pairs


def _dissonance_score(chord: list[int]) -> float:
    if len(chord) < 2:
        return 0.0
    from models.music_cognition.consonance import IntervalConsonance
    ic = IntervalConsonance()
    return 1.0 - ic.chord_consonance([float(p) for p in chord])
