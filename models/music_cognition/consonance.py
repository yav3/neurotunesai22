"""Psychoacoustic consonance/dissonance: roughness, interval rating, chord scoring."""
from __future__ import annotations
import numpy as np

INTERVAL_CONSONANCE = {
    0: 1.00,   # unison
    1: 0.30,   # minor 2nd
    2: 0.45,   # major 2nd
    3: 0.75,   # minor 3rd
    4: 0.78,   # major 3rd
    5: 0.85,   # perfect 4th
    6: 0.20,   # tritone
    7: 0.90,   # perfect 5th
    8: 0.70,   # minor 6th
    9: 0.72,   # major 6th
    10: 0.40,  # minor 7th
    11: 0.50,  # major 7th
    12: 0.95,  # octave
}


class RoughnessModel:
    def roughness(self, f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
        if f1 <= 0 or f2 <= 0:
            return 0.0
        if f1 > f2:
            f1, f2 = f2, f1
            a1, a2 = a2, a1
        s = 0.24 * (f1 * 0.021 + 19)
        x = abs(f2 - f1) / s if s > 1e-12 else 0.0
        r = a1 * a2 * (np.exp(-3.5 * x) - np.exp(-5.75 * x))
        return float(max(0.0, r))

    def chord_roughness(self, freqs: list[float], amps: list[float] | None = None) -> float:
        n = len(freqs)
        if n < 2:
            return 0.0
        if amps is None:
            amps = [1.0] * n
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += self.roughness(freqs[i], freqs[j], amps[i], amps[j])
        return total


class IntervalConsonance:
    def rate(self, semitones: int) -> float:
        semitones = abs(semitones) % 13
        return INTERVAL_CONSONANCE.get(semitones, 0.5)

    def chord_consonance(self, pitches: list[float]) -> float:
        if len(pitches) < 2:
            return 1.0
        scores = []
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                interval = abs(int(round(pitches[j] - pitches[i]))) % 13
                scores.append(self.rate(interval))
        return float(np.mean(scores))


def roughness(f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    return RoughnessModel().roughness(f1, f2, a1, a2)


def interval_consonance(semitones: int) -> float:
    return IntervalConsonance().rate(semitones)


def chord_consonance(pitches: list[float]) -> float:
    return IntervalConsonance().chord_consonance(pitches)
