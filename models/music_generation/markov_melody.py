"""Markov chain melody generation: n-th order pitch/rhythm chains.

Trains transition tables from observed melodic sequences and generates
new melodies via stochastic sampling with musical constraints (range,
scale membership, phrase structure). Supports interval-based training
for transposition invariance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Note:
    pitch: int
    duration: float
    onset: float = 0.0


class MarkovChain:
    """Generic n-th order Markov chain over discrete states."""

    def __init__(self, order: int = 2) -> None:
        self.order = order
        self._counts: dict[tuple, dict[int, int]] = {}
        self._states: set[int] = set()

    def train(self, sequences: list[list[int]]) -> None:
        for seq in sequences:
            for s in seq:
                self._states.add(s)
            for i in range(self.order, len(seq)):
                context = tuple(seq[i - self.order : i])
                nxt = seq[i]
                if context not in self._counts:
                    self._counts[context] = {}
                self._counts[context][nxt] = self._counts[context].get(nxt, 0) + 1

    def transition_probs(self, context: tuple) -> dict[int, float]:
        counts = self._counts.get(context, {})
        total = sum(counts.values())
        if total == 0:
            n = len(self._states)
            return {s: 1.0 / n for s in self._states} if n > 0 else {}
        return {s: c / total for s, c in counts.items()}

    def generate(self, length: int, seed: list[int] | None = None,
                 temperature: float = 1.0, rng: np.random.RandomState | None = None) -> list[int]:
        if rng is None:
            rng = np.random.RandomState()
        if seed is None or len(seed) < self.order:
            states = sorted(self._states)
            if not states:
                return []
            seed = [states[rng.randint(len(states))] for _ in range(self.order)]
        result = list(seed)
        for _ in range(length - len(result)):
            context = tuple(result[-self.order :])
            probs = self.transition_probs(context)
            if not probs:
                break
            keys = list(probs.keys())
            vals = np.array([probs[k] for k in keys], dtype=np.float64)
            if temperature != 1.0:
                log_vals = np.log(vals + 1e-12) / max(temperature, 1e-6)
                vals = np.exp(log_vals - np.max(log_vals))
            vals /= vals.sum()
            idx = rng.choice(len(keys), p=vals)
            result.append(keys[idx])
        return result

    def log_likelihood(self, sequence: list[int]) -> float:
        ll = 0.0
        for i in range(self.order, len(sequence)):
            context = tuple(sequence[i - self.order : i])
            probs = self.transition_probs(context)
            p = probs.get(sequence[i], 1e-12)
            ll += np.log(max(p, 1e-12))
        return float(ll)


class MelodyMarkov(MarkovChain):
    """Pitch-specialized Markov chain with interval-based training."""

    def __init__(self, order: int = 2, use_intervals: bool = True) -> None:
        super().__init__(order)
        self.use_intervals = use_intervals
        self._pitch_range: tuple[int, int] = (48, 84)
        self._scale: set[int] | None = None

    def set_range(self, low: int, high: int) -> None:
        self._pitch_range = (low, high)

    def set_scale(self, scale_pitches: list[int]) -> None:
        self._scale = set(scale_pitches)

    def train_from_melodies(self, melodies: list[list[int]]) -> None:
        if self.use_intervals:
            interval_seqs = []
            for m in melodies:
                intervals = [m[i + 1] - m[i] for i in range(len(m) - 1)]
                interval_seqs.append(intervals)
            self.train(interval_seqs)
        else:
            self.train(melodies)

    def generate_melody(self, length: int, start_pitch: int = 60,
                        temperature: float = 1.0,
                        rng: np.random.RandomState | None = None) -> list[int]:
        if rng is None:
            rng = np.random.RandomState()
        if self.use_intervals:
            intervals = self.generate(length, temperature=temperature, rng=rng)
            pitches = [start_pitch]
            for iv in intervals:
                p = pitches[-1] + iv
                p = max(self._pitch_range[0], min(p, self._pitch_range[1]))
                if self._scale is not None:
                    pc = p % 12
                    if pc not in {s % 12 for s in self._scale}:
                        candidates = [s for s in self._scale if abs(s % 12 - pc) <= 1]
                        if candidates:
                            closest = min(candidates, key=lambda s: abs(s % 12 - pc))
                            p = (p // 12) * 12 + closest % 12
                pitches.append(p)
            return pitches[:length]
        return self.generate(length, temperature=temperature, rng=rng)


class RhythmMarkov(MarkovChain):
    """Duration-specialized Markov chain."""

    DURATIONS = {
        "whole": 4.0, "half": 2.0, "quarter": 1.0,
        "eighth": 0.5, "sixteenth": 0.25, "dotted_quarter": 1.5,
        "dotted_half": 3.0, "triplet_eighth": 1.0 / 3.0,
    }

    def __init__(self, order: int = 2) -> None:
        super().__init__(order)
        self._dur_to_id: dict[float, int] = {}
        self._id_to_dur: dict[int, float] = {}
        for i, (_, v) in enumerate(sorted(self.DURATIONS.items(), key=lambda x: x[1])):
            self._dur_to_id[v] = i
            self._id_to_dur[i] = v

    def _quantize(self, dur: float) -> int:
        best = min(self._dur_to_id.keys(), key=lambda d: abs(d - dur))
        return self._dur_to_id[best]

    def train_from_durations(self, sequences: list[list[float]]) -> None:
        id_seqs = [[self._quantize(d) for d in seq] for seq in sequences]
        self.train(id_seqs)

    def generate_rhythm(self, length: int, temperature: float = 1.0,
                        rng: np.random.RandomState | None = None) -> list[float]:
        ids = self.generate(length, temperature=temperature, rng=rng)
        return [self._id_to_dur.get(i, 1.0) for i in ids]


class CombinedMelodyGenerator:
    """Joint pitch+rhythm generation from independent or coupled chains."""

    def __init__(self, pitch_order: int = 2, rhythm_order: int = 2) -> None:
        self.pitch_chain = MelodyMarkov(pitch_order)
        self.rhythm_chain = RhythmMarkov(rhythm_order)

    def train(self, melodies: list[list[int]], rhythms: list[list[float]]) -> None:
        self.pitch_chain.train_from_melodies(melodies)
        self.rhythm_chain.train_from_durations(rhythms)

    def generate(self, n_notes: int, start_pitch: int = 60,
                 temperature: float = 1.0, rng: np.random.RandomState | None = None
                 ) -> list[Note]:
        if rng is None:
            rng = np.random.RandomState()
        pitches = self.pitch_chain.generate_melody(n_notes, start_pitch, temperature, rng)
        durations = self.rhythm_chain.generate_rhythm(n_notes, temperature, rng)
        notes = []
        onset = 0.0
        for p, d in zip(pitches, durations):
            notes.append(Note(p, d, onset))
            onset += d
        return notes
