"""Melodic generation: Markov chains, contour-based, mutation operators."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class MelodyConstraints:
    scale_degrees: list[int] = field(default_factory=lambda: [0, 2, 4, 5, 7, 9, 11])
    root_midi: int = 60
    octave_range: tuple[int, int] = (0, 2)
    note_duration_range: tuple[float, float] = (0.125, 1.0)
    rest_probability: float = 0.1
    max_interval: int = 7


class MarkovMelodyGenerator:
    def __init__(self, order: int = 1):
        self._order = order
        self._transitions: dict[tuple[int, ...], dict[int, int]] = {}

    def train(self, sequences: list[list[int]]) -> None:
        self._transitions.clear()
        for seq in sequences:
            for i in range(len(seq) - self._order):
                state = tuple(seq[i:i + self._order])
                nxt = seq[i + self._order]
                if state not in self._transitions:
                    self._transitions[state] = {}
                self._transitions[state][nxt] = self._transitions[state].get(nxt, 0) + 1

    def generate(
        self,
        n_notes: int,
        constraints: MelodyConstraints | None = None,
        seed: int | None = None,
    ) -> list[int]:
        rng = np.random.default_rng(seed)
        constraints = constraints or MelodyConstraints()

        all_notes = []
        for oct in range(constraints.octave_range[0], constraints.octave_range[1] + 1):
            for deg in constraints.scale_degrees:
                all_notes.append(constraints.root_midi + oct * 12 + deg)

        if not self._transitions:
            return [rng.choice(all_notes) for _ in range(n_notes)]

        states = list(self._transitions.keys())
        state = states[rng.integers(len(states))]
        melody = list(state)

        while len(melody) < n_notes:
            state = tuple(melody[-self._order:])
            if state in self._transitions:
                nexts = self._transitions[state]
                notes = list(nexts.keys())
                weights = np.array(list(nexts.values()), dtype=float)
                weights /= weights.sum()
                melody.append(rng.choice(notes, p=weights))
            else:
                melody.append(rng.choice(all_notes))

        return melody[:n_notes]

    @property
    def transition_count(self) -> int:
        return sum(sum(v.values()) for v in self._transitions.values())


class ContourMelodyGenerator:
    CONTOURS = ("ascending", "descending", "arch", "valley", "flat")

    def generate(
        self,
        contour: str,
        n_notes: int,
        constraints: MelodyConstraints | None = None,
        seed: int | None = None,
    ) -> list[int]:
        rng = np.random.default_rng(seed)
        constraints = constraints or MelodyConstraints()

        all_notes = sorted(set(
            constraints.root_midi + oct * 12 + deg
            for oct in range(constraints.octave_range[0], constraints.octave_range[1] + 1)
            for deg in constraints.scale_degrees
        ))

        t = np.linspace(0, 1, n_notes)
        if contour == "ascending":
            curve = t
        elif contour == "descending":
            curve = 1.0 - t
        elif contour == "arch":
            curve = np.sin(np.pi * t)
        elif contour == "valley":
            curve = 1.0 - np.sin(np.pi * t)
        else:
            curve = np.full(n_notes, 0.5)

        melody = []
        for v in curve:
            idx = int(v * (len(all_notes) - 1))
            idx = max(0, min(idx, len(all_notes) - 1))
            jitter = rng.integers(-1, 2)
            idx = max(0, min(idx + jitter, len(all_notes) - 1))
            melody.append(all_notes[idx])
        return melody


class MelodyMutator:
    @staticmethod
    def transpose(melody: list[int], semitones: int) -> list[int]:
        return [n + semitones for n in melody]

    @staticmethod
    def invert(melody: list[int], axis: int | None = None) -> list[int]:
        if not melody:
            return []
        ax = axis if axis is not None else melody[0]
        return [2 * ax - n for n in melody]

    @staticmethod
    def retrograde(melody: list[int]) -> list[int]:
        return list(reversed(melody))

    @staticmethod
    def augment(melody: list[int], factor: float = 2.0) -> list[int]:
        if not melody:
            return []
        ref = melody[0]
        return [ref + int((n - ref) * factor) for n in melody]

    @staticmethod
    def diminish(melody: list[int], factor: float = 0.5) -> list[int]:
        return MelodyMutator.augment(melody, factor)
