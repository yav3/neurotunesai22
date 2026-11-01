"""Arrangement and orchestration: instrument roles, dynamics, part distribution."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class InstrumentRange:
    name: str = ""
    midi_program: int = 0
    min_note: int = 21
    max_note: int = 108
    typical_role: str = "melody"

    def contains(self, note: int) -> bool:
        return self.min_note <= note <= self.max_note

    def clamp(self, note: int) -> int:
        while note < self.min_note:
            note += 12
        while note > self.max_note:
            note -= 12
        return max(self.min_note, min(note, self.max_note))


STANDARD_INSTRUMENTS = {
    "piano": InstrumentRange("Piano", 0, 21, 108, "melody"),
    "bass": InstrumentRange("Bass", 33, 28, 60, "bass"),
    "strings": InstrumentRange("Strings", 48, 36, 96, "pad"),
    "flute": InstrumentRange("Flute", 73, 60, 96, "melody"),
    "guitar": InstrumentRange("Guitar", 25, 40, 84, "harmony"),
    "drums": InstrumentRange("Drums", 0, 35, 81, "rhythm"),
    "synth_lead": InstrumentRange("Synth Lead", 80, 48, 96, "melody"),
    "synth_pad": InstrumentRange("Synth Pad", 89, 36, 84, "pad"),
}


@dataclass
class ArrangementTemplate:
    name: str = ""
    roles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def pop_band(cls) -> ArrangementTemplate:
        return cls("Pop Band", {
            "synth_lead": "melody", "guitar": "harmony",
            "bass": "bass", "drums": "rhythm", "synth_pad": "pad",
        })

    @classmethod
    def orchestra(cls) -> ArrangementTemplate:
        return cls("Orchestra", {
            "flute": "melody", "strings": "harmony",
            "bass": "bass", "piano": "pad",
        })

    @classmethod
    def electronic(cls) -> ArrangementTemplate:
        return cls("Electronic", {
            "synth_lead": "melody", "synth_pad": "pad",
            "bass": "bass", "drums": "rhythm",
        })


class ArrangementEngine:
    def assign_parts(
        self,
        melody: list[int],
        chord_roots: list[int],
        template: ArrangementTemplate,
    ) -> dict[str, list[int]]:
        parts: dict[str, list[int]] = {}
        for inst_key, role in template.roles.items():
            inst = STANDARD_INSTRUMENTS.get(inst_key)
            if inst is None:
                continue
            if role == "melody":
                parts[inst_key] = [inst.clamp(n) for n in melody]
            elif role == "bass":
                parts[inst_key] = [inst.clamp(r + 36) for r in chord_roots]
            elif role == "harmony":
                parts[inst_key] = [inst.clamp(r + 60 + 4) for r in chord_roots]
            elif role == "pad":
                parts[inst_key] = [inst.clamp(r + 60 + 7) for r in chord_roots]
            elif role == "rhythm":
                parts[inst_key] = [36] * max(len(melody), len(chord_roots))
        return parts


class DynamicsEngine:
    @staticmethod
    def crescendo(n_steps: int, start: float = 0.3, end: float = 1.0) -> np.ndarray:
        return np.linspace(start, end, n_steps, dtype=np.float32)

    @staticmethod
    def diminuendo(n_steps: int, start: float = 1.0, end: float = 0.3) -> np.ndarray:
        return np.linspace(start, end, n_steps, dtype=np.float32)

    @staticmethod
    def sforzando(n_steps: int, accent_pos: int = 0, accent_level: float = 1.0, base: float = 0.5) -> np.ndarray:
        curve = np.full(n_steps, base, dtype=np.float32)
        if 0 <= accent_pos < n_steps:
            curve[accent_pos] = accent_level
        return curve

    @staticmethod
    def apply_dynamics(velocities: list[int], curve: np.ndarray) -> list[int]:
        n = min(len(velocities), len(curve))
        return [max(1, min(127, int(velocities[i] * curve[i]))) for i in range(n)]
