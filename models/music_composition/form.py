"""Musical form: song structures, variation techniques, formal templates."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SongForm:
    name: str
    sections: list[str]

    @classmethod
    def verse_chorus(cls) -> SongForm:
        return cls("verse-chorus", ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"])

    @classmethod
    def aaba(cls) -> SongForm:
        return cls("AABA", ["A", "A", "B", "A"])

    @classmethod
    def rondo(cls) -> SongForm:
        return cls("rondo", ["A", "B", "A", "C", "A"])

    @classmethod
    def blues_12bar(cls) -> SongForm:
        return cls("12-bar blues", ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"])

    @classmethod
    def sonata(cls) -> SongForm:
        return cls("sonata", ["exposition", "development", "recapitulation", "coda"])

    def section_count(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.sections:
            counts[s] = counts.get(s, 0) + 1
        return counts


class FormGenerator:
    TEMPLATES: dict[str, list[str]] = {
        "pop": ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"],
        "aaba": ["A", "A", "B", "A"],
        "rondo": ["A", "B", "A", "C", "A"],
        "binary": ["A", "B"],
        "ternary": ["A", "B", "A"],
        "strophic": ["A", "A", "A", "A"],
        "through-composed": ["A", "B", "C", "D"],
    }

    def generate(self, template: str = "pop", repeats: int = 1) -> list[str]:
        base = self.TEMPLATES.get(template, self.TEMPLATES["pop"])
        return base * repeats


class VariationEngine:
    def transpose(self, pitches: list[int], semitones: int) -> list[int]:
        return [p + semitones for p in pitches]

    def invert(self, pitches: list[int], axis: int | None = None) -> list[int]:
        if not pitches:
            return []
        if axis is None:
            axis = pitches[0]
        return [2 * axis - p for p in pitches]

    def retrograde(self, pitches: list[int]) -> list[int]:
        return list(reversed(pitches))

    def retrograde_inversion(self, pitches: list[int], axis: int | None = None) -> list[int]:
        return self.retrograde(self.invert(pitches, axis))

    def augmentation(self, durations: list[float], factor: float = 2.0) -> list[float]:
        return [d * factor for d in durations]

    def diminution(self, durations: list[float], factor: float = 2.0) -> list[float]:
        return [d / factor for d in durations]

    def sequence(self, pitches: list[int], interval: int, repetitions: int = 3) -> list[int]:
        result = []
        for r in range(repetitions):
            result.extend([p + interval * r for p in pitches])
        return result


def transpose(pitches: list[int], semitones: int) -> list[int]:
    return VariationEngine().transpose(pitches, semitones)


def invert(pitches: list[int], axis: int) -> list[int]:
    return VariationEngine().invert(pitches, axis)


def retrograde(pitches: list[int]) -> list[int]:
    return VariationEngine().retrograde(pitches)
