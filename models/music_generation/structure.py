"""Song structure: sections, templates, structure generation, transitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SongSection(Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    SOLO = "solo"
    OUTRO = "outro"
    INTERLUDE = "interlude"


SECTION_ENERGY = {
    SongSection.INTRO: 0.3,
    SongSection.VERSE: 0.5,
    SongSection.PRE_CHORUS: 0.65,
    SongSection.CHORUS: 0.9,
    SongSection.BRIDGE: 0.6,
    SongSection.SOLO: 0.8,
    SongSection.OUTRO: 0.3,
    SongSection.INTERLUDE: 0.4,
}

SECTION_DURATION_BARS = {
    SongSection.INTRO: 4,
    SongSection.VERSE: 8,
    SongSection.PRE_CHORUS: 4,
    SongSection.CHORUS: 8,
    SongSection.BRIDGE: 8,
    SongSection.SOLO: 8,
    SongSection.OUTRO: 4,
    SongSection.INTERLUDE: 4,
}


@dataclass
class SongStructureTemplate:
    name: str
    sections: list[SongSection]

    @classmethod
    def verse_chorus(cls) -> SongStructureTemplate:
        return cls("Verse-Chorus", [
            SongSection.INTRO, SongSection.VERSE, SongSection.CHORUS,
            SongSection.VERSE, SongSection.CHORUS,
            SongSection.BRIDGE, SongSection.CHORUS, SongSection.OUTRO,
        ])

    @classmethod
    def aaba(cls) -> SongStructureTemplate:
        return cls("AABA", [
            SongSection.VERSE, SongSection.VERSE,
            SongSection.BRIDGE, SongSection.VERSE,
        ])

    @classmethod
    def rondo(cls) -> SongStructureTemplate:
        return cls("Rondo", [
            SongSection.CHORUS, SongSection.VERSE,
            SongSection.CHORUS, SongSection.BRIDGE,
            SongSection.CHORUS,
        ])

    @classmethod
    def through_composed(cls) -> SongStructureTemplate:
        return cls("Through-Composed", [
            SongSection.INTRO, SongSection.VERSE, SongSection.BRIDGE,
            SongSection.SOLO, SongSection.OUTRO,
        ])

    def total_bars(self) -> int:
        return sum(SECTION_DURATION_BARS.get(s, 4) for s in self.sections)

    def energy_profile(self) -> list[float]:
        return [SECTION_ENERGY.get(s, 0.5) for s in self.sections]


@dataclass
class SectionTransition:
    from_section: SongSection
    to_section: SongSection
    transition_type: str = "direct"
    key_change: int = 0
    tempo_change: float = 0.0
    crossfade_bars: int = 0
    texture_change: str = "none"

    @classmethod
    def direct(cls, from_s: SongSection, to_s: SongSection) -> SectionTransition:
        return cls(from_s, to_s, "direct")

    @classmethod
    def modulation(cls, from_s: SongSection, to_s: SongSection, semitones: int) -> SectionTransition:
        return cls(from_s, to_s, "modulation", key_change=semitones)

    @classmethod
    def crossfade(cls, from_s: SongSection, to_s: SongSection, bars: int = 2) -> SectionTransition:
        return cls(from_s, to_s, "crossfade", crossfade_bars=bars)


class SongStructureGenerator:
    def generate(
        self,
        target_duration_s: float = 210.0,
        bpm: float = 120.0,
        energy_shape: str = "peak",
        seed: int | None = None,
    ) -> SongStructureTemplate:
        rng = np.random.default_rng(seed)
        bar_duration = 4 * 60.0 / bpm
        target_bars = int(target_duration_s / bar_duration)

        sections: list[SongSection] = [SongSection.INTRO]
        bars_used = SECTION_DURATION_BARS[SongSection.INTRO]

        main_pool = [SongSection.VERSE, SongSection.CHORUS, SongSection.PRE_CHORUS,
                     SongSection.BRIDGE, SongSection.SOLO, SongSection.INTERLUDE]

        while bars_used < target_bars - SECTION_DURATION_BARS[SongSection.OUTRO]:
            s = main_pool[rng.integers(len(main_pool))]
            cost = SECTION_DURATION_BARS.get(s, 4)
            if bars_used + cost > target_bars - SECTION_DURATION_BARS[SongSection.OUTRO]:
                break
            sections.append(s)
            bars_used += cost

        sections.append(SongSection.OUTRO)
        return SongStructureTemplate(f"Generated ({energy_shape})", sections)

    def plan_transitions(self, template: SongStructureTemplate) -> list[SectionTransition]:
        transitions = []
        for i in range(len(template.sections) - 1):
            from_s = template.sections[i]
            to_s = template.sections[i + 1]
            e_from = SECTION_ENERGY.get(from_s, 0.5)
            e_to = SECTION_ENERGY.get(to_s, 0.5)
            if abs(e_to - e_from) > 0.3:
                transitions.append(SectionTransition.crossfade(from_s, to_s, bars=2))
            else:
                transitions.append(SectionTransition.direct(from_s, to_s))
        return transitions
