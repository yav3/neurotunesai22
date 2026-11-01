"""Lyric structure analysis: section detection, hook density, syllable alignment."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class SectionType(Enum):
    VERSE = "verse"
    CHORUS = "chorus"
    PRE_CHORUS = "pre_chorus"
    BRIDGE = "bridge"
    INTRO = "intro"
    OUTRO = "outro"
    HOOK = "hook"
    REFRAIN = "refrain"
    INTERLUDE = "interlude"
    SPOKEN = "spoken"
    AD_LIB = "ad_lib"
    UNKNOWN = "unknown"


SECTION_MARKERS: dict[str, SectionType] = {
    "verse": SectionType.VERSE,
    "chorus": SectionType.CHORUS,
    "pre-chorus": SectionType.PRE_CHORUS,
    "prechorus": SectionType.PRE_CHORUS,
    "bridge": SectionType.BRIDGE,
    "intro": SectionType.INTRO,
    "outro": SectionType.OUTRO,
    "hook": SectionType.HOOK,
    "refrain": SectionType.REFRAIN,
    "interlude": SectionType.INTERLUDE,
    "spoken": SectionType.SPOKEN,
    "ad lib": SectionType.AD_LIB,
    "ad-lib": SectionType.AD_LIB,
    "coda": SectionType.OUTRO,
    "tag": SectionType.OUTRO,
    "vamp": SectionType.INTERLUDE,
}

_BRACKET_RE = re.compile(r"^\s*[\[\(]\s*(.+?)\s*[\]\)]\s*$", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"^(verse|chorus|bridge)\s*\d+", re.IGNORECASE)


@dataclass
class LyricSection:
    section_type: SectionType
    label: str
    start_line: int
    end_line: int
    lines: list[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def word_count(self) -> int:
        return sum(len(line.split()) for line in self.lines)


@dataclass
class LyricStructure:
    sections: list[LyricSection]
    total_lines: int
    total_words: int

    @property
    def section_sequence(self) -> list[SectionType]:
        return [s.section_type for s in self.sections]

    @property
    def unique_sections(self) -> set[SectionType]:
        return set(self.section_sequence)

    def sections_of_type(self, st: SectionType) -> list[LyricSection]:
        return [s for s in self.sections if s.section_type == st]

    @property
    def chorus_ratio(self) -> float:
        chorus_lines = sum(s.line_count for s in self.sections if s.section_type == SectionType.CHORUS)
        return chorus_lines / max(self.total_lines, 1)

    @property
    def form_string(self) -> str:
        mapping = {
            SectionType.VERSE: "V",
            SectionType.CHORUS: "C",
            SectionType.PRE_CHORUS: "P",
            SectionType.BRIDGE: "B",
            SectionType.INTRO: "I",
            SectionType.OUTRO: "O",
            SectionType.HOOK: "H",
            SectionType.REFRAIN: "R",
            SectionType.INTERLUDE: "X",
        }
        return "-".join(mapping.get(s.section_type, "?") for s in self.sections)


def _classify_marker(text: str) -> tuple[SectionType, str]:
    text_lower = text.lower().strip()
    numbered = _NUMBERED_RE.match(text_lower)
    if numbered:
        base = numbered.group(1).lower()
        if base in SECTION_MARKERS:
            return SECTION_MARKERS[base], text.strip()
    for marker, stype in SECTION_MARKERS.items():
        if text_lower.startswith(marker):
            return stype, text.strip()
    return SectionType.UNKNOWN, text.strip()


def detect_sections(lyrics: str) -> LyricStructure:
    raw_lines = lyrics.split("\n")
    sections: list[LyricSection] = []
    current_type = SectionType.VERSE
    current_label = "Verse"
    current_start = 0
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines, current_start
        content = [l for l in current_lines if l.strip()]
        if content:
            sections.append(LyricSection(
                section_type=current_type,
                label=current_label,
                start_line=current_start,
                end_line=current_start + len(current_lines),
                lines=content,
            ))
        current_lines = []

    for i, line in enumerate(raw_lines):
        m = _BRACKET_RE.match(line)
        if m:
            flush()
            stype, label = _classify_marker(m.group(1))
            current_type = stype
            current_label = label
            current_start = i + 1
            continue

        if not line.strip() and current_lines:
            stripped = [l for l in current_lines if l.strip()]
            if len(stripped) >= 2:
                flush()
                current_start = i + 1
            continue

        current_lines.append(line)

    flush()

    total_lines = sum(s.line_count for s in sections)
    total_words = sum(s.word_count for s in sections)
    return LyricStructure(sections=sections, total_lines=total_lines, total_words=total_words)


def _normalize_line(line: str) -> str:
    return re.sub(r"[^\w\s]", "", line.lower().strip())


def find_repeated_lines(lyrics: str, min_repeats: int = 2) -> dict[str, int]:
    lines = [_normalize_line(l) for l in lyrics.split("\n") if l.strip()]
    counts: dict[str, int] = {}
    for line in lines:
        if line:
            counts[line] = counts.get(line, 0) + 1
    return {k: v for k, v in counts.items() if v >= min_repeats}


def hook_density(structure: LyricStructure) -> float:
    repeated = find_repeated_lines(
        "\n".join(line for s in structure.sections for line in s.lines)
    )
    repeated_line_count = sum(v for v in repeated.values())
    return repeated_line_count / max(structure.total_lines, 1)


@dataclass
class SyllableAlignment:
    line_syllables: list[int]
    target_syllables: list[int | None]
    deviations: list[int | None]
    alignment_score: float


def compute_syllable_alignment(
    line_syllable_counts: list[int],
    target: list[int | None],
) -> SyllableAlignment:
    deviations: list[int | None] = []
    matched = 0
    total = 0
    for i, count in enumerate(line_syllable_counts):
        if i < len(target) and target[i] is not None:
            dev = count - target[i]
            deviations.append(dev)
            total += 1
            if abs(dev) <= 1:
                matched += 1
        else:
            deviations.append(None)
    score = matched / max(total, 1)
    return SyllableAlignment(
        line_syllables=line_syllable_counts,
        target_syllables=target,
        deviations=deviations,
        alignment_score=score,
    )


def classify_song_form(structure: LyricStructure) -> str:
    seq = structure.section_sequence
    if not seq:
        return "empty"
    types_only = [s for s in seq if s in (SectionType.VERSE, SectionType.CHORUS, SectionType.BRIDGE)]
    if not types_only:
        return "through-composed"

    v_count = types_only.count(SectionType.VERSE)
    c_count = types_only.count(SectionType.CHORUS)
    b_count = types_only.count(SectionType.BRIDGE)

    if c_count == 0:
        if v_count >= 3:
            return "strophic (AAA)"
        return "through-composed"
    if b_count == 0 and c_count >= 2:
        return "verse-chorus (VCVC)"
    if b_count >= 1 and c_count >= 2:
        return "verse-chorus-bridge (ABABCB)"
    if c_count == 1:
        return "AABA (32-bar)"
    return "compound"


COMMON_FORMS = {
    "strophic": "AAA — same melody, different lyrics each verse (folk, hymns)",
    "verse-chorus": "VCVC — alternating verse and chorus (pop, rock)",
    "AABA": "32-bar form — two A sections, bridge (B), return to A (Tin Pan Alley, jazz standards)",
    "verse-chorus-bridge": "ABABCB — verse-chorus with contrasting bridge (modern pop)",
    "through-composed": "No repeating sections (art song, some rap)",
    "12-bar-blues": "AAB × N — statement, repetition, resolution (blues, early rock)",
    "rondo": "ABACADA — recurring theme with episodes (classical adaptation)",
}
