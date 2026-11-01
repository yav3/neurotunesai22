"""Data classes for lyrics and poetry records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LyricRecord:
    id: str = ""
    text: str = ""
    language: str = "en"
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    source: str = ""
    licence: str = ""

    vad_valence: float = 0.0
    vad_arousal: float = 0.0
    vad_dominance: float = 0.0

    line_count: int = 0
    word_count: int = 0
    syllable_count_est: int = 0
    rhyme_scheme: str = ""
    detected_meter: str = ""

    sections: list[dict[str, Any]] = field(default_factory=list)
    timestamps: list[tuple[float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.text and not self.line_count:
            self.line_count = len([l for l in self.text.split("\n") if l.strip()])
        if self.text and not self.word_count:
            self.word_count = len(self.text.split())


@dataclass
class PoemRecord:
    id: str = ""
    text: str = ""
    language: str = "en"
    title: str = ""
    author: str = ""
    source: str = ""
    licence: str = ""

    form_type: str = ""
    era: str = ""
    tradition: str = ""

    vad_valence: float = 0.0
    vad_arousal: float = 0.0
    vad_dominance: float = 0.0

    line_count: int = 0
    word_count: int = 0
    syllable_count_est: int = 0
    rhyme_scheme: str = ""
    detected_meter: str = ""
    stanza_count: int = 0

    def __post_init__(self) -> None:
        if self.text and not self.line_count:
            self.line_count = len([l for l in self.text.split("\n") if l.strip()])
        if self.text and not self.word_count:
            self.word_count = len(self.text.split())
        if self.text and not self.stanza_count:
            stanzas = [s for s in self.text.split("\n\n") if s.strip()]
            self.stanza_count = len(stanzas) if stanzas else 1


@dataclass
class CorpusStats:
    total_records: int = 0
    by_language: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_genre: dict[str, int] = field(default_factory=dict)
    total_words: int = 0
    total_lines: int = 0
    languages_count: int = 0

    def summary(self) -> str:
        return (
            f"{self.total_records} records, {self.languages_count} languages, "
            f"{self.total_words} words, {self.total_lines} lines"
        )
