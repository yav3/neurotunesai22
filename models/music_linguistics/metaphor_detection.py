"""Figurative language detection in lyrics.

Identifies metaphor, simile, personification, synesthesia, hyperbole.
Pure numpy string/pattern analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class FigureType(Enum):
    METAPHOR = "metaphor"
    SIMILE = "simile"
    PERSONIFICATION = "personification"
    SYNESTHESIA = "synesthesia"
    HYPERBOLE = "hyperbole"
    METONYMY = "metonymy"


@dataclass
class FigureInstance:
    figure_type: FigureType
    text: str
    line_index: int
    confidence: float
    source_domain: str = ""
    target_domain: str = ""


SIMILE_MARKERS = {"like", "as", "as if", "as though", "resembles", "similar to"}

PERSONIFICATION_VERBS = {
    "whisper", "cry", "scream", "dance", "breathe", "sleep", "wake",
    "sing", "laugh", "weep", "embrace", "kiss", "run", "walk",
    "bleed", "die", "dream", "speak", "call",
}

SYNESTHESIA_PAIRS = {
    ("loud", "color"), ("bright", "sound"), ("warm", "voice"),
    ("cold", "melody"), ("sharp", "tone"), ("sweet", "sound"),
    ("dark", "voice"), ("soft", "light"), ("heavy", "silence"),
    ("smooth", "rhythm"),
}

HYPERBOLE_MARKERS = {
    "forever", "never", "always", "infinite", "eternal", "million",
    "billion", "thousand", "entire", "every single", "all of",
    "nothing", "everything", "no one", "everyone",
}

DOMAIN_WORDS: dict[str, list[str]] = {
    "nature": ["rain", "sun", "moon", "star", "ocean", "river", "fire", "wind",
               "storm", "sky", "earth", "flower", "tree", "mountain", "sea"],
    "body": ["heart", "blood", "bone", "skin", "eye", "hand", "breath", "vein"],
    "emotion": ["love", "pain", "fear", "joy", "anger", "sorrow", "hope", "desire"],
    "light": ["shine", "glow", "bright", "dark", "shadow", "light", "flame", "spark"],
    "motion": ["fall", "rise", "fly", "sink", "float", "drift", "crash", "burn"],
    "music": ["song", "melody", "rhythm", "beat", "harmony", "chord", "note"],
}


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def detect_simile(line: str) -> FigureInstance | None:
    tokens = _tokenize(line)
    text_lower = line.lower()
    for marker in SIMILE_MARKERS:
        if marker in text_lower:
            return FigureInstance(
                figure_type=FigureType.SIMILE,
                text=line, line_index=0, confidence=0.7,
            )
    return None


def detect_personification(line: str) -> FigureInstance | None:
    tokens = _tokenize(line)
    non_human_subjects = {"rain", "sun", "moon", "night", "wind", "time",
                          "death", "love", "silence", "darkness", "music",
                          "heart", "sky", "ocean", "world"}
    for i, tok in enumerate(tokens):
        if tok in non_human_subjects and i + 1 < len(tokens):
            if tokens[i + 1] in PERSONIFICATION_VERBS:
                return FigureInstance(
                    figure_type=FigureType.PERSONIFICATION,
                    text=line, line_index=0, confidence=0.65,
                    source_domain="human", target_domain=tok,
                )
    return None


def detect_hyperbole(line: str) -> FigureInstance | None:
    tokens = set(_tokenize(line))
    matches = tokens & HYPERBOLE_MARKERS
    if matches:
        return FigureInstance(
            figure_type=FigureType.HYPERBOLE,
            text=line, line_index=0, confidence=0.5 + 0.1 * len(matches),
        )
    return None


def domain_membership(word: str) -> str:
    for domain, words in DOMAIN_WORDS.items():
        if word in words:
            return domain
    return ""


def detect_cross_domain(line: str) -> FigureInstance | None:
    """Detect metaphor via cross-domain word co-occurrence."""
    tokens = _tokenize(line)
    domains_found: dict[str, list[str]] = {}
    for tok in tokens:
        d = domain_membership(tok)
        if d:
            domains_found.setdefault(d, []).append(tok)
    if len(domains_found) >= 2:
        dom_list = list(domains_found.keys())
        return FigureInstance(
            figure_type=FigureType.METAPHOR,
            text=line, line_index=0, confidence=0.6,
            source_domain=dom_list[0], target_domain=dom_list[1],
        )
    return None


def analyze_lyrics(lines: Sequence[str]) -> list[FigureInstance]:
    """Find all figurative language instances in a set of lyric lines."""
    results: list[FigureInstance] = []
    for i, line in enumerate(lines):
        for detector in (detect_simile, detect_personification, detect_hyperbole, detect_cross_domain):
            fig = detector(line)
            if fig is not None:
                fig.line_index = i
                results.append(fig)
    return results


def figurative_density(lines: Sequence[str]) -> float:
    """Ratio of lines containing at least one figure to total lines."""
    if not lines:
        return 0.0
    figures = analyze_lyrics(lines)
    lines_with_figure = len(set(f.line_index for f in figures))
    return lines_with_figure / len(lines)


def figure_type_distribution(figures: Sequence[FigureInstance]) -> dict[str, float]:
    """Distribution of figure types as proportions."""
    if not figures:
        return {}
    counts: dict[str, int] = {}
    for f in figures:
        counts[f.figure_type.value] = counts.get(f.figure_type.value, 0) + 1
    total = len(figures)
    return {k: v / total for k, v in counts.items()}
