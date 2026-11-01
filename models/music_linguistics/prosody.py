"""Prosodic analysis: meter, stress, rhyme, enjambment, caesura."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Foot(Enum):
    IAMB = "iamb"
    TROCHEE = "trochee"
    DACTYL = "dactyl"
    ANAPEST = "anapest"
    SPONDEE = "spondee"
    PYRRHIC = "pyrrhic"


class RhymeType(Enum):
    PERFECT = "perfect"
    SLANT = "slant"
    EYE = "eye"
    NONE = "none"


@dataclass
class MeterResult:
    feet: list[Foot]
    pattern: str
    regularity: float


@dataclass
class RhymeSchemeResult:
    scheme: str
    pairs: list[tuple[int, int]]
    rhyme_types: dict[tuple[int, int], RhymeType]


def stress_pattern(syllable_stresses: list[bool]) -> str:
    return "".join("/" if s else "u" for s in syllable_stresses)


def classify_feet(stresses: list[bool]) -> list[Foot]:
    feet = []
    i = 0
    while i < len(stresses):
        if i + 2 < len(stresses) and stresses[i] and not stresses[i + 1] and not stresses[i + 2]:
            feet.append(Foot.DACTYL)
            i += 3
        elif i + 2 < len(stresses) and not stresses[i] and not stresses[i + 1] and stresses[i + 2]:
            feet.append(Foot.ANAPEST)
            i += 3
        elif i + 1 < len(stresses):
            if not stresses[i] and stresses[i + 1]:
                feet.append(Foot.IAMB)
            elif stresses[i] and not stresses[i + 1]:
                feet.append(Foot.TROCHEE)
            elif stresses[i] and stresses[i + 1]:
                feet.append(Foot.SPONDEE)
            else:
                feet.append(Foot.PYRRHIC)
            i += 2
        else:
            i += 1
    return feet


def analyze_meter(stresses: list[bool]) -> MeterResult:
    feet = classify_feet(stresses)
    if not feet:
        return MeterResult([], "", 0.0)
    pattern = stress_pattern(stresses)
    from collections import Counter
    counts = Counter(f.value for f in feet)
    dominant_count = counts.most_common(1)[0][1] if counts else 0
    regularity = dominant_count / len(feet) if feet else 0.0
    return MeterResult(feet=feet, pattern=pattern, regularity=regularity)


def _ending_phonemes(word: str, n: int = 3) -> str:
    return word.lower().strip()[-n:] if len(word) >= n else word.lower().strip()


def classify_rhyme(word_a: str, word_b: str) -> RhymeType:
    a = word_a.lower().strip()
    b = word_b.lower().strip()
    if a == b:
        return RhymeType.PERFECT
    if _ending_phonemes(a, 3) == _ending_phonemes(b, 3):
        return RhymeType.PERFECT
    if _ending_phonemes(a, 2) == _ending_phonemes(b, 2):
        return RhymeType.PERFECT
    vowels = set("aeiou")
    a_vowels = "".join(c for c in a[-3:] if c in vowels)
    b_vowels = "".join(c for c in b[-3:] if c in vowels)
    if a_vowels and a_vowels == b_vowels and a[-1] != b[-1]:
        return RhymeType.SLANT
    a_cons = "".join(c for c in a[-3:] if c not in vowels)
    b_cons = "".join(c for c in b[-3:] if c not in vowels)
    if a_cons and a_cons == b_cons and a_vowels != b_vowels:
        return RhymeType.SLANT
    if a[-2:] == b[-2:] and a != b:
        return RhymeType.EYE
    return RhymeType.NONE


def detect_rhyme_scheme(line_endings: list[str], max_distance: int = 8) -> RhymeSchemeResult:
    n = len(line_endings)
    labels = [""] * n
    pairs: list[tuple[int, int]] = []
    types: dict[tuple[int, int], RhymeType] = {}
    next_label = 0

    for i in range(n):
        if labels[i]:
            continue
        label_char = chr(ord("A") + next_label % 26)
        labels[i] = label_char
        found = False
        for j in range(i + 1, min(i + max_distance + 1, n)):
            if labels[j]:
                continue
            rt = classify_rhyme(line_endings[i], line_endings[j])
            if rt in (RhymeType.PERFECT, RhymeType.SLANT):
                labels[j] = label_char
                pairs.append((i, j))
                types[(i, j)] = rt
                found = True
        if not found:
            pass
        next_label += 1

    return RhymeSchemeResult(
        scheme="".join(labels),
        pairs=pairs,
        rhyme_types=types,
    )


def detect_enjambment(lines: list[str]) -> list[int]:
    enjambed = []
    non_terminal = set(",:-–—")
    conjunctions = {"and", "or", "but", "nor", "yet", "so", "for", "the", "a", "an",
                    "to", "of", "in", "on", "at", "by", "with", "from"}
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        if not line:
            continue
        if line[-1] in non_terminal:
            enjambed.append(i)
        elif i + 1 < len(lines):
            next_first = lines[i + 1].strip().split()[0].lower() if lines[i + 1].strip() else ""
            if next_first in conjunctions:
                enjambed.append(i)
    return enjambed


def detect_caesura(line: str) -> list[int]:
    positions = []
    mid = len(line) // 2
    for i, ch in enumerate(line):
        if ch in (",", ";", ":", "—", "–", "||"):
            if abs(i - mid) < len(line) // 3:
                positions.append(i)
    return positions
