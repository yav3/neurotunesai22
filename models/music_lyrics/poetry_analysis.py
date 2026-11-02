"""Poetry and lyric structural analysis: meter, rhyme scheme, stanza form."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MeterType(Enum):
    IAMBIC = "iambic"
    TROCHAIC = "trochaic"
    ANAPESTIC = "anapestic"
    DACTYLIC = "dactylic"
    SPONDAIC = "spondaic"
    FREE = "free"


class StanzaForm(Enum):
    COUPLET = "couplet"
    TERCET = "tercet"
    QUATRAIN = "quatrain"
    SESTET = "sestet"
    OCTAVE = "octave"
    SONNET = "sonnet"
    HAIKU = "haiku"
    LIMERICK = "limerick"
    VILLANELLE = "villanelle"
    GHAZAL = "ghazal"
    TANKA = "tanka"
    FREE_VERSE = "free_verse"


@dataclass
class RhymeAnalysis:
    scheme: str
    rhyme_pairs: list[tuple[int, int]]
    end_words: list[str]
    rhyme_density: float


@dataclass
class MeterAnalysis:
    meter_type: MeterType
    feet_per_line: list[int]
    regularity: float
    stress_pattern: list[list[int]]


@dataclass
class PoetryAnalysis:
    n_lines: int
    n_stanzas: int
    stanza_form: StanzaForm
    rhyme: RhymeAnalysis
    meter: MeterAnalysis
    syllable_counts: list[int]
    avg_line_length: float
    vocabulary_richness: float


VOWEL_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)


def estimate_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    vowel_groups = VOWEL_PATTERN.findall(word)
    count = len(vowel_groups)
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1
    return max(1, count)


def syllable_count_line(line: str) -> int:
    words = re.findall(r"[a-zA-Z']+", line)
    return sum(estimate_syllables(w) for w in words)


def end_sound(word: str, n_chars: int = 3) -> str:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    return cleaned[-n_chars:] if len(cleaned) >= n_chars else cleaned


def detect_rhyme_scheme(lines: list[str], tolerance: int = 2) -> RhymeAnalysis:
    end_words = []
    for line in lines:
        words = re.findall(r"[a-zA-Z']+", line)
        end_words.append(words[-1].lower() if words else "")
    sounds = [end_sound(w) for w in end_words]
    labels: list[str] = []
    sound_to_label: dict[str, str] = {}
    label_idx = 0
    for s in sounds:
        if s in sound_to_label:
            labels.append(sound_to_label[s])
        else:
            lbl = chr(ord("A") + label_idx % 26)
            sound_to_label[s] = lbl
            labels.append(lbl)
            label_idx += 1
    scheme = "".join(labels)
    pairs = []
    for i in range(len(sounds)):
        for j in range(i + 1, len(sounds)):
            if sounds[i] == sounds[j] and sounds[i]:
                pairs.append((i, j))
    n = len(lines)
    max_pairs = n * (n - 1) / 2 if n > 1 else 1
    density = len(pairs) / max_pairs if max_pairs > 0 else 0.0
    return RhymeAnalysis(scheme=scheme, rhyme_pairs=pairs, end_words=end_words, rhyme_density=density)


def estimate_stress(word: str) -> list[int]:
    syls = estimate_syllables(word)
    if syls <= 1:
        return [1]
    pattern = []
    for i in range(syls):
        pattern.append(1 if i % 2 == (0 if syls % 2 == 1 else 1) else 0)
    return pattern


def detect_meter(lines: list[str]) -> MeterAnalysis:
    stress_patterns: list[list[int]] = []
    feet_per_line: list[int] = []
    for line in lines:
        words = re.findall(r"[a-zA-Z']+", line)
        stresses: list[int] = []
        for w in words:
            stresses.extend(estimate_stress(w))
        stress_patterns.append(stresses)
        feet_per_line.append(max(1, len(stresses) // 2))
    iambic_score = 0
    trochaic_score = 0
    total = 0
    for pattern in stress_patterns:
        for i in range(0, len(pattern) - 1, 2):
            total += 1
            if i + 1 < len(pattern):
                if pattern[i] == 0 and pattern[i + 1] == 1:
                    iambic_score += 1
                elif pattern[i] == 1 and pattern[i + 1] == 0:
                    trochaic_score += 1
    if total == 0:
        return MeterAnalysis(MeterType.FREE, feet_per_line, 0.0, stress_patterns)
    best = max(iambic_score, trochaic_score)
    regularity = best / total
    if regularity < 0.4:
        meter = MeterType.FREE
    elif iambic_score >= trochaic_score:
        meter = MeterType.IAMBIC
    else:
        meter = MeterType.TROCHAIC
    return MeterAnalysis(meter, feet_per_line, regularity, stress_patterns)


def classify_stanza_form(lines: list[str]) -> StanzaForm:
    stanzas = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        else:
            if current:
                stanzas.append(current)
                current = []
    if current:
        stanzas.append(current)
    n_lines = sum(len(s) for s in stanzas)
    if n_lines == 3:
        syls = [syllable_count_line(l) for l in lines if l.strip()]
        if len(syls) == 3 and abs(syls[0] - 5) <= 1 and abs(syls[1] - 7) <= 1 and abs(syls[2] - 5) <= 1:
            return StanzaForm.HAIKU
        return StanzaForm.TERCET
    if n_lines == 5:
        syls = [syllable_count_line(l) for l in lines if l.strip()]
        if len(syls) == 5 and abs(syls[0] - 5) <= 2 and abs(syls[1] - 7) <= 2 and abs(syls[2] - 5) <= 2:
            return StanzaForm.TANKA
        return StanzaForm.LIMERICK
    if n_lines == 14:
        return StanzaForm.SONNET
    if all(len(s) == 2 for s in stanzas):
        return StanzaForm.COUPLET
    if all(len(s) == 4 for s in stanzas):
        return StanzaForm.QUATRAIN
    return StanzaForm.FREE_VERSE


def vocabulary_richness(text: str) -> float:
    words = [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def analyze_poetry(text: str) -> PoetryAnalysis:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        lines = [""]
    stanzas = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            current.append(line)
        else:
            if current:
                stanzas.append(current)
                current = []
    if current:
        stanzas.append(current)
    rhyme = detect_rhyme_scheme(lines)
    meter = detect_meter(lines)
    form = classify_stanza_form(text.split("\n"))
    syl_counts = [syllable_count_line(l) for l in lines]
    avg_len = np.mean([len(l) for l in lines]) if lines else 0.0
    vr = vocabulary_richness(text)
    return PoetryAnalysis(
        n_lines=len(lines), n_stanzas=len(stanzas), stanza_form=form,
        rhyme=rhyme, meter=meter, syllable_counts=syl_counts,
        avg_line_length=float(avg_len), vocabulary_richness=vr,
    )
