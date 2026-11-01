"""Fine-grained rhyme quality scoring: phonetic distance, rhyme richness, schemes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RhymeQuality(Enum):
    IDENTICAL = "identical"      # same word
    RICH = "rich"                # same final syllable onset + nucleus + coda
    PERFECT = "perfect"          # nucleus + coda match
    NEAR = "near"                # nucleus match, coda differs by one feature
    SLANT = "slant"              # coda match, nucleus differs
    ASSONANCE = "assonance"      # vowel match only
    CONSONANCE = "consonance"    # consonant frame match only
    NONE = "none"


@dataclass
class RhymeScore:
    quality: RhymeQuality
    phonetic_similarity: float   # 0-1 based on edit distance on phonemes
    vowel_match: float           # 0-1 vowel correspondence
    consonant_match: float       # 0-1 consonant frame correspondence
    syllable_match: int          # number of trailing syllables that rhyme


@dataclass
class SchemeAnalysis:
    pattern: str                 # e.g. "ABAB"
    density: float               # fraction of lines that rhyme with something
    avg_quality: float           # mean phonetic_similarity across pairs
    richness: float              # fraction of perfect or rich rhymes
    pair_scores: list[tuple[int, int, RhymeScore]]


# --- Phoneme representation (ARPAbet-like, no external deps) ---

# Rough English phoneme mapping from spelling (very simplified — production
# systems would use CMU dict; this gives a usable approximation for scoring).

_VOWEL_PHONEMES = {
    "a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH",
    "ai": "EY", "ay": "EY", "ea": "IY", "ee": "IY", "ie": "AY",
    "oa": "OW", "oo": "UW", "ou": "AW", "ow": "AW", "oy": "OY",
    "oi": "OY", "ar": "AAR", "er": "ER", "ir": "ER", "or": "AOR",
    "ur": "ER", "igh": "AY", "ight": "AY",
}

_CONSONANT_FEATURES = {
    # phoneme -> (place, manner, voiced) — simplified 3-feature system
    "b": (0, 0, 1), "p": (0, 0, 0),
    "d": (1, 0, 1), "t": (1, 0, 0),
    "g": (2, 0, 1), "k": (2, 0, 0),
    "v": (0, 1, 1), "f": (0, 1, 0),
    "z": (1, 1, 1), "s": (1, 1, 0),
    "zh": (2, 1, 1), "sh": (2, 1, 0),
    "dh": (1, 1, 1), "th": (1, 1, 0),
    "m": (0, 2, 1), "n": (1, 2, 1), "ng": (2, 2, 1),
    "l": (1, 3, 1), "r": (1, 4, 1),
    "w": (0, 5, 1), "y": (2, 5, 1),
    "h": (3, 1, 0), "ch": (2, 6, 0), "j": (2, 6, 1),
}


def _extract_phonemes(word: str) -> list[str]:
    """Rough grapheme-to-phoneme for English (approximate)."""
    w = word.lower().strip()
    phonemes: list[str] = []
    i = 0
    while i < len(w):
        # Try digraphs first
        if i + 3 <= len(w) and w[i:i+3] in _VOWEL_PHONEMES:
            phonemes.append(_VOWEL_PHONEMES[w[i:i+3]])
            i += 3
        elif i + 2 <= len(w) and w[i:i+2] in _VOWEL_PHONEMES:
            phonemes.append(_VOWEL_PHONEMES[w[i:i+2]])
            i += 2
        elif i + 2 <= len(w) and w[i:i+2] in _CONSONANT_FEATURES:
            phonemes.append(w[i:i+2])
            i += 2
        elif w[i] in _VOWEL_PHONEMES:
            phonemes.append(_VOWEL_PHONEMES[w[i]])
            i += 1
        elif w[i] in _CONSONANT_FEATURES:
            phonemes.append(w[i])
            i += 1
        else:
            i += 1
    return phonemes


def _is_vowel_phoneme(p: str) -> bool:
    return p.isupper()  # vowel phonemes are uppercase in our scheme


def _split_rime(phonemes: list[str]) -> tuple[list[str], list[str]]:
    """Split into (nucleus vowels, coda consonants) from the last vowel."""
    last_v = -1
    for i, p in enumerate(phonemes):
        if _is_vowel_phoneme(p):
            last_v = i
    if last_v < 0:
        return [], phonemes[:]
    nucleus = []
    for p in phonemes[last_v:]:
        if _is_vowel_phoneme(p):
            nucleus.append(p)
        else:
            break
    coda_start = last_v + len(nucleus)
    coda = phonemes[coda_start:]
    return nucleus, coda


def _consonant_distance(a: str, b: str) -> float:
    """Feature distance between two consonant phonemes (0-1)."""
    fa = _CONSONANT_FEATURES.get(a, (0, 0, 0))
    fb = _CONSONANT_FEATURES.get(b, (0, 0, 0))
    diffs = sum(1 for x, y in zip(fa, fb) if x != y)
    return diffs / 3.0


def _edit_distance_phonemes(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance on phoneme sequences."""
    na, nb = len(a), len(b)
    dp = list(range(nb + 1))
    for i in range(1, na + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, nb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = temp
    return dp[nb]


def score_rhyme(word_a: str, word_b: str) -> RhymeScore:
    """Score the rhyme quality between two words."""
    a_ph = _extract_phonemes(word_a)
    b_ph = _extract_phonemes(word_b)

    if not a_ph or not b_ph:
        return RhymeScore(RhymeQuality.NONE, 0.0, 0.0, 0.0, 0)

    # Phonetic similarity via normalised edit distance
    max_len = max(len(a_ph), len(b_ph))
    edit_d = _edit_distance_phonemes(a_ph, b_ph)
    phonetic_sim = 1.0 - edit_d / max_len if max_len > 0 else 1.0

    # Rime analysis (nucleus + coda)
    a_nuc, a_coda = _split_rime(a_ph)
    b_nuc, b_coda = _split_rime(b_ph)

    # Vowel match: fraction of matching vowels in nucleus
    if a_nuc and b_nuc:
        common = min(len(a_nuc), len(b_nuc))
        matches = sum(1 for x, y in zip(reversed(a_nuc), reversed(b_nuc)) if x == y)
        vowel_match = matches / common
    else:
        vowel_match = 0.0

    # Consonant match: compare coda frames
    if a_coda or b_coda:
        max_c = max(len(a_coda), len(b_coda))
        c_edit = _edit_distance_phonemes(a_coda, b_coda)
        consonant_match = 1.0 - c_edit / max_c if max_c > 0 else 1.0
    else:
        consonant_match = 1.0  # both open syllables

    # Count trailing syllable rhyme depth
    syl_match = 0
    a_rev = list(reversed(a_ph))
    b_rev = list(reversed(b_ph))
    in_vowel = False
    for pa, pb in zip(a_rev, b_rev):
        if pa != pb:
            break
        if _is_vowel_phoneme(pa) and not in_vowel:
            syl_match += 1
            in_vowel = True
        elif not _is_vowel_phoneme(pa):
            in_vowel = False

    # Classify quality
    wa = word_a.lower().strip()
    wb = word_b.lower().strip()
    if wa == wb:
        quality = RhymeQuality.IDENTICAL
    elif phonetic_sim > 0.85 and vowel_match > 0.9 and consonant_match > 0.9:
        quality = RhymeQuality.RICH
    elif vowel_match > 0.8 and consonant_match > 0.7:
        quality = RhymeQuality.PERFECT
    elif vowel_match > 0.7 and consonant_match > 0.4:
        quality = RhymeQuality.NEAR
    elif consonant_match > 0.7 and vowel_match < 0.5:
        quality = RhymeQuality.CONSONANCE
    elif vowel_match > 0.7 and consonant_match < 0.3:
        quality = RhymeQuality.ASSONANCE
    elif vowel_match > 0.3 or consonant_match > 0.3:
        quality = RhymeQuality.SLANT
    else:
        quality = RhymeQuality.NONE

    return RhymeScore(quality, phonetic_sim, vowel_match, consonant_match, syl_match)


def analyze_rhyme_scheme(line_endings: list[str]) -> SchemeAnalysis:
    """Analyse a rhyme scheme with quality scores for each pair.

    Returns the detected ABAB-style pattern, density, average quality,
    richness fraction, and per-pair scores.
    """
    n = len(line_endings)
    if n == 0:
        return SchemeAnalysis("", 0.0, 0.0, 0.0, [])

    # Score all pairs and assign scheme labels
    labels = [""] * n
    pairs: list[tuple[int, int, RhymeScore]] = []
    next_label = 0
    threshold = 0.3  # minimum phonetic_similarity to count as rhyming

    for i in range(n):
        if labels[i]:
            continue
        labels[i] = chr(ord("A") + next_label)
        for j in range(i + 1, n):
            if labels[j]:
                continue
            sc = score_rhyme(line_endings[i], line_endings[j])
            if sc.phonetic_similarity >= threshold and sc.quality != RhymeQuality.NONE:
                labels[j] = labels[i]
                pairs.append((i, j, sc))
        next_label += 1

    pattern = "".join(labels)
    rhyming_lines = set()
    for a, b, _ in pairs:
        rhyming_lines.add(a)
        rhyming_lines.add(b)
    density = len(rhyming_lines) / n if n > 0 else 0.0
    avg_q = float(np.mean([p[2].phonetic_similarity for p in pairs])) if pairs else 0.0
    rich_count = sum(1 for _, _, s in pairs
                     if s.quality in (RhymeQuality.PERFECT, RhymeQuality.RICH))
    richness = rich_count / len(pairs) if pairs else 0.0

    return SchemeAnalysis(pattern, density, avg_q, richness, pairs)


def internal_rhyme_density(line: str) -> float:
    """Detect internal rhyme within a single line.

    Scores all word pairs within the line and returns the fraction
    that achieve at least SLANT quality.
    """
    words = re.findall(r"[a-z']+", line.lower())
    if len(words) < 2:
        return 0.0
    count = 0
    total = 0
    for i in range(len(words)):
        for j in range(i + 2, len(words)):  # skip adjacent
            sc = score_rhyme(words[i], words[j])
            total += 1
            if sc.quality not in (RhymeQuality.NONE, RhymeQuality.IDENTICAL):
                count += 1
    return count / total if total > 0 else 0.0


def multisyllabic_rhyme_score(word_a: str, word_b: str) -> float:
    """Score how many trailing syllables rhyme (normalised 0-1).

    Multi-syllabic rhyme (e.g. "communicate" / "annihilate") is prized
    in hip-hop and complex lyric writing.
    """
    sc = score_rhyme(word_a, word_b)
    max_syl = max(
        sum(1 for p in _extract_phonemes(word_a) if _is_vowel_phoneme(p)),
        sum(1 for p in _extract_phonemes(word_b) if _is_vowel_phoneme(p)),
        1,
    )
    return sc.syllable_match / max_syl
