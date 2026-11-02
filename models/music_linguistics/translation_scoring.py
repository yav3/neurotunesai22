"""Lyric translation / adaptation scoring: singability preservation across languages."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np


class AdaptationDimension(Enum):
    SYLLABLE_COUNT = "syllable_count"
    STRESS_ALIGNMENT = "stress_alignment"
    VOWEL_OPENNESS = "vowel_openness"
    SEMANTIC_FIDELITY = "semantic_fidelity"
    RHYME_PRESERVATION = "rhyme_preservation"
    SINGABILITY = "singability"


@dataclass
class AdaptationScore:
    overall: float  # weighted mean, 0-1
    dimensions: dict[str, float]  # per-dimension scores
    penalties: list[str]  # descriptions of specific problems
    suggestions: list[str]  # actionable improvement hints


@dataclass
class SyllableMap:
    """Maps syllable counts between source and target lines."""
    source_counts: list[int]
    target_counts: list[int]
    deltas: list[int]  # target - source per line
    excess_ratio: float  # fraction of lines where |delta| > 1


@dataclass
class VowelProfile:
    """Distribution of vowel openness in a text."""
    open_frac: float    # /a/, /æ/, /ɑ/
    mid_frac: float     # /e/, /ɛ/, /o/, /ɔ/
    close_frac: float   # /i/, /u/, /ɪ/, /ʊ/
    mean_openness: float  # 0=close, 1=open


# Vowel openness classification (simplified, spelling-based heuristic)
_OPEN_VOWELS = set("aà")
_MID_VOWELS = set("eèéoòó")
_CLOSE_VOWELS = set("iìíuùú")


def _count_syllables(text: str) -> int:
    """Language-agnostic syllable count heuristic based on vowel clusters."""
    vowels = set("aeiouyàèéìòóùúäëïöüâêîôû")
    text = text.lower()
    count = 0
    in_vowel = False
    for ch in text:
        if ch in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(count, 1) if text.strip() else 0


def _line_syllables(text: str) -> list[int]:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return [_count_syllables(l) for l in lines]


def syllable_map(source: str, target: str) -> SyllableMap:
    """Compare syllable counts line-by-line between source and target."""
    sc = _line_syllables(source)
    tc = _line_syllables(target)
    # Pad shorter to match longer
    max_len = max(len(sc), len(tc))
    sc.extend([0] * (max_len - len(sc)))
    tc.extend([0] * (max_len - len(tc)))
    deltas = [t - s for s, t in zip(sc, tc)]
    excess = sum(1 for d in deltas if abs(d) > 1)
    ratio = excess / max_len if max_len > 0 else 0.0
    return SyllableMap(sc, tc, deltas, ratio)


def syllable_fidelity(source: str, target: str) -> float:
    """Score how well the target preserves source syllable counts (0-1).

    Perfect score when every line has the same syllable count.
    Penalty proportional to the relative deviation.
    """
    sm = syllable_map(source, target)
    if not sm.source_counts:
        return 1.0
    scores = []
    for s, t in zip(sm.source_counts, sm.target_counts):
        if s == 0:
            scores.append(1.0 if t == 0 else 0.0)
        else:
            scores.append(max(0.0, 1.0 - abs(t - s) / s))
    return float(np.mean(scores))


def _extract_vowels(text: str) -> list[str]:
    all_v = _OPEN_VOWELS | _MID_VOWELS | _CLOSE_VOWELS
    return [ch for ch in text.lower() if ch in all_v]


def vowel_profile(text: str) -> VowelProfile:
    """Analyse vowel openness distribution in text."""
    vowels = _extract_vowels(text)
    if not vowels:
        return VowelProfile(0.0, 0.0, 0.0, 0.5)
    n = len(vowels)
    o = sum(1 for v in vowels if v in _OPEN_VOWELS) / n
    m = sum(1 for v in vowels if v in _MID_VOWELS) / n
    c = sum(1 for v in vowels if v in _CLOSE_VOWELS) / n
    openness = o * 1.0 + m * 0.5 + c * 0.0
    return VowelProfile(o, m, c, openness)


def vowel_openness_similarity(source: str, target: str) -> float:
    """Score how well the target preserves the source's vowel openness profile.

    Important for singability — open vowels on high notes, close vowels for
    fast passages. Uses Jensen-Shannon divergence on the 3-bin distribution.
    """
    sp = vowel_profile(source)
    tp = vowel_profile(target)
    p = np.array([sp.open_frac, sp.mid_frac, sp.close_frac]) + 1e-10
    q = np.array([tp.open_frac, tp.mid_frac, tp.close_frac]) + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * float(np.sum(p * np.log2(p / m))) + 0.5 * float(np.sum(q * np.log2(q / m)))
    return float(1.0 - np.sqrt(np.clip(jsd, 0, 1)))


def _stress_pattern_simple(text: str) -> list[int]:
    """Rough binary stress pattern: 1 for stressed, 0 for unstressed.

    Uses syllable count as a proxy — monosyllables stressed, polysyllabic
    words alternate starting stressed.
    """
    words = re.findall(r"[a-zàèéìòóùúäëïöüâêîôû']+", text.lower())
    pattern: list[int] = []
    for w in words:
        n = _count_syllables(w)
        if n <= 1:
            pattern.append(1)
        else:
            for i in range(n):
                pattern.append(1 if i % 2 == 0 else 0)
    return pattern


def stress_alignment_score(source: str, target: str) -> float:
    """Score how well the target's stress pattern aligns with the source's.

    Computed per-line, using normalised Hamming distance on the binary
    stress sequences.
    """
    src_lines = [l.strip() for l in source.strip().split("\n") if l.strip()]
    tgt_lines = [l.strip() for l in target.strip().split("\n") if l.strip()]
    max_len = max(len(src_lines), len(tgt_lines))
    if max_len == 0:
        return 1.0
    scores = []
    for sl, tl in zip(src_lines, tgt_lines):
        sp = _stress_pattern_simple(sl)
        tp = _stress_pattern_simple(tl)
        # Align to shorter
        n = min(len(sp), len(tp))
        if n == 0:
            scores.append(0.0)
            continue
        matches = sum(1 for a, b in zip(sp[:n], tp[:n]) if a == b)
        scores.append(matches / n)
    return float(np.mean(scores)) if scores else 1.0


def _get_line_endings(text: str) -> list[str]:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    endings = []
    for l in lines:
        words = re.findall(r"[a-zàèéìòóùúäëïöüâêîôû']+", l.lower())
        endings.append(words[-1] if words else "")
    return endings


def rhyme_preservation_score(source: str, target: str) -> float:
    """Score how well the target preserves the source's rhyme structure.

    Detects the source rhyme scheme (which lines rhyme with each other)
    and checks whether the same line pairs rhyme in the target.
    """
    src_ends = _get_line_endings(source)
    tgt_ends = _get_line_endings(target)
    n = min(len(src_ends), len(tgt_ends))
    if n < 2:
        return 1.0

    # Detect source rhyme pairs (using simple suffix matching)
    def _rhymes(a: str, b: str) -> bool:
        if not a or not b or a == b:
            return False
        min_len = min(len(a), len(b), 3)
        return a[-min_len:] == b[-min_len:]

    src_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _rhymes(src_ends[i], src_ends[j]):
                src_pairs.append((i, j))

    if not src_pairs:
        return 1.0  # no rhymes to preserve

    preserved = sum(1 for i, j in src_pairs
                    if j < len(tgt_ends) and _rhymes(tgt_ends[i], tgt_ends[j]))
    return preserved / len(src_pairs)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zàèéìòóùúäëïöüâêîôû']+", text.lower())


def semantic_fidelity_score(source: str, target: str) -> float:
    """Estimate semantic preservation using word-overlap (Jaccard + IDF-weighted).

    In production, this would use cross-lingual embeddings. This implementation
    uses token overlap as a lower-bound proxy, which works when source and
    target share cognates (e.g. Romance languages) or when both are in English
    (original vs. adaptation).
    """
    src_tok = set(_tokenize(source))
    tgt_tok = set(_tokenize(target))
    if not src_tok and not tgt_tok:
        return 1.0
    if not src_tok or not tgt_tok:
        return 0.0

    # Jaccard similarity
    intersection = src_tok & tgt_tok
    union = src_tok | tgt_tok
    jaccard = len(intersection) / len(union)

    # Weight by "importance" (longer words more likely to be content words)
    weighted_overlap = sum(len(w) for w in intersection)
    weighted_total = sum(len(w) for w in union)
    weighted = weighted_overlap / weighted_total if weighted_total > 0 else 0.0

    return 0.4 * jaccard + 0.6 * weighted


def score_adaptation(source: str, target: str,
                     weights: dict[str, float] | None = None) -> AdaptationScore:
    """Comprehensive adaptation quality score across all dimensions.

    Default weights emphasise singability (syllable count + vowel openness)
    over semantic fidelity, reflecting that sung translations prioritise
    performability.
    """
    if weights is None:
        weights = {
            "syllable_count": 0.25,
            "stress_alignment": 0.20,
            "vowel_openness": 0.15,
            "semantic_fidelity": 0.15,
            "rhyme_preservation": 0.15,
            "singability": 0.10,
        }

    dims: dict[str, float] = {}
    dims["syllable_count"] = syllable_fidelity(source, target)
    dims["stress_alignment"] = stress_alignment_score(source, target)
    dims["vowel_openness"] = vowel_openness_similarity(source, target)
    dims["semantic_fidelity"] = semantic_fidelity_score(source, target)
    dims["rhyme_preservation"] = rhyme_preservation_score(source, target)

    # Composite singability: geometric mean of syllable and vowel scores
    dims["singability"] = math.sqrt(
        max(0, dims["syllable_count"]) * max(0, dims["vowel_openness"])
    )

    # Weighted overall
    total_w = sum(weights.values()) or 1.0
    overall = sum(dims.get(k, 0) * w for k, w in weights.items()) / total_w

    # Generate penalties and suggestions
    penalties: list[str] = []
    suggestions: list[str] = []

    sm = syllable_map(source, target)
    if sm.excess_ratio > 0.3:
        penalties.append(
            f"{sm.excess_ratio:.0%} of lines differ by >1 syllable"
        )
        worst = max(range(len(sm.deltas)), key=lambda i: abs(sm.deltas[i]))
        suggestions.append(
            f"Line {worst+1}: {sm.source_counts[worst]} → {sm.target_counts[worst]} syllables"
        )

    if dims["stress_alignment"] < 0.6:
        penalties.append("Stress alignment below 60%: rhythmic feel will differ")
        suggestions.append("Re-order words to match strong-beat positions")

    if dims["vowel_openness"] < 0.7:
        sp = vowel_profile(source)
        tp = vowel_profile(target)
        if tp.mean_openness < sp.mean_openness - 0.15:
            suggestions.append("Target uses more closed vowels — swap for open equivalents on sustained notes")
        elif tp.mean_openness > sp.mean_openness + 0.15:
            suggestions.append("Target uses more open vowels — may sound broader than original")

    if dims["rhyme_preservation"] < 0.5:
        penalties.append("More than half of source rhyme pairs lost in translation")

    return AdaptationScore(overall, dims, penalties, suggestions)


def back_translation_consistency(original: str, back_translated: str) -> float:
    """Score round-trip translation consistency (original → L2 → back to L1).

    Higher scores suggest the intermediate translation preserved meaning.
    Uses weighted Jaccard on the original vs back-translation.
    """
    return semantic_fidelity_score(original, back_translated)


def note_density_ratio(source: str, target: str) -> float:
    """Ratio of target syllables to source syllables (overall).

    Values near 1.0 mean the adaptation fits the same number of notes.
    Values >1.0 mean the target needs more notes (melisma reduction needed).
    """
    sc = sum(_line_syllables(source))
    tc = sum(_line_syllables(target))
    if sc == 0:
        return 1.0
    return tc / sc
