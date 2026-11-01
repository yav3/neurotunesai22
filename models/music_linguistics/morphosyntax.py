"""Morphosyntax of lyrics: word-level and phrase-level linguistic analysis.

Analyse lyric morphology (inflection, derivation, compounding) and syntax
(phrase structure, dependency patterns, clause types) as they interact with
musical phrasing, stress, and metre. Includes syllable-stress alignment
scoring and syntactic complexity metrics.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PhraseType(str, Enum):
    NOUN_PHRASE = "NP"
    VERB_PHRASE = "VP"
    ADJECTIVE_PHRASE = "AdjP"
    ADVERB_PHRASE = "AdvP"
    PREPOSITIONAL_PHRASE = "PP"
    CLAUSE = "S"
    SUBORDINATE = "SBAR"


class StressPattern(str, Enum):
    IAMBIC = "iambic"
    TROCHAIC = "trochaic"
    DACTYLIC = "dactylic"
    ANAPESTIC = "anapestic"
    SPONDAIC = "spondaic"
    FREE = "free"


@dataclass
class SyllableInfo:
    text: str
    stressed: bool
    beat_position: float
    on_downbeat: bool


@dataclass
class PhraseNode:
    phrase_type: PhraseType
    words: List[str]
    children: List["PhraseNode"]
    start_beat: float = 0.0
    end_beat: float = 0.0


def syllable_count(word: str) -> int:
    vowels = set("aeiouyAEIOUY")
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    return max(count, 1)


def syllabify_line(line: str) -> List[str]:
    words = line.split()
    syllables = []
    for w in words:
        n = syllable_count(w)
        if n == 1:
            syllables.append(w)
        else:
            chunk_size = max(1, len(w) // n)
            for i in range(n):
                start = i * chunk_size
                end = start + chunk_size if i < n - 1 else len(w)
                syllables.append(w[start:end])
    return syllables


def stress_pattern_detect(stressed_flags: List[bool]) -> StressPattern:
    if len(stressed_flags) < 2:
        return StressPattern.FREE
    pattern = "".join("S" if s else "W" for s in stressed_flags)
    if "WS" * 2 in pattern:
        return StressPattern.IAMBIC
    if "SW" * 2 in pattern:
        return StressPattern.TROCHAIC
    if "SWW" in pattern:
        return StressPattern.DACTYLIC
    if "WWS" in pattern:
        return StressPattern.ANAPESTIC
    if "SS" in pattern and pattern.count("W") < len(pattern) // 3:
        return StressPattern.SPONDAIC
    return StressPattern.FREE


def stress_beat_alignment(syllables: List[SyllableInfo]) -> float:
    if not syllables:
        return 0.0
    aligned = sum(1 for s in syllables if s.stressed == s.on_downbeat)
    return aligned / len(syllables)


def syntactic_complexity(phrase_tree: PhraseNode) -> Dict[str, float]:
    def _depth(node: PhraseNode) -> int:
        if not node.children:
            return 1
        return 1 + max(_depth(c) for c in node.children)

    def _count_nodes(node: PhraseNode) -> int:
        return 1 + sum(_count_nodes(c) for c in node.children)

    def _count_clauses(node: PhraseNode) -> int:
        count = 1 if node.phrase_type in (PhraseType.CLAUSE, PhraseType.SUBORDINATE) else 0
        return count + sum(_count_clauses(c) for c in node.children)

    depth = _depth(phrase_tree)
    nodes = _count_nodes(phrase_tree)
    clauses = _count_clauses(phrase_tree)
    n_words = len(phrase_tree.words) if phrase_tree.words else 1
    return {
        "tree_depth": float(depth),
        "node_count": float(nodes),
        "clause_count": float(clauses),
        "words_per_clause": n_words / max(clauses, 1),
        "branching_factor": nodes / max(depth, 1),
    }


def phrase_music_alignment(phrases: List[PhraseNode],
                           bar_boundaries: np.ndarray) -> float:
    if not phrases or len(bar_boundaries) < 2:
        return 0.0
    aligned = 0
    for p in phrases:
        for bb in bar_boundaries:
            if abs(p.end_beat - bb) < 0.25:
                aligned += 1
                break
    return aligned / len(phrases)


def morphological_density(words: List[str]) -> float:
    if not words:
        return 0.0
    total_morphemes = 0
    suffixes = {"ing", "tion", "ness", "ment", "able", "ful", "less", "ous",
                "ive", "ity", "ence", "ance", "ism", "ist", "ize", "ise",
                "ly", "er", "est", "ed", "es", "al", "en"}
    prefixes = {"un", "re", "dis", "mis", "pre", "non", "over", "under",
                "out", "sub", "inter", "trans", "super"}
    for w in words:
        morph = 1
        w_lower = w.lower()
        for p in prefixes:
            if w_lower.startswith(p) and len(w_lower) > len(p) + 2:
                morph += 1
                break
        for s in suffixes:
            if w_lower.endswith(s) and len(w_lower) > len(s) + 2:
                morph += 1
                break
        total_morphemes += morph
    return total_morphemes / len(words)


def lexical_diversity(words: List[str], window: int = 50) -> float:
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(w.lower() for w in words)) / len(words)
    ttrs = []
    for i in range(0, len(words) - window + 1, window // 2):
        chunk = words[i:i + window]
        ttrs.append(len(set(w.lower() for w in chunk)) / len(chunk))
    return float(np.mean(ttrs))


def function_content_ratio(words: List[str]) -> float:
    function_words = {"the", "a", "an", "is", "am", "are", "was", "were",
                      "be", "been", "being", "have", "has", "had", "do",
                      "does", "did", "will", "would", "shall", "should",
                      "may", "might", "can", "could", "must", "to", "of",
                      "in", "for", "on", "with", "at", "by", "from", "as",
                      "into", "about", "like", "through", "after", "over",
                      "between", "out", "against", "during", "and", "but",
                      "or", "nor", "not", "so", "yet", "both", "either",
                      "neither", "each", "every", "all", "any", "few",
                      "more", "most", "other", "some", "such", "no",
                      "only", "same", "than", "too", "very", "just",
                      "i", "me", "my", "we", "us", "our", "you", "your",
                      "he", "him", "his", "she", "her", "it", "its",
                      "they", "them", "their", "this", "that", "these",
                      "those", "what", "which", "who", "whom", "whose"}
    if not words:
        return 0.0
    func = sum(1 for w in words if w.lower() in function_words)
    return func / len(words)


def enjambment_score(line_ends: List[str], next_line_starts: List[str]) -> float:
    if not line_ends or not next_line_starts:
        return 0.0
    enjambed = 0
    continuation_words = {"and", "but", "or", "the", "a", "an", "to", "of",
                          "in", "that", "which", "who", "with", "from"}
    for end, start in zip(line_ends, next_line_starts):
        end_w = end.strip().rstrip(",.;:!?").lower()
        start_w = start.strip().lower().split()[0] if start.strip() else ""
        if end_w in {"the", "a", "an", "to", "of", "in", "and", "but", "or"}:
            enjambed += 1
        elif start_w in continuation_words:
            enjambed += 0.5
    return enjambed / max(len(line_ends), 1)
