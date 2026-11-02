"""Lyric generation via Markov chains and template-constrained verse construction."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "MarkovOrder", "LyricTemplate", "MarkovChain",
    "build_markov_chain", "generate_line", "syllable_count_estimate",
    "fits_template", "rhyme_score", "generate_verse", "perplexity",
    "vocabulary_richness",
]


class MarkovOrder(Enum):
    BIGRAM = 2
    TRIGRAM = 3


@dataclass
class LyricTemplate:
    syllable_pattern: List[int]
    rhyme_scheme: str
    num_lines: int


@dataclass
class MarkovChain:
    transitions: Dict[Tuple[str, ...], Dict[str, int]] = field(default_factory=dict)
    order: int = 2


def build_markov_chain(corpus: List[str], order: int = 2) -> MarkovChain:
    chain = MarkovChain(order=order)
    for line in corpus:
        tokens = line.lower().split()
        if len(tokens) < order + 1:
            continue
        for i in range(len(tokens) - order):
            key = tuple(tokens[i : i + order])
            nxt = tokens[i + order]
            if key not in chain.transitions:
                chain.transitions[key] = {}
            chain.transitions[key][nxt] = chain.transitions[key].get(nxt, 0) + 1
    return chain


def generate_line(
    chain: MarkovChain,
    seed: Optional[Tuple[str, ...]] = None,
    max_tokens: int = 12,
    temperature: float = 1.0,
    rng: Optional[random.Random] = None,
) -> List[str]:
    rng = rng or random.Random()
    if seed is None:
        if not chain.transitions:
            return []
        seed = rng.choice(list(chain.transitions.keys()))
    result = list(seed)
    current = seed
    for _ in range(max_tokens):
        dist = chain.transitions.get(current)
        if not dist:
            break
        words = list(dist.keys())
        weights = [dist[w] ** (1.0 / max(temperature, 0.01)) for w in words]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen = rng.choices(words, weights=weights, k=1)[0]
        result.append(chosen)
        current = tuple(result[-chain.order :])
    return result


_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def syllable_count_estimate(word: str) -> int:
    word = word.strip().lower()
    if not word:
        return 0
    matches = _VOWEL_RE.findall(word)
    count = len(matches)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def fits_template(line: List[str], target_syllables: int) -> bool:
    total = sum(syllable_count_estimate(w) for w in line)
    return total == target_syllables


def rhyme_score(word_a: str, word_b: str) -> float:
    a = word_a.lower().strip()
    b = word_b.lower().strip()
    if not a or not b:
        return 0.0
    max_suffix = min(len(a), len(b))
    match_len = 0
    for i in range(1, max_suffix + 1):
        if a[-i] == b[-i]:
            match_len = i
        else:
            break
    return match_len / max(len(a), len(b))


def generate_verse(
    chain: MarkovChain,
    template: LyricTemplate,
    seed: Optional[int] = None,
    max_attempts: int = 50,
) -> List[str]:
    rng = random.Random(seed)
    lines: List[str] = []
    rhyme_groups: Dict[str, str] = {}

    for i in range(template.num_lines):
        target_syl = template.syllable_pattern[i % len(template.syllable_pattern)]
        scheme_char = template.rhyme_scheme[i % len(template.rhyme_scheme)]
        best_line: Optional[List[str]] = None
        best_score = -1.0

        for _ in range(max_attempts):
            candidate = generate_line(chain, max_tokens=target_syl + 4, rng=rng)
            if not candidate:
                continue
            syl = sum(syllable_count_estimate(w) for w in candidate)
            syl_penalty = abs(syl - target_syl) / max(target_syl, 1)
            rhyme_bonus = 0.0
            if scheme_char in rhyme_groups and candidate:
                rhyme_bonus = rhyme_score(candidate[-1], rhyme_groups[scheme_char])
            score = (1.0 - syl_penalty) + rhyme_bonus
            if score > best_score:
                best_score = score
                best_line = candidate

        line_str = " ".join(best_line) if best_line else ""
        lines.append(line_str)
        if scheme_char not in rhyme_groups and best_line:
            rhyme_groups[scheme_char] = best_line[-1]

    return lines


def perplexity(chain: MarkovChain, tokens: List[str]) -> float:
    if len(tokens) <= chain.order:
        return float("inf")
    log_prob_sum = 0.0
    count = 0
    for i in range(len(tokens) - chain.order):
        key = tuple(tokens[i : i + chain.order])
        nxt = tokens[i + chain.order]
        dist = chain.transitions.get(key)
        if not dist:
            return float("inf")
        total = sum(dist.values())
        prob = dist.get(nxt, 0) / total
        if prob == 0:
            return float("inf")
        log_prob_sum += math.log2(prob)
        count += 1
    if count == 0:
        return float("inf")
    return 2.0 ** (-log_prob_sum / count)


def vocabulary_richness(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(t.lower() for t in tokens)) / len(tokens)
