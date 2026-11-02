"""Rhyme scheme generation and near-rhyme graph construction.

Builds a phonetic similarity graph over a vocabulary, generates rhyming
word suggestions, scores multi-syllable (compound) rhymes, and produces
rhyme-scheme-constrained line endings for lyric drafting.

References:
  - Lerdahl & Jackendoff (1983) on hierarchical structure
  - Patel (2003) on linguistic-musical parallels
  - Hirjee & Brown (2010) on multi-syllable rhyme in rap
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum


class RhymeType(Enum):
    PERFECT = "perfect"
    NEAR = "near"
    SLANT = "slant"
    ASSONANCE = "assonance"
    CONSONANCE = "consonance"
    IDENTITY = "identity"
    MULTI_SYLLABLE = "multi_syllable"


@dataclass
class RhymeCandidate:
    word: str
    rhyme_type: RhymeType
    score: float  # 0..1
    shared_suffix: str
    syllable_count: int


@dataclass
class RhymeGraphNode:
    word: str
    phonetic_key: str
    neighbors: List[Tuple[str, float]]  # (word, similarity)


@dataclass
class RhymeGraph:
    nodes: Dict[str, RhymeGraphNode]
    n_edges: int
    density: float

    def neighbors(self, word: str, min_score: float = 0.3) -> List[Tuple[str, float]]:
        node = self.nodes.get(word.lower())
        if node is None:
            return []
        return [(w, s) for w, s in node.neighbors if s >= min_score]

    def strongest_rhyme(self, word: str) -> Optional[Tuple[str, float]]:
        nbrs = self.neighbors(word, min_score=0.0)
        if not nbrs:
            return None
        return max(nbrs, key=lambda x: x[1])


# ── Phonetic representation ──

_VOWELS = set("aeiouy")
_CONSONANT_CLASSES = {
    'b': 'B', 'p': 'B',
    'd': 'D', 't': 'D',
    'g': 'G', 'k': 'G', 'c': 'G',
    'v': 'V', 'f': 'V',
    'z': 'Z', 's': 'Z',
    'j': 'J', 'ch': 'J',
    'n': 'N', 'm': 'N',
    'l': 'L', 'r': 'L',
    'w': 'W', 'wh': 'W',
    'sh': 'S', 'th': 'T',
}


def _phonetic_key(word: str) -> str:
    w = word.lower().strip()
    if not w:
        return ""
    result: List[str] = []
    i = 0
    while i < len(w):
        if i + 1 < len(w) and w[i:i+2] in _CONSONANT_CLASSES:
            result.append(_CONSONANT_CLASSES[w[i:i+2]])
            i += 2
        elif w[i] in _CONSONANT_CLASSES:
            result.append(_CONSONANT_CLASSES[w[i]])
        elif w[i] in _VOWELS:
            result.append(w[i])
        i += 1 if (i < len(w) and (i == 0 or w[i-1:i+1] not in _CONSONANT_CLASSES)) else 0
        if i >= len(w):
            break
        i += 1
    return "".join(result) if result else w


def _vowel_suffix(word: str, min_len: int = 2) -> str:
    w = word.lower().strip()
    import re
    w = re.sub(r"[^a-z]", "", w)
    if len(w) <= min_len:
        return w
    for i in range(max(0, len(w) - 5), len(w)):
        if w[i] in _VOWELS:
            suffix = w[i:]
            if len(suffix) >= min_len:
                return suffix
    return w[-min_len:]


def _count_syllables(word: str) -> int:
    w = word.lower().strip()
    if not w:
        return 0
    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ie", "ee", "ye")):
        count -= 1
    return max(1, count) if any(c.isalpha() for c in w) else 0


# ── Similarity scoring ──

def rhyme_similarity(word_a: str, word_b: str) -> Tuple[float, RhymeType]:
    a, b = word_a.lower().strip(), word_b.lower().strip()
    if a == b:
        return (1.0, RhymeType.IDENTITY)

    suf_a = _vowel_suffix(a)
    suf_b = _vowel_suffix(b)

    if suf_a == suf_b and a != b:
        return (0.95, RhymeType.PERFECT)

    common_end = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca == cb:
            common_end += 1
        else:
            break

    max_len = max(len(a), len(b), 1)
    end_ratio = common_end / max_len

    vowels_a = [c for c in a if c in _VOWELS]
    vowels_b = [c for c in b if c in _VOWELS]
    if vowels_a and vowels_b:
        common_v = sum(1 for va, vb in zip(reversed(vowels_a), reversed(vowels_b)) if va == vb)
        assonance = common_v / max(len(vowels_a), len(vowels_b), 1)
    else:
        assonance = 0.0

    cons_a = [c for c in a if c.isalpha() and c not in _VOWELS]
    cons_b = [c for c in b if c.isalpha() and c not in _VOWELS]
    if cons_a and cons_b:
        common_c = sum(1 for ca, cb in zip(reversed(cons_a), reversed(cons_b)) if ca == cb)
        consonance = common_c / max(len(cons_a), len(cons_b), 1)
    else:
        consonance = 0.0

    if end_ratio >= 0.4 and assonance >= 0.5:
        score = 0.5 * end_ratio + 0.3 * assonance + 0.2 * consonance
        return (min(0.9, score), RhymeType.NEAR)
    elif assonance >= 0.6:
        return (assonance * 0.7, RhymeType.ASSONANCE)
    elif consonance >= 0.6:
        return (consonance * 0.6, RhymeType.CONSONANCE)
    else:
        score = 0.3 * end_ratio + 0.4 * assonance + 0.3 * consonance
        if score >= 0.3:
            return (score, RhymeType.SLANT)
        return (score, RhymeType.SLANT)


def multi_syllable_rhyme_score(phrase_a: str, phrase_b: str) -> Tuple[float, int]:
    import re
    words_a = re.findall(r"[a-zA-Z']+", phrase_a.lower())
    words_b = re.findall(r"[a-zA-Z']+", phrase_b.lower())
    if not words_a or not words_b:
        return (0.0, 0)

    text_a = "".join(words_a)
    text_b = "".join(words_b)

    syl_a = _count_syllables(text_a)
    syl_b = _count_syllables(text_b)
    min_syl = min(syl_a, syl_b)

    if min_syl <= 1:
        score, _ = rhyme_similarity(words_a[-1], words_b[-1])
        return (score, 1)

    matching_syls = 0
    for k in range(1, min(min_syl + 1, min(len(text_a), len(text_b)) + 1)):
        tail_a = text_a[-k:]
        tail_b = text_b[-k:]
        if tail_a == tail_b:
            matching_syls = max(matching_syls, k)

    approx_syl_matches = 0
    suf_a = _vowel_suffix(text_a, min_len=3)
    suf_b = _vowel_suffix(text_b, min_len=3)
    for va, vb in zip(reversed(suf_a), reversed(suf_b)):
        if va == vb:
            approx_syl_matches += 1
        else:
            break

    rhyme_syls = max(1, approx_syl_matches // 2 + (1 if matching_syls > 0 else 0))
    score = min(1.0, rhyme_syls / max(1, min_syl) * 0.8 + matching_syls / max(1, len(text_a)) * 0.2)
    return (score, rhyme_syls)


# ── Rhyme graph ──

def build_rhyme_graph(vocabulary: List[str],
                      min_similarity: float = 0.3) -> RhymeGraph:
    nodes: Dict[str, RhymeGraphNode] = {}
    n_edges = 0

    for word in vocabulary:
        w = word.lower().strip()
        if not w or not any(c.isalpha() for c in w):
            continue
        key = _phonetic_key(w)
        nodes[w] = RhymeGraphNode(word=w, phonetic_key=key, neighbors=[])

    words = list(nodes.keys())
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            score, rtype = rhyme_similarity(words[i], words[j])
            if score >= min_similarity and rtype != RhymeType.IDENTITY:
                nodes[words[i]].neighbors.append((words[j], score))
                nodes[words[j]].neighbors.append((words[i], score))
                n_edges += 1

    for node in nodes.values():
        node.neighbors.sort(key=lambda x: -x[1])

    max_edges = n * (n - 1) / 2 if n > 1 else 1
    density = n_edges / max_edges if max_edges > 0 else 0.0

    return RhymeGraph(nodes=nodes, n_edges=n_edges, density=density)


def find_rhyme_candidates(target: str, vocabulary: List[str],
                          top_k: int = 10,
                          min_score: float = 0.2) -> List[RhymeCandidate]:
    candidates: List[RhymeCandidate] = []
    t = target.lower().strip()
    for word in vocabulary:
        w = word.lower().strip()
        if w == t or not w:
            continue
        score, rtype = rhyme_similarity(t, w)
        if score >= min_score:
            candidates.append(RhymeCandidate(
                word=w,
                rhyme_type=rtype,
                score=score,
                shared_suffix=_vowel_suffix(w),
                syllable_count=_count_syllables(w),
            ))
    candidates.sort(key=lambda c: -c.score)
    return candidates[:top_k]


def rhyme_chain(start: str, graph: RhymeGraph,
                chain_length: int = 5,
                min_score: float = 0.4) -> List[str]:
    chain = [start.lower().strip()]
    visited: Set[str] = {chain[0]}
    current = chain[0]
    for _ in range(chain_length - 1):
        nbrs = graph.neighbors(current, min_score)
        next_word = None
        for w, s in nbrs:
            if w not in visited:
                next_word = w
                break
        if next_word is None:
            break
        chain.append(next_word)
        visited.add(next_word)
        current = next_word
    return chain


# ── Rhyme scheme generation ──

_COMMON_SCHEMES = {
    "couplet": "AABB",
    "alternate": "ABAB",
    "enclosed": "ABBA",
    "terza_rima": "ABABCBCDC",
    "limerick": "AABBA",
    "monorhyme": "AAAA",
    "ballad": "ABCB",
    "shakespearean_sonnet": "ABABCDCDEFEFGG",
    "petrarchan_octave": "ABBAABBA",
    "villanelle": "ABAABABAABAA",
}


def get_scheme(name: str) -> str:
    return _COMMON_SCHEMES.get(name.lower(), name.upper())


def generate_rhyme_assignments(scheme: str,
                               vocabulary: List[str],
                               seed: int = 42) -> Dict[str, List[str]]:
    rng = np.random.RandomState(seed)
    labels = sorted(set(scheme))
    graph = build_rhyme_graph(vocabulary, min_similarity=0.3)

    assignments: Dict[str, List[str]] = {label: [] for label in labels}
    used: Set[str] = set()
    vocab_lower = [w.lower() for w in vocabulary if w.strip()]

    for label in labels:
        positions = [i for i, ch in enumerate(scheme) if ch == label]
        n_needed = len(positions)
        if not vocab_lower:
            continue
        seed_word = vocab_lower[rng.randint(len(vocab_lower))]
        while seed_word in used and len(used) < len(vocab_lower):
            seed_word = vocab_lower[rng.randint(len(vocab_lower))]

        group = [seed_word]
        used.add(seed_word)

        candidates = find_rhyme_candidates(seed_word, vocab_lower,
                                           top_k=n_needed * 3,
                                           min_score=0.25)
        for c in candidates:
            if len(group) >= n_needed:
                break
            if c.word not in used:
                group.append(c.word)
                used.add(c.word)

        while len(group) < n_needed:
            fallback = vocab_lower[rng.randint(len(vocab_lower))]
            if fallback not in used:
                group.append(fallback)
                used.add(fallback)

        assignments[label] = group[:n_needed]

    return assignments


def scheme_complexity(scheme: str) -> dict:
    n = len(scheme)
    unique = set(scheme)
    n_unique = len(unique)
    pairs = sum(1 for i in range(n) for j in range(i + 1, n)
                if scheme[i] == scheme[j])
    max_pairs = n * (n - 1) / 2 if n > 1 else 1
    rhyme_density = pairs / max_pairs

    distances: List[int] = []
    for label in unique:
        positions = [i for i, ch in enumerate(scheme) if ch == label]
        for k in range(1, len(positions)):
            distances.append(positions[k] - positions[k - 1])

    avg_dist = np.mean(distances) if distances else 0.0
    dist_var = np.var(distances) if distances else 0.0

    return {
        "length": n,
        "unique_labels": n_unique,
        "rhyme_density": float(rhyme_density),
        "avg_rhyme_distance": float(avg_dist),
        "distance_variance": float(dist_var),
        "is_regular": float(dist_var) < 1.0 and n_unique > 1,
    }


def suggest_scheme(n_lines: int, style: str = "default") -> str:
    if style == "rap" or style == "hip_hop":
        return "AA" * (n_lines // 2) + ("A" if n_lines % 2 else "")
    elif style == "ballad":
        full = "ABCB" * (n_lines // 4)
        remainder = "ABCB"[:n_lines % 4]
        return full + remainder
    elif style == "sonnet" and n_lines >= 14:
        return "ABABCDCDEFEFGG"[:n_lines]
    else:
        if n_lines <= 2:
            return "AA"[:n_lines]
        elif n_lines <= 4:
            return "ABAB"[:n_lines]
        elif n_lines <= 8:
            return "ABABCDCD"[:n_lines]
        else:
            return "ABABCDCD"[:8] + "E" * (n_lines - 8)
