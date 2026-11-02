"""Lyric discourse analysis: narrative arc, semantic shift, thematic coherence."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class NarrativePhase(Enum):
    SETUP = "setup"
    RISING = "rising"
    CLIMAX = "climax"
    FALLING = "falling"
    RESOLUTION = "resolution"


@dataclass
class NarrativeArc:
    phase_labels: list[NarrativePhase]
    tension_curve: np.ndarray
    climax_index: int
    arc_type: str  # "classic", "flat", "inverted", "double_peak"


@dataclass
class SemanticShift:
    section_a: int
    section_b: int
    cosine_distance: float
    novel_terms: list[str]
    dropped_terms: list[str]


@dataclass
class ThematicProfile:
    themes: dict[str, float]  # theme -> weight
    coherence: float  # avg pairwise cosine sim
    diversity: float  # entropy over themes
    progression: list[dict[str, float]]  # per-section theme weights


# --- TF-IDF machinery (no external deps) ---

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens) or 1
    return {t: c / n for t, c in counts.items()}


def _idf(documents: list[list[str]]) -> dict[str, float]:
    n = len(documents)
    df: dict[str, int] = {}
    for doc in documents:
        seen = set(doc)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def _tfidf_vector(tf: dict[str, float], idf: dict[str, float],
                  vocab: list[str]) -> np.ndarray:
    v = np.zeros(len(vocab))
    idx = {w: i for i, w in enumerate(vocab)}
    for w, tfw in tf.items():
        if w in idx:
            v[idx[w]] = tfw * idf.get(w, 1.0)
    norm = np.linalg.norm(v)
    if norm > 1e-12:
        v /= norm
    return v


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(d / (na * nb), -1.0, 1.0))


# --- Narrative tension estimation ---

# Lexicon: words that correlate with narrative tension
_TENSION_POS = {
    "fire", "burn", "fight", "scream", "cry", "break", "crash", "storm",
    "rage", "blood", "die", "death", "destroy", "pain", "war", "hate",
    "desperate", "fear", "dark", "lost", "fall", "never", "always",
    "explode", "shatter", "collapse", "drown", "bleed", "suffocate",
}
_TENSION_NEG = {
    "peace", "calm", "gentle", "soft", "quiet", "rest", "easy", "slow",
    "home", "warm", "light", "sun", "smile", "love", "sweet", "dream",
    "float", "breathe", "still", "safe", "held", "together", "stay",
}


def _tension_score(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in _TENSION_POS)
    neg = sum(1 for t in tokens if t in _TENSION_NEG)
    n = len(tokens)
    excl = tokens.count("!") if "!" in tokens else 0
    raw = (pos - neg) / n + 0.1 * excl / n
    return float(np.clip(raw, -1.0, 1.0))


def estimate_narrative_arc(sections: list[str]) -> NarrativeArc:
    """Estimate narrative arc from ordered lyric sections.

    Each section is a block of text (verse, chorus, bridge, etc.).
    Tension is computed per section via a lexicon-based score, then
    the curve is smoothed and classified into a canonical arc type.
    """
    if not sections:
        return NarrativeArc([], np.array([]), 0, "flat")

    tensions = np.array([_tension_score(_tokenize(s)) for s in sections])

    # Smooth with a 3-point moving average
    if len(tensions) >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        smoothed = np.convolve(tensions, kernel, mode="same")
    else:
        smoothed = tensions.copy()

    climax_idx = int(np.argmax(smoothed))
    n = len(sections)

    # Classify arc type
    if n <= 1 or np.ptp(smoothed) < 0.05:
        arc_type = "flat"
    else:
        rel_pos = climax_idx / max(n - 1, 1)
        peaks = []
        for i in range(1, n - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                peaks.append(i)
        if len(peaks) >= 2:
            arc_type = "double_peak"
        elif rel_pos < 0.3:
            arc_type = "inverted"
        else:
            arc_type = "classic"

    # Assign phases by position relative to climax
    phases: list[NarrativePhase] = []
    for i in range(n):
        if i < climax_idx * 0.4:
            phases.append(NarrativePhase.SETUP)
        elif i < climax_idx:
            phases.append(NarrativePhase.RISING)
        elif i == climax_idx:
            phases.append(NarrativePhase.CLIMAX)
        elif i < climax_idx + (n - climax_idx) * 0.6:
            phases.append(NarrativePhase.FALLING)
        else:
            phases.append(NarrativePhase.RESOLUTION)

    return NarrativeArc(phases, smoothed, climax_idx, arc_type)


def semantic_shift_between(section_a: str, section_b: str,
                           idx_a: int = 0, idx_b: int = 1) -> SemanticShift:
    """Measure semantic distance between two lyric sections via TF-IDF cosine."""
    tok_a = _tokenize(section_a)
    tok_b = _tokenize(section_b)
    all_docs = [tok_a, tok_b]
    idf = _idf(all_docs)
    vocab = sorted(idf.keys())
    if not vocab:
        return SemanticShift(idx_a, idx_b, 1.0, [], [])
    va = _tfidf_vector(_tf(tok_a), idf, vocab)
    vb = _tfidf_vector(_tf(tok_b), idf, vocab)
    dist = 1.0 - _cosine_sim(va, vb)
    set_a, set_b = set(tok_a), set(tok_b)
    novel = sorted(set_b - set_a)
    dropped = sorted(set_a - set_b)
    return SemanticShift(idx_a, idx_b, float(dist), novel, dropped)


def pairwise_semantic_shifts(sections: list[str]) -> list[SemanticShift]:
    """Compute semantic shift between consecutive section pairs."""
    shifts = []
    for i in range(len(sections) - 1):
        shifts.append(semantic_shift_between(sections[i], sections[i + 1], i, i + 1))
    return shifts


def thematic_profile(sections: list[str],
                     theme_seeds: dict[str, list[str]] | None = None) -> ThematicProfile:
    """Build a thematic profile across sections.

    theme_seeds maps theme names to seed words. If None, themes are
    derived from the top-k TF-IDF terms across all sections.
    """
    all_tokens = [_tokenize(s) for s in sections]
    flat = [t for doc in all_tokens for t in doc]
    idf = _idf(all_tokens)
    vocab = sorted(idf.keys())
    if not vocab:
        return ThematicProfile({}, 0.0, 0.0, [])

    if theme_seeds is None:
        # Auto-derive themes from top TF-IDF terms
        global_tf = _tf(flat)
        scored = {w: global_tf.get(w, 0) * idf.get(w, 1.0) for w in vocab}
        top = sorted(scored, key=scored.get, reverse=True)[:10]
        theme_seeds = {w: [w] for w in top}

    # Build theme vectors (centroid of seed word TF-IDF vectors)
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    theme_vecs: dict[str, np.ndarray] = {}
    for name, seeds in theme_seeds.items():
        v = np.zeros(len(vocab))
        count = 0
        for s in seeds:
            if s in vocab_idx:
                v[vocab_idx[s]] = idf.get(s, 1.0)
                count += 1
        if count > 0:
            v /= count
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v /= norm
        theme_vecs[name] = v

    # Score each section against themes
    progression: list[dict[str, float]] = []
    for tokens in all_tokens:
        sv = _tfidf_vector(_tf(tokens), idf, vocab)
        weights = {}
        for name, tv in theme_vecs.items():
            weights[name] = max(0.0, float(_cosine_sim(sv, tv)))
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        progression.append(weights)

    # Global theme weights (average)
    themes: dict[str, float] = {}
    for name in theme_seeds:
        themes[name] = float(np.mean([p.get(name, 0) for p in progression]))

    # Coherence: avg pairwise cosine between section TF-IDF vectors
    sec_vecs = [_tfidf_vector(_tf(t), idf, vocab) for t in all_tokens]
    sims = []
    for i in range(len(sec_vecs)):
        for j in range(i + 1, len(sec_vecs)):
            sims.append(_cosine_sim(sec_vecs[i], sec_vecs[j]))
    coherence = float(np.mean(sims)) if sims else 1.0

    # Diversity: entropy over theme distribution
    vals = np.array(list(themes.values()))
    vals = vals / (vals.sum() or 1.0)
    diversity = float(-np.sum(v * np.log2(v + 1e-12) for v in vals if v > 0))

    return ThematicProfile(themes, coherence, diversity, progression)


def repetition_density(text: str, ngram_range: tuple[int, int] = (2, 4)) -> float:
    """Fraction of text covered by repeated n-grams.

    Captures the degree of lyrical repetition (hooks, refrains, anaphora).
    """
    tokens = _tokenize(text)
    if len(tokens) < ngram_range[0]:
        return 0.0
    repeated_positions: set[int] = set()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            ngrams.setdefault(gram, []).append(i)
        for positions in ngrams.values():
            if len(positions) > 1:
                for p in positions:
                    for k in range(n):
                        repeated_positions.add(p + k)
    return len(repeated_positions) / len(tokens)


def lexical_density(text: str) -> float:
    """Ratio of unique tokens to total tokens (type-token ratio)."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def imagery_score(text: str) -> float:
    """Heuristic imagery vividness score based on concrete/sensory word density.

    Uses a compact sensory lexicon covering five modalities (visual, auditory,
    tactile, olfactory, gustatory).
    """
    _IMAGERY = {
        "red", "blue", "green", "gold", "silver", "white", "black", "bright",
        "dark", "shadow", "light", "glow", "shimmer", "flash", "blind",
        "loud", "silent", "echo", "ring", "whisper", "roar", "hum", "buzz",
        "sharp", "soft", "rough", "smooth", "cold", "hot", "wet", "dry",
        "sweet", "bitter", "sour", "salt", "taste", "tongue", "mouth",
        "smoke", "perfume", "scent", "stink", "fresh", "rotten",
        "blood", "bone", "skin", "hair", "eye", "hand", "heart",
        "stone", "iron", "glass", "silk", "velvet", "ice", "flame",
        "rain", "snow", "wind", "dust", "mud", "sand", "ocean", "river",
        "mountain", "forest", "sky", "moon", "star", "sun",
    }
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _IMAGERY) / len(tokens)
