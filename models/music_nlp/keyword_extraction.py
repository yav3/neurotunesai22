"""Music-domain keyword and keyphrase extraction from text.

TF-IDF, TextRank-style graph scoring, n-gram extraction. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import re

import numpy as np


@dataclass
class Keyword:
    text: str
    score: float
    frequency: int
    is_music_term: bool = False


MUSIC_VOCABULARY = {
    "melody", "harmony", "rhythm", "chord", "scale", "key", "tempo",
    "beat", "bass", "treble", "minor", "major", "flat", "sharp",
    "octave", "pitch", "timbre", "tone", "note", "rest", "measure",
    "bar", "chorus", "verse", "bridge", "hook", "riff", "groove",
    "syncopation", "arpeggio", "crescendo", "diminuendo", "forte",
    "piano", "staccato", "legato", "vibrato", "tremolo", "glissando",
    "counterpoint", "fugue", "sonata", "symphony", "concerto",
    "overture", "prelude", "nocturne", "etude", "waltz", "march",
    "blues", "jazz", "rock", "pop", "classical", "folk", "country",
    "hip-hop", "electronic", "ambient", "metal", "punk", "soul",
    "funk", "reggae", "latin", "swing", "bossa", "samba",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "and", "but", "or", "nor", "not", "so",
    "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "about", "above", "below",
    "between", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "what", "which", "who", "whom", "this", "that", "these",
    "those", "i", "me", "my", "myself", "we", "our", "ours", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it",
    "its", "they", "them", "their", "theirs",
}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower()) if w not in STOPWORDS]


def term_frequency(tokens: list[str]) -> dict[str, int]:
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def idf_scores(documents: Sequence[Sequence[str]]) -> dict[str, float]:
    """Inverse document frequency: log(N / df)."""
    n_docs = len(documents)
    df: dict[str, int] = {}
    for doc in documents:
        unique = set(doc)
        for w in unique:
            df[w] = df.get(w, 0) + 1
    return {w: np.log(n_docs / max(count, 1)) for w, count in df.items()}


def tfidf_keywords(
    text: str, corpus: Sequence[str] | None = None, top_k: int = 20,
) -> list[Keyword]:
    """Extract keywords from text using TF-IDF scoring."""
    tokens = tokenize(text)
    tf = term_frequency(tokens)

    if corpus:
        all_docs = [tokenize(d) for d in corpus]
        idf = idf_scores(all_docs)
    else:
        idf = {w: 1.0 for w in tf}

    scored: list[Keyword] = []
    for word, freq in tf.items():
        score = freq * idf.get(word, 1.0)
        scored.append(Keyword(
            text=word, score=score, frequency=freq,
            is_music_term=word in MUSIC_VOCABULARY,
        ))

    scored.sort(key=lambda k: k.score, reverse=True)
    return scored[:top_k]


def textrank_keywords(text: str, window: int = 4, top_k: int = 20) -> list[Keyword]:
    """TextRank-inspired keyword extraction via co-occurrence graph."""
    tokens = tokenize(text)
    if len(tokens) < 2:
        return []

    vocab = list(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)

    adj = np.zeros((n, n))
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window, len(tokens))):
            a, b = word_to_idx[tokens[i]], word_to_idx[tokens[j]]
            adj[a, b] += 1
            adj[b, a] += 1

    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = adj / row_sums

    d = 0.85
    scores = np.ones(n) / n
    for _ in range(30):
        scores = (1 - d) / n + d * transition.T @ scores

    tf = term_frequency(tokens)
    keywords = []
    for i, word in enumerate(vocab):
        keywords.append(Keyword(
            text=word, score=float(scores[i]), frequency=tf.get(word, 0),
            is_music_term=word in MUSIC_VOCABULARY,
        ))

    keywords.sort(key=lambda k: k.score, reverse=True)
    return keywords[:top_k]


def extract_ngrams(tokens: list[str], n: int = 2) -> dict[tuple[str, ...], int]:
    """Extract n-gram frequencies."""
    ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def music_term_density(text: str) -> float:
    """Proportion of tokens that are recognized music vocabulary."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    music_count = sum(1 for t in tokens if t in MUSIC_VOCABULARY)
    return music_count / len(tokens)
