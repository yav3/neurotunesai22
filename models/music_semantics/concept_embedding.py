"""Musical concept embedding: dense vector space for musical concepts.

Maps musical concepts (keys, modes, intervals, chords, genres, emotions)
to a shared embedding space where semantic similarity ≈ cosine distance.
Uses hand-crafted seed embeddings from music-theoretic relationships
(circle of fifths, interval consonance, genre taxonomy distance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EMBEDDING_DIM = 32

CONCEPT_CATEGORIES = [
    "pitch_class", "interval", "chord_quality", "scale_mode",
    "genre", "emotion", "tempo_class", "dynamic", "articulation",
    "texture", "form", "instrument_family",
]

_RNG = np.random.RandomState(42)


@dataclass
class MusicalConcept:
    name: str
    category: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_INTERVALS = [
    "unison", "minor_2nd", "major_2nd", "minor_3rd", "major_3rd",
    "perfect_4th", "tritone", "perfect_5th", "minor_6th", "major_6th",
    "minor_7th", "major_7th", "octave",
]
_CHORD_QUALITIES = ["major", "minor", "diminished", "augmented", "dominant7", "major7", "minor7"]
_SCALE_MODES = ["ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]
_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
_TEMPO_CLASSES = ["grave", "largo", "adagio", "andante", "moderato", "allegro", "vivace", "presto"]
_DYNAMICS = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]


def _circle_of_fifths_embedding(note_idx: int, dim: int) -> np.ndarray:
    fifths_pos = (note_idx * 7) % 12
    angle = 2 * np.pi * fifths_pos / 12
    emb = np.zeros(dim)
    emb[0] = np.cos(angle)
    emb[1] = np.sin(angle)
    emb[2] = np.cos(2 * angle)
    emb[3] = np.sin(2 * angle)
    seed = _RNG.RandomState(note_idx + 100)
    emb[4:] = seed.randn(dim - 4) * 0.1
    return emb / (np.linalg.norm(emb) + 1e-12)


def _interval_embedding(semitones: int, dim: int) -> np.ndarray:
    consonance = {0: 1.0, 7: 0.95, 5: 0.9, 4: 0.85, 3: 0.8, 8: 0.75,
                  9: 0.7, 2: 0.6, 10: 0.55, 1: 0.3, 11: 0.3, 6: 0.2, 12: 1.0}
    c = consonance.get(semitones, 0.5)
    emb = np.zeros(dim)
    emb[0] = c
    emb[1] = semitones / 12.0
    emb[2] = np.sin(np.pi * semitones / 12)
    emb[3] = np.cos(np.pi * semitones / 12)
    seed = _RNG.RandomState(semitones + 200)
    emb[4:] = seed.randn(dim - 4) * 0.1
    return emb / (np.linalg.norm(emb) + 1e-12)


def _ordinal_embedding(idx: int, total: int, dim: int, seed_offset: int) -> np.ndarray:
    emb = np.zeros(dim)
    t = idx / max(total - 1, 1)
    emb[0] = t
    emb[1] = np.sin(np.pi * t)
    emb[2] = np.cos(np.pi * t)
    seed = _RNG.RandomState(idx + seed_offset)
    emb[3:] = seed.randn(dim - 3) * 0.1
    return emb / (np.linalg.norm(emb) + 1e-12)


def _build_registry() -> dict[str, MusicalConcept]:
    reg: dict[str, MusicalConcept] = {}
    for i, name in enumerate(_PITCH_CLASSES):
        reg[f"pitch:{name}"] = MusicalConcept(name, "pitch_class", f"Pitch class {name}",
                                               {"semitone": i, "fifths_pos": (i * 7) % 12})
    for i, name in enumerate(_INTERVALS):
        reg[f"interval:{name}"] = MusicalConcept(name, "interval", f"Interval of {i} semitones",
                                                  {"semitones": i})
    for name in _CHORD_QUALITIES:
        reg[f"chord:{name}"] = MusicalConcept(name, "chord_quality", f"{name} chord quality")
    for i, name in enumerate(_SCALE_MODES):
        reg[f"mode:{name}"] = MusicalConcept(name, "scale_mode", f"{name} mode (degree {i+1})",
                                              {"degree": i + 1})
    for name in _EMOTIONS:
        reg[f"emotion:{name}"] = MusicalConcept(name, "emotion", f"Plutchik emotion: {name}")
    for i, name in enumerate(_TEMPO_CLASSES):
        reg[f"tempo:{name}"] = MusicalConcept(name, "tempo_class", f"Tempo class: {name}",
                                               {"rank": i})
    for i, name in enumerate(_DYNAMICS):
        reg[f"dynamic:{name}"] = MusicalConcept(name, "dynamic", f"Dynamic level: {name}",
                                                 {"rank": i})
    return reg


CONCEPT_REGISTRY = _build_registry()


class ConceptEmbedding:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self._embeddings: dict[str, np.ndarray] = {}
        self._build_seed_embeddings()

    def _build_seed_embeddings(self) -> None:
        for i, name in enumerate(_PITCH_CLASSES):
            self._embeddings[f"pitch:{name}"] = _circle_of_fifths_embedding(i, self.dim)
        for i in range(len(_INTERVALS)):
            self._embeddings[f"interval:{_INTERVALS[i]}"] = _interval_embedding(i, self.dim)
        for i, name in enumerate(_CHORD_QUALITIES):
            self._embeddings[f"chord:{name}"] = _ordinal_embedding(i, len(_CHORD_QUALITIES),
                                                                     self.dim, 300)
        for i, name in enumerate(_SCALE_MODES):
            self._embeddings[f"mode:{name}"] = _ordinal_embedding(i, len(_SCALE_MODES),
                                                                    self.dim, 400)
        for i, name in enumerate(_EMOTIONS):
            self._embeddings[f"emotion:{name}"] = _ordinal_embedding(i, len(_EMOTIONS),
                                                                      self.dim, 500)
        for i, name in enumerate(_TEMPO_CLASSES):
            self._embeddings[f"tempo:{name}"] = _ordinal_embedding(i, len(_TEMPO_CLASSES),
                                                                     self.dim, 600)
        for i, name in enumerate(_DYNAMICS):
            self._embeddings[f"dynamic:{name}"] = _ordinal_embedding(i, len(_DYNAMICS),
                                                                      self.dim, 700)

    def embed(self, concept_key: str) -> np.ndarray:
        if concept_key in self._embeddings:
            return self._embeddings[concept_key].copy()
        seed = hash(concept_key) % (2**31)
        rng = np.random.RandomState(seed)
        emb = rng.randn(self.dim)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        self._embeddings[concept_key] = emb
        return emb.copy()

    def embed_batch(self, keys: list[str]) -> np.ndarray:
        return np.array([self.embed(k) for k in keys])

    @property
    def vocabulary(self) -> list[str]:
        return list(self._embeddings.keys())


def concept_similarity(emb: ConceptEmbedding, key_a: str, key_b: str) -> float:
    a = emb.embed(key_a)
    b = emb.embed(key_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def nearest_concepts(emb: ConceptEmbedding, query_key: str, k: int = 5) -> list[tuple[str, float]]:
    q = emb.embed(query_key)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    results = []
    for key in emb.vocabulary:
        if key == query_key:
            continue
        v = emb.embed(key)
        sim = float(np.dot(q_norm, v / (np.linalg.norm(v) + 1e-12)))
        results.append((key, sim))
    results.sort(key=lambda x: -x[1])
    return results[:k]


def analogy(emb: ConceptEmbedding, a: str, b: str, c: str, k: int = 5) -> list[tuple[str, float]]:
    va = emb.embed(a)
    vb = emb.embed(b)
    vc = emb.embed(c)
    target = vb - va + vc
    target = target / (np.linalg.norm(target) + 1e-12)
    exclude = {a, b, c}
    results = []
    for key in emb.vocabulary:
        if key in exclude:
            continue
        v = emb.embed(key)
        sim = float(np.dot(target, v / (np.linalg.norm(v) + 1e-12)))
        results.append((key, sim))
    results.sort(key=lambda x: -x[1])
    return results[:k]
