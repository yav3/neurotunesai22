"""Musical concept embedding: vector space for musical ideas, analogy, clustering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

EMBEDDING_DIM = 64

CONCEPT_CATEGORIES = [
    "pitch", "rhythm", "harmony", "timbre", "dynamics",
    "form", "texture", "articulation", "expression", "genre",
]

_SEED = 42


def _deterministic_vector(name: str, category: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    seed = hash((name, category)) % (2 ** 31)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    cat_idx = CONCEPT_CATEGORIES.index(category) if category in CONCEPT_CATEGORIES else 0
    v[cat_idx * (dim // len(CONCEPT_CATEGORIES)) : (cat_idx + 1) * (dim // len(CONCEPT_CATEGORIES))] *= 2.0
    return v / (np.linalg.norm(v) + 1e-12)


@dataclass(frozen=True)
class MusicalConcept:
    name: str
    category: str
    description: str
    vector: np.ndarray = field(repr=False, compare=False)


CONCEPT_REGISTRY: dict[str, MusicalConcept] = {}

_CONCEPT_DEFS: list[tuple[str, str, str]] = [
    ("melody", "pitch", "A sequence of single pitches forming a musical line"),
    ("interval", "pitch", "Distance between two pitches"),
    ("scale", "pitch", "Ordered set of pitches within an octave"),
    ("beat", "rhythm", "Basic unit of musical time"),
    ("syncopation", "rhythm", "Rhythmic emphasis on weak beats"),
    ("polyrhythm", "rhythm", "Simultaneous use of conflicting rhythms"),
    ("chord", "harmony", "Three or more pitches sounding together"),
    ("cadence", "harmony", "Harmonic progression signalling a phrase ending"),
    ("modulation", "harmony", "Change of key within a piece"),
    ("voice", "timbre", "Characteristic sound quality of a source"),
    ("orchestration", "timbre", "Assignment of instruments to musical parts"),
    ("distortion", "timbre", "Nonlinear alteration of a sound wave"),
    ("crescendo", "dynamics", "Gradual increase in loudness"),
    ("accent", "dynamics", "Emphasis on a particular note"),
    ("pianissimo", "dynamics", "Very soft dynamic level"),
    ("verse", "form", "A recurring section with the same melody but different lyrics"),
    ("chorus", "form", "The hook section that repeats with the same words"),
    ("bridge", "form", "A contrasting section providing variety"),
    ("counterpoint", "texture", "Independent melodic lines sounding together"),
    ("monophony", "texture", "A single unaccompanied melodic line"),
    ("homophony", "texture", "A melody with chordal accompaniment"),
    ("legato", "articulation", "Smooth, connected playing"),
    ("staccato", "articulation", "Short, detached notes"),
    ("vibrato", "articulation", "Periodic variation in pitch"),
    ("rubato", "expression", "Flexible tempo for expressive effect"),
    ("fermata", "expression", "A held note or pause"),
    ("glissando", "expression", "A continuous slide between pitches"),
    ("blues", "genre", "African-American origin genre with blue notes and 12-bar form"),
    ("jazz", "genre", "Improvisatory genre with swing and complex harmony"),
    ("classical", "genre", "Western art music tradition"),
    ("electronic", "genre", "Music produced primarily with electronic instruments"),
    ("folk", "genre", "Traditional music passed through oral tradition"),
]

for _name, _cat, _desc in _CONCEPT_DEFS:
    CONCEPT_REGISTRY[_name] = MusicalConcept(
        name=_name, category=_cat, description=_desc,
        vector=_deterministic_vector(_name, _cat),
    )


def concept_similarity(a: MusicalConcept, b: MusicalConcept) -> float:
    dot = float(np.dot(a.vector, b.vector))
    na = float(np.linalg.norm(a.vector))
    nb = float(np.linalg.norm(b.vector))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def nearest_concepts(
    query: np.ndarray | MusicalConcept, k: int = 5, exclude: Sequence[str] = (),
) -> list[tuple[str, float]]:
    if isinstance(query, MusicalConcept):
        qv = query.vector
    else:
        qv = query
    qn = np.linalg.norm(qv)
    if qn < 1e-12:
        return []
    results: list[tuple[str, float]] = []
    for name, concept in CONCEPT_REGISTRY.items():
        if name in exclude:
            continue
        cn = np.linalg.norm(concept.vector)
        if cn < 1e-12:
            continue
        sim = float(np.dot(qv, concept.vector) / (qn * cn))
        results.append((name, sim))
    results.sort(key=lambda x: -x[1])
    return results[:k]


def analogy(a: str, b: str, c: str, k: int = 3) -> list[tuple[str, float]]:
    if a not in CONCEPT_REGISTRY or b not in CONCEPT_REGISTRY or c not in CONCEPT_REGISTRY:
        return []
    va = CONCEPT_REGISTRY[a].vector
    vb = CONCEPT_REGISTRY[b].vector
    vc = CONCEPT_REGISTRY[c].vector
    target = vb - va + vc
    return nearest_concepts(target, k=k, exclude=(a, b, c))


class ConceptEmbedding:
    def __init__(self) -> None:
        self.concepts: dict[str, MusicalConcept] = dict(CONCEPT_REGISTRY)

    def add(self, name: str, category: str, description: str,
            vector: np.ndarray | None = None) -> MusicalConcept:
        if vector is None:
            vector = _deterministic_vector(name, category)
        else:
            n = np.linalg.norm(vector)
            if n > 1e-12:
                vector = vector / n
        c = MusicalConcept(name=name, category=category, description=description, vector=vector)
        self.concepts[name] = c
        return c

    def get(self, name: str) -> MusicalConcept | None:
        return self.concepts.get(name)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        qn = np.linalg.norm(query_vector)
        if qn < 1e-12:
            return []
        results: list[tuple[str, float]] = []
        for name, concept in self.concepts.items():
            cn = np.linalg.norm(concept.vector)
            if cn < 1e-12:
                continue
            sim = float(np.dot(query_vector, concept.vector) / (qn * cn))
            results.append((name, sim))
        results.sort(key=lambda x: -x[1])
        return results[:k]

    def cluster(self, k: int = 5, max_iter: int = 50) -> dict[int, list[str]]:
        names = list(self.concepts.keys())
        if len(names) == 0:
            return {}
        vecs = np.array([self.concepts[n].vector for n in names])
        rng = np.random.RandomState(_SEED)
        indices = rng.choice(len(names), size=min(k, len(names)), replace=False)
        centroids = vecs[indices].copy()
        labels = np.zeros(len(names), dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(vecs[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(len(centroids)):
                mask = labels == ci
                if mask.any():
                    centroids[ci] = vecs[mask].mean(axis=0)
        clusters: dict[int, list[str]] = {}
        for i, name in enumerate(names):
            cl = int(labels[i])
            clusters.setdefault(cl, []).append(name)
        return clusters
