"""Lyric style transfer: vocabulary, syntax, and tone adaptation.

Transforms text to match a target genre/era/artist style via statistical models.
Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
import re

import numpy as np


class StyleDimension(Enum):
    FORMALITY = "formality"
    EMOTIONALITY = "emotionality"
    CONCRETENESS = "concreteness"
    COMPLEXITY = "complexity"
    REPETITIVENESS = "repetitiveness"


@dataclass
class StyleProfile:
    name: str
    formality: float
    emotionality: float
    concreteness: float
    complexity: float
    repetitiveness: float
    vocabulary_size_ratio: float = 0.0
    avg_word_length: float = 0.0
    avg_line_length: float = 0.0


GENRE_PROFILES: dict[str, StyleProfile] = {
    "pop": StyleProfile("pop", 0.3, 0.7, 0.6, 0.3, 0.8, 0.4, 4.2, 6.0),
    "hip_hop": StyleProfile("hip_hop", 0.2, 0.6, 0.7, 0.5, 0.5, 0.6, 4.5, 10.0),
    "country": StyleProfile("country", 0.4, 0.6, 0.8, 0.3, 0.6, 0.5, 4.0, 7.0),
    "rock": StyleProfile("rock", 0.3, 0.7, 0.5, 0.4, 0.6, 0.5, 4.3, 7.0),
    "folk": StyleProfile("folk", 0.5, 0.5, 0.7, 0.4, 0.4, 0.5, 4.1, 8.0),
    "r_and_b": StyleProfile("r_and_b", 0.3, 0.8, 0.5, 0.3, 0.7, 0.4, 4.0, 6.5),
    "classical_art_song": StyleProfile("classical_art_song", 0.8, 0.6, 0.4, 0.7, 0.3, 0.7, 5.5, 9.0),
    "punk": StyleProfile("punk", 0.1, 0.8, 0.6, 0.2, 0.7, 0.3, 3.8, 5.0),
    "metal": StyleProfile("metal", 0.3, 0.8, 0.5, 0.5, 0.5, 0.6, 4.8, 7.5),
    "jazz": StyleProfile("jazz", 0.4, 0.7, 0.5, 0.5, 0.4, 0.5, 4.5, 7.0),
}


FORMAL_SUBSTITUTIONS: dict[str, str] = {
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "ain't": "is not", "y'all": "you all", "'cause": "because",
    "yeah": "yes", "nah": "no", "kinda": "kind of",
    "bout": "about", "em": "them",
}

INFORMAL_SUBSTITUTIONS: dict[str, str] = {v: k for k, v in FORMAL_SUBSTITUTIONS.items()}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+|[.,!?;:]", text.lower())


def measure_formality(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.5
    informal_count = sum(1 for t in tokens if t in FORMAL_SUBSTITUTIONS)
    return max(0.0, 1.0 - informal_count / max(len(tokens), 1))


def measure_emotionality(text: str) -> float:
    emotion_words = {
        "love", "hate", "cry", "scream", "feel", "hurt", "pain", "joy",
        "fear", "anger", "happy", "sad", "lonely", "broken", "desire",
        "passion", "rage", "tears", "heart", "soul", "bleed", "ache",
    }
    tokens = set(_tokenize(text))
    return min(1.0, len(tokens & emotion_words) * 0.15)


def measure_concreteness(text: str) -> float:
    concrete_words = {
        "rain", "sun", "road", "car", "house", "door", "hand", "eye",
        "river", "mountain", "fire", "smoke", "blood", "gun", "knife",
        "flower", "tree", "stone", "glass", "mirror", "clock", "phone",
    }
    tokens = set(_tokenize(text))
    return min(1.0, len(tokens & concrete_words) * 0.12)


def measure_complexity(text: str) -> float:
    tokens = [t for t in _tokenize(text) if t.isalpha()]
    if not tokens:
        return 0.0
    avg_len = np.mean([len(t) for t in tokens])
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    return float(np.clip((avg_len - 3) / 5 * 0.5 + unique_ratio * 0.5, 0, 1))


def measure_repetitiveness(lines: Sequence[str]) -> float:
    if len(lines) <= 1:
        return 0.0
    normalized = [l.lower().strip() for l in lines]
    unique = set(normalized)
    return 1.0 - len(unique) / len(normalized)


def compute_style_profile(lines: Sequence[str], name: str = "unknown") -> StyleProfile:
    """Compute a StyleProfile from lyric lines."""
    full_text = " ".join(lines)
    tokens = [t for t in _tokenize(full_text) if t.isalpha()]
    return StyleProfile(
        name=name,
        formality=measure_formality(full_text),
        emotionality=measure_emotionality(full_text),
        concreteness=measure_concreteness(full_text),
        complexity=measure_complexity(full_text),
        repetitiveness=measure_repetitiveness(lines),
        vocabulary_size_ratio=len(set(tokens)) / max(len(tokens), 1) if tokens else 0.0,
        avg_word_length=float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        avg_line_length=float(np.mean([len(l.split()) for l in lines])) if lines else 0.0,
    )


def style_distance(a: StyleProfile, b: StyleProfile) -> float:
    """Euclidean distance between two style profiles on the 5 core dimensions."""
    va = np.array([a.formality, a.emotionality, a.concreteness, a.complexity, a.repetitiveness])
    vb = np.array([b.formality, b.emotionality, b.concreteness, b.complexity, b.repetitiveness])
    return float(np.linalg.norm(va - vb))


def closest_genre(profile: StyleProfile) -> tuple[str, float]:
    """Find the closest genre profile and its distance."""
    best_genre = ""
    best_dist = float("inf")
    for name, gp in GENRE_PROFILES.items():
        d = style_distance(profile, gp)
        if d < best_dist:
            best_dist = d
            best_genre = name
    return best_genre, best_dist


def formality_shift(text: str, direction: float) -> str:
    """Shift text formality. direction > 0 = more formal, < 0 = more informal."""
    tokens = text.split()
    result = []
    subs = FORMAL_SUBSTITUTIONS if direction > 0 else INFORMAL_SUBSTITUTIONS
    for tok in tokens:
        key = tok.lower().strip(".,!?;:'\"")
        if key in subs and np.random.random() < abs(direction):
            result.append(subs[key])
        else:
            result.append(tok)
    return " ".join(result)
