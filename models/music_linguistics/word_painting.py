"""Word painting: mapping semantic content to musical gestures (madrigalism)."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


DIRECTION_WORDS = {
    "up": 1, "rise": 1, "ascend": 1, "climb": 1, "soar": 1, "fly": 1,
    "heaven": 1, "high": 1, "above": 1, "sky": 1, "peak": 1, "lift": 1,
    "down": -1, "fall": -1, "descend": -1, "drop": -1, "sink": -1,
    "low": -1, "below": -1, "deep": -1, "ground": -1, "grave": -1,
    "hell": -1, "abyss": -1, "plunge": -1, "dive": -1,
}

SPEED_WORDS = {
    "fast": 1.3, "quick": 1.2, "rapid": 1.3, "rush": 1.4, "hurry": 1.3,
    "run": 1.2, "sprint": 1.5, "flee": 1.3, "chase": 1.2,
    "slow": 0.7, "still": 0.5, "calm": 0.6, "rest": 0.4, "sleep": 0.3,
    "pause": 0.5, "wait": 0.6, "linger": 0.5, "freeze": 0.3,
    "stop": 0.2, "death": 0.1, "silence": 0.0,
}

TEXTURE_WORDS = {
    "alone": 1, "solo": 1, "single": 1, "one": 1, "lonely": 1,
    "together": 4, "all": 4, "crowd": 4, "chorus": 4, "many": 3,
    "pair": 2, "two": 2, "couple": 2, "duet": 2,
    "three": 3, "trio": 3, "triple": 3,
}

DYNAMIC_WORDS = {
    "loud": 0.9, "shout": 0.95, "thunder": 1.0, "roar": 0.95,
    "scream": 1.0, "cry": 0.8, "strong": 0.8, "power": 0.85,
    "soft": 0.3, "whisper": 0.15, "quiet": 0.2, "gentle": 0.35,
    "tender": 0.3, "murmur": 0.2, "sigh": 0.25, "hush": 0.1,
}

RHYTHMIC_WORDS = {
    "dance": "dotted", "waltz": "triple", "march": "duple",
    "swing": "swing", "sway": "compound", "skip": "dotted",
    "heartbeat": "duple", "pulse": "steady", "tick": "steady",
    "stutter": "irregular", "stammer": "irregular", "chaos": "irregular",
}


@dataclass
class PaintingGesture:
    word: str
    category: str
    direction: Optional[int] = None
    tempo_factor: Optional[float] = None
    voices: Optional[int] = None
    dynamic_level: Optional[float] = None
    rhythm_type: Optional[str] = None
    confidence: float = 1.0


@dataclass
class PaintingSuggestion:
    text: str
    gestures: List[PaintingGesture]
    melodic_contour: Optional[np.ndarray] = None
    dynamic_curve: Optional[np.ndarray] = None
    density_curve: Optional[np.ndarray] = None


def detect_painting_words(text: str) -> List[PaintingGesture]:
    words = text.lower().split()
    gestures = []
    for w in words:
        w_clean = w.strip(",.!?;:'\"()-")
        if w_clean in DIRECTION_WORDS:
            gestures.append(PaintingGesture(
                word=w_clean, category="direction",
                direction=DIRECTION_WORDS[w_clean], confidence=0.9,
            ))
        if w_clean in SPEED_WORDS:
            gestures.append(PaintingGesture(
                word=w_clean, category="speed",
                tempo_factor=SPEED_WORDS[w_clean], confidence=0.85,
            ))
        if w_clean in TEXTURE_WORDS:
            gestures.append(PaintingGesture(
                word=w_clean, category="texture",
                voices=TEXTURE_WORDS[w_clean], confidence=0.8,
            ))
        if w_clean in DYNAMIC_WORDS:
            gestures.append(PaintingGesture(
                word=w_clean, category="dynamic",
                dynamic_level=DYNAMIC_WORDS[w_clean], confidence=0.85,
            ))
        if w_clean in RHYTHMIC_WORDS:
            gestures.append(PaintingGesture(
                word=w_clean, category="rhythm",
                rhythm_type=RHYTHMIC_WORDS[w_clean], confidence=0.75,
            ))
    return gestures


def direction_to_contour(gestures: List[PaintingGesture],
                         n_notes: int = 8, step_size: int = 2) -> np.ndarray:
    contour = np.zeros(n_notes)
    pos = 0
    direction_gestures = [g for g in gestures if g.category == "direction"]
    if not direction_gestures:
        return contour
    per_gesture = max(1, n_notes // len(direction_gestures))
    idx = 0
    for g in direction_gestures:
        d = g.direction or 0
        for j in range(per_gesture):
            if idx < n_notes:
                pos += d * step_size
                contour[idx] = pos
                idx += 1
    while idx < n_notes:
        contour[idx] = pos
        idx += 1
    return contour


def dynamic_curve_from_gestures(gestures: List[PaintingGesture],
                                 n_steps: int = 16) -> np.ndarray:
    curve = np.full(n_steps, 0.5)
    dyn_gestures = [g for g in gestures if g.category == "dynamic"]
    if not dyn_gestures:
        return curve
    per_gesture = max(1, n_steps // len(dyn_gestures))
    idx = 0
    for g in dyn_gestures:
        level = g.dynamic_level or 0.5
        for j in range(per_gesture):
            if idx < n_steps:
                curve[idx] = level
                idx += 1
    while idx < n_steps:
        curve[idx] = curve[idx - 1] if idx > 0 else 0.5
        idx += 1
    return curve


def suggest_painting(text: str, n_notes: int = 16) -> PaintingSuggestion:
    gestures = detect_painting_words(text)
    melodic = direction_to_contour(gestures, n_notes) if gestures else None
    dynamics = dynamic_curve_from_gestures(gestures, n_notes)
    density = np.ones(n_notes)
    for g in gestures:
        if g.category == "texture" and g.voices is not None:
            density[:] = g.voices
    return PaintingSuggestion(
        text=text, gestures=gestures,
        melodic_contour=melodic, dynamic_curve=dynamics,
        density_curve=density,
    )


def painting_density(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    all_lexicons = set(DIRECTION_WORDS) | set(SPEED_WORDS) | set(TEXTURE_WORDS) | \
                   set(DYNAMIC_WORDS) | set(RHYTHMIC_WORDS)
    count = sum(1 for w in words if w.strip(",.!?;:'\"()-") in all_lexicons)
    return count / len(words)


def melodic_interval_from_direction(direction: int, intensity: float = 1.0) -> int:
    base = 2 if intensity < 0.5 else 4 if intensity < 0.8 else 7
    return direction * base


def tempo_adjustment(gestures: List[PaintingGesture]) -> float:
    speed_gestures = [g for g in gestures if g.category == "speed" and g.tempo_factor is not None]
    if not speed_gestures:
        return 1.0
    return float(np.mean([g.tempo_factor for g in speed_gestures]))


@dataclass
class RepetitionPainting:
    word: str
    count: int
    category: str = "repetition"


def detect_word_repetition(text: str, min_count: int = 3) -> List[RepetitionPainting]:
    words = text.lower().split()
    cleaned = [w.strip(",.!?;:'\"()-") for w in words]
    counts: Dict[str, int] = {}
    for w in cleaned:
        if len(w) > 2:
            counts[w] = counts.get(w, 0) + 1
    return [RepetitionPainting(word=w, count=c) for w, c in counts.items() if c >= min_count]


def onomatopoeia_gestures(text: str) -> List[PaintingGesture]:
    ONOMATOPOEIA = {
        "bang": ("dynamic", 1.0), "crash": ("dynamic", 1.0),
        "boom": ("dynamic", 0.95), "pop": ("dynamic", 0.7),
        "click": ("rhythm", 0.4), "tick": ("rhythm", 0.3),
        "buzz": ("texture", 0.5), "hum": ("texture", 0.3),
        "splash": ("texture", 0.7), "drip": ("rhythm", 0.3),
        "whistle": ("direction", 0.6), "hiss": ("texture", 0.4),
        "ring": ("texture", 0.5), "clap": ("rhythm", 0.8),
    }
    gestures = []
    for w in text.lower().split():
        w_clean = w.strip(",.!?;:'\"()-")
        if w_clean in ONOMATOPOEIA:
            cat, level = ONOMATOPOEIA[w_clean]
            gestures.append(PaintingGesture(
                word=w_clean, category=cat,
                dynamic_level=level, confidence=0.7,
            ))
    return gestures
