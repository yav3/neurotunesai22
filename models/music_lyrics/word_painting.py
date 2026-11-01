"""Word painting: mapping lyrical semantics to musical parameters.

Maps words/phrases to pitch contour, dynamics, rhythm, and articulation suggestions
based on semantic categories (motion, emotion, nature, spatial, temporal)."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class SemanticCategory(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    STILLNESS = "stillness"
    FAST = "fast"
    SLOW = "slow"
    LOUD = "loud"
    QUIET = "quiet"
    BRIGHT = "bright"
    DARK = "dark"
    HEAVY = "heavy"
    LIGHT = "light"
    SHARP = "sharp"
    SMOOTH = "smooth"
    JOY = "joy"
    SORROW = "sorrow"
    ANGER = "anger"
    FEAR = "fear"
    LOVE = "love"


WORD_SEMANTICS: Dict[str, SemanticCategory] = {
    "rise": SemanticCategory.ASCENDING,
    "climb": SemanticCategory.ASCENDING,
    "soar": SemanticCategory.ASCENDING,
    "ascend": SemanticCategory.ASCENDING,
    "heaven": SemanticCategory.ASCENDING,
    "sky": SemanticCategory.ASCENDING,
    "up": SemanticCategory.ASCENDING,
    "high": SemanticCategory.ASCENDING,
    "fall": SemanticCategory.DESCENDING,
    "drop": SemanticCategory.DESCENDING,
    "sink": SemanticCategory.DESCENDING,
    "descend": SemanticCategory.DESCENDING,
    "down": SemanticCategory.DESCENDING,
    "low": SemanticCategory.DESCENDING,
    "deep": SemanticCategory.DESCENDING,
    "hell": SemanticCategory.DESCENDING,
    "still": SemanticCategory.STILLNESS,
    "calm": SemanticCategory.STILLNESS,
    "peace": SemanticCategory.STILLNESS,
    "rest": SemanticCategory.STILLNESS,
    "silence": SemanticCategory.STILLNESS,
    "run": SemanticCategory.FAST,
    "rush": SemanticCategory.FAST,
    "fly": SemanticCategory.FAST,
    "quick": SemanticCategory.FAST,
    "fast": SemanticCategory.FAST,
    "slow": SemanticCategory.SLOW,
    "crawl": SemanticCategory.SLOW,
    "linger": SemanticCategory.SLOW,
    "wait": SemanticCategory.SLOW,
    "thunder": SemanticCategory.LOUD,
    "shout": SemanticCategory.LOUD,
    "roar": SemanticCategory.LOUD,
    "scream": SemanticCategory.LOUD,
    "whisper": SemanticCategory.QUIET,
    "murmur": SemanticCategory.QUIET,
    "hush": SemanticCategory.QUIET,
    "soft": SemanticCategory.QUIET,
    "light": SemanticCategory.BRIGHT,
    "shine": SemanticCategory.BRIGHT,
    "sun": SemanticCategory.BRIGHT,
    "bright": SemanticCategory.BRIGHT,
    "gold": SemanticCategory.BRIGHT,
    "dark": SemanticCategory.DARK,
    "shadow": SemanticCategory.DARK,
    "night": SemanticCategory.DARK,
    "black": SemanticCategory.DARK,
    "gloom": SemanticCategory.DARK,
    "heavy": SemanticCategory.HEAVY,
    "burden": SemanticCategory.HEAVY,
    "weight": SemanticCategory.HEAVY,
    "crush": SemanticCategory.HEAVY,
    "feather": SemanticCategory.LIGHT,
    "float": SemanticCategory.LIGHT,
    "gentle": SemanticCategory.LIGHT,
    "joy": SemanticCategory.JOY,
    "happy": SemanticCategory.JOY,
    "laugh": SemanticCategory.JOY,
    "dance": SemanticCategory.JOY,
    "celebrate": SemanticCategory.JOY,
    "sorrow": SemanticCategory.SORROW,
    "cry": SemanticCategory.SORROW,
    "tears": SemanticCategory.SORROW,
    "weep": SemanticCategory.SORROW,
    "mourn": SemanticCategory.SORROW,
    "rage": SemanticCategory.ANGER,
    "fury": SemanticCategory.ANGER,
    "fire": SemanticCategory.ANGER,
    "burn": SemanticCategory.ANGER,
    "fear": SemanticCategory.FEAR,
    "tremble": SemanticCategory.FEAR,
    "dread": SemanticCategory.FEAR,
    "terror": SemanticCategory.FEAR,
    "love": SemanticCategory.LOVE,
    "heart": SemanticCategory.LOVE,
    "embrace": SemanticCategory.LOVE,
    "kiss": SemanticCategory.LOVE,
    "tender": SemanticCategory.LOVE,
}


@dataclass
class PaintingSuggestion:
    """Musical suggestion for a word/phrase based on its semantic category."""
    word: str
    category: SemanticCategory
    pitch_direction: int  # -1 descending, 0 static, +1 ascending
    dynamics_delta: float  # change in velocity/loudness [-1, 1]
    tempo_factor: float  # multiplicative tempo adjustment
    register_preference: str  # "low", "mid", "high"
    articulation: str  # "legato", "staccato", "marcato", "tenuto"
    ornament: Optional[str] = None  # "trill", "mordent", "glissando", None


CATEGORY_MAPPINGS: Dict[SemanticCategory, dict] = {
    SemanticCategory.ASCENDING: dict(pitch=1, dynamics=0.1, tempo=1.0, register="high", artic="legato", ornament="glissando"),
    SemanticCategory.DESCENDING: dict(pitch=-1, dynamics=-0.1, tempo=1.0, register="low", artic="legato", ornament="glissando"),
    SemanticCategory.STILLNESS: dict(pitch=0, dynamics=-0.3, tempo=0.85, register="mid", artic="tenuto", ornament=None),
    SemanticCategory.FAST: dict(pitch=0, dynamics=0.1, tempo=1.3, register="mid", artic="staccato", ornament=None),
    SemanticCategory.SLOW: dict(pitch=0, dynamics=-0.1, tempo=0.7, register="mid", artic="tenuto", ornament=None),
    SemanticCategory.LOUD: dict(pitch=0, dynamics=0.5, tempo=1.05, register="mid", artic="marcato", ornament=None),
    SemanticCategory.QUIET: dict(pitch=0, dynamics=-0.5, tempo=0.95, register="mid", artic="legato", ornament=None),
    SemanticCategory.BRIGHT: dict(pitch=1, dynamics=0.2, tempo=1.05, register="high", artic="staccato", ornament="trill"),
    SemanticCategory.DARK: dict(pitch=-1, dynamics=-0.2, tempo=0.95, register="low", artic="legato", ornament=None),
    SemanticCategory.HEAVY: dict(pitch=-1, dynamics=0.3, tempo=0.8, register="low", artic="marcato", ornament=None),
    SemanticCategory.LIGHT: dict(pitch=1, dynamics=-0.2, tempo=1.1, register="high", artic="staccato", ornament=None),
    SemanticCategory.SHARP: dict(pitch=0, dynamics=0.3, tempo=1.1, register="high", artic="staccato", ornament="mordent"),
    SemanticCategory.SMOOTH: dict(pitch=0, dynamics=-0.1, tempo=0.95, register="mid", artic="legato", ornament=None),
    SemanticCategory.JOY: dict(pitch=1, dynamics=0.3, tempo=1.15, register="high", artic="staccato", ornament="trill"),
    SemanticCategory.SORROW: dict(pitch=-1, dynamics=-0.3, tempo=0.8, register="low", artic="legato", ornament=None),
    SemanticCategory.ANGER: dict(pitch=0, dynamics=0.6, tempo=1.2, register="low", artic="marcato", ornament=None),
    SemanticCategory.FEAR: dict(pitch=1, dynamics=-0.1, tempo=1.3, register="high", artic="staccato", ornament="trill"),
    SemanticCategory.LOVE: dict(pitch=0, dynamics=-0.1, tempo=0.9, register="mid", artic="legato", ornament=None),
}


def classify_word(word: str) -> Optional[SemanticCategory]:
    """Look up a word's semantic category for word painting."""
    return WORD_SEMANTICS.get(word.lower().strip())


def suggest_painting(word: str) -> Optional[PaintingSuggestion]:
    """Generate a musical painting suggestion for a single word."""
    cat = classify_word(word)
    if cat is None:
        return None
    m = CATEGORY_MAPPINGS[cat]
    return PaintingSuggestion(
        word=word,
        category=cat,
        pitch_direction=m["pitch"],
        dynamics_delta=m["dynamics"],
        tempo_factor=m["tempo"],
        register_preference=m["register"],
        articulation=m["artic"],
        ornament=m.get("ornament"),
    )


def paint_line(words: List[str]) -> List[PaintingSuggestion]:
    """Generate painting suggestions for a sequence of words."""
    results = []
    for w in words:
        s = suggest_painting(w)
        if s is not None:
            results.append(s)
    return results


def aggregate_dynamics(suggestions: List[PaintingSuggestion]) -> float:
    """Compute average dynamics delta for a set of painting suggestions."""
    if not suggestions:
        return 0.0
    return sum(s.dynamics_delta for s in suggestions) / len(suggestions)


def aggregate_tempo(suggestions: List[PaintingSuggestion]) -> float:
    """Compute geometric mean of tempo factors."""
    if not suggestions:
        return 1.0
    product = 1.0
    for s in suggestions:
        product *= s.tempo_factor
    return product ** (1.0 / len(suggestions))


def pitch_contour_from_text(words: List[str], base_pitch: int = 60) -> List[int]:
    """Generate a MIDI pitch contour guided by word painting semantics.

    Non-semantic words hold the previous pitch.
    """
    contour = []
    current = base_pitch
    for w in words:
        s = suggest_painting(w)
        if s is not None:
            step = 2 * s.pitch_direction
            if s.register_preference == "high" and current < 72:
                step += 1
            elif s.register_preference == "low" and current > 48:
                step -= 1
            current = max(24, min(108, current + step))
        contour.append(current)
    return contour


def word_painting_density(words: List[str]) -> float:
    """Fraction of words that trigger a painting suggestion."""
    if not words:
        return 0.0
    return sum(1 for w in words if classify_word(w) is not None) / len(words)


def semantic_arc(words: List[str]) -> List[float]:
    """Compute a continuous semantic "intensity" arc over a word sequence.

    Maps each word to abs(dynamics_delta) as a proxy for semantic weight.
    """
    arc = []
    for w in words:
        s = suggest_painting(w)
        arc.append(abs(s.dynamics_delta) if s else 0.0)
    return arc
