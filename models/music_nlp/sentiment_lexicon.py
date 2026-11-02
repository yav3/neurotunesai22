"""Music-domain sentiment lexicon and scoring.

Hand-curated valence/arousal/dominance lexicon for music-specific terms,
with context-aware scoring for lyrics, reviews, and music descriptions.
Based on Warriner et al. (2013) norms and Russell (1980) circumplex.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LexiconEntry:
    word: str
    valence: float      # [-1, 1] negative-positive
    arousal: float      # [-1, 1] calm-excited
    dominance: float    # [-1, 1] submissive-dominant
    music_domain: bool = False


# Music-specific terms not in standard sentiment lexicons
MUSIC_LEXICON: Dict[str, LexiconEntry] = {
    # Positive-high-arousal
    "crescendo": LexiconEntry("crescendo", 0.6, 0.8, 0.7, True),
    "fortissimo": LexiconEntry("fortissimo", 0.3, 0.9, 0.8, True),
    "virtuoso": LexiconEntry("virtuoso", 0.8, 0.6, 0.9, True),
    "anthem": LexiconEntry("anthem", 0.7, 0.7, 0.8, True),
    "groove": LexiconEntry("groove", 0.7, 0.6, 0.5, True),
    "riff": LexiconEntry("riff", 0.5, 0.6, 0.5, True),
    "soaring": LexiconEntry("soaring", 0.8, 0.7, 0.6, True),
    "driving": LexiconEntry("driving", 0.4, 0.8, 0.7, True),
    "infectious": LexiconEntry("infectious", 0.6, 0.6, 0.4, True),
    "electrifying": LexiconEntry("electrifying", 0.8, 0.9, 0.7, True),
    # Positive-low-arousal
    "lullaby": LexiconEntry("lullaby", 0.6, -0.7, -0.3, True),
    "serenade": LexiconEntry("serenade", 0.7, -0.4, 0.2, True),
    "adagio": LexiconEntry("adagio", 0.3, -0.6, 0.0, True),
    "legato": LexiconEntry("legato", 0.4, -0.3, 0.1, True),
    "pianissimo": LexiconEntry("pianissimo", 0.2, -0.7, -0.3, True),
    "ethereal": LexiconEntry("ethereal", 0.6, -0.3, -0.1, True),
    "delicate": LexiconEntry("delicate", 0.5, -0.5, -0.2, True),
    "tender": LexiconEntry("tender", 0.6, -0.4, -0.1, True),
    # Negative-high-arousal
    "dissonance": LexiconEntry("dissonance", -0.4, 0.5, 0.1, True),
    "cacophony": LexiconEntry("cacophony", -0.7, 0.8, -0.2, True),
    "discordant": LexiconEntry("discordant", -0.5, 0.4, -0.1, True),
    "jarring": LexiconEntry("jarring", -0.5, 0.7, 0.0, True),
    "shrill": LexiconEntry("shrill", -0.5, 0.6, -0.2, True),
    "screeching": LexiconEntry("screeching", -0.6, 0.8, -0.3, True),
    # Negative-low-arousal
    "dirge": LexiconEntry("dirge", -0.6, -0.5, -0.4, True),
    "monotonous": LexiconEntry("monotonous", -0.5, -0.6, -0.3, True),
    "droning": LexiconEntry("droning", -0.3, -0.4, -0.2, True),
    "flat": LexiconEntry("flat", -0.4, -0.3, -0.2, True),
    "tuneless": LexiconEntry("tuneless", -0.6, -0.2, -0.3, True),
    # Neutral/ambiguous
    "syncopation": LexiconEntry("syncopation", 0.2, 0.4, 0.3, True),
    "modulation": LexiconEntry("modulation", 0.1, 0.2, 0.2, True),
    "cadence": LexiconEntry("cadence", 0.2, -0.1, 0.3, True),
    "ostinato": LexiconEntry("ostinato", 0.0, 0.3, 0.2, True),
    "rubato": LexiconEntry("rubato", 0.3, -0.2, 0.4, True),
    "tremolo": LexiconEntry("tremolo", 0.1, 0.4, 0.1, True),
    "staccato": LexiconEntry("staccato", 0.1, 0.5, 0.3, True),
    "pizzicato": LexiconEntry("pizzicato", 0.2, 0.2, 0.2, True),
    "glissando": LexiconEntry("glissando", 0.3, 0.4, 0.2, True),
    "arpeggio": LexiconEntry("arpeggio", 0.4, 0.3, 0.2, True),
    "vibrato": LexiconEntry("vibrato", 0.3, 0.2, 0.3, True),
}

# Standard intensifiers/negators
NEGATORS = {"not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely"}
INTENSIFIERS = {"very", "extremely", "incredibly", "remarkably", "absolutely", "truly"}
DIMINISHERS = {"somewhat", "slightly", "rather", "fairly", "quite", "a bit"}


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z']+", text.lower())


def score_token(token: str) -> Optional[LexiconEntry]:
    return MUSIC_LEXICON.get(token)


def score_text(text: str, window: int = 3) -> Tuple[float, float, float]:
    """Score text for (valence, arousal, dominance) using context window.

    Handles negation and intensification within a sliding window.
    Returns mean scores across all matched lexicon tokens.
    """
    tokens = tokenize(text)
    scores_v, scores_a, scores_d = [], [], []

    for i, token in enumerate(tokens):
        entry = score_token(token)
        if entry is None:
            continue

        v, a, d = entry.valence, entry.arousal, entry.dominance
        context = tokens[max(0, i - window):i]

        has_negation = any(t in NEGATORS for t in context)
        has_intensifier = any(t in INTENSIFIERS for t in context)
        has_diminisher = any(t in DIMINISHERS for t in context)

        if has_negation:
            v = -v * 0.8
        if has_intensifier:
            v *= 1.3
            a *= 1.3
        if has_diminisher:
            v *= 0.7
            a *= 0.7

        scores_v.append(np.clip(v, -1, 1))
        scores_a.append(np.clip(a, -1, 1))
        scores_d.append(np.clip(d, -1, 1))

    if not scores_v:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(scores_v)),
        float(np.mean(scores_a)),
        float(np.mean(scores_d)),
    )


def sentiment_trajectory(texts: List[str]) -> np.ndarray:
    """Compute VAD trajectory over a sequence of text segments.

    Returns (N, 3) array of [valence, arousal, dominance] per segment.
    """
    trajectory = np.zeros((len(texts), 3))
    for i, text in enumerate(texts):
        v, a, d = score_text(text)
        trajectory[i] = [v, a, d]
    return trajectory


def emotional_arc(texts: List[str], smooth_sigma: float = 1.0) -> np.ndarray:
    """Gaussian-smoothed emotional arc (valence only) over text segments."""
    from scipy.ndimage import gaussian_filter1d
    traj = sentiment_trajectory(texts)
    valence = traj[:, 0]
    if len(valence) < 3:
        return valence
    return gaussian_filter1d(valence, smooth_sigma)


def lexicon_coverage(text: str) -> float:
    """Fraction of tokens found in the music sentiment lexicon."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    matched = sum(1 for t in tokens if t in MUSIC_LEXICON)
    return matched / len(tokens)
