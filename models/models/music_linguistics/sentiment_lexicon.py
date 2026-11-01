"""Sentiment lexicon: music-domain sentiment analysis for lyrics and reviews.

Provides a curated music-domain sentiment lexicon, phrase-level sentiment
scoring, and emotion-valence mapping tuned for musical vocabulary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SentimentScore:
    valence: float
    arousal: float
    dominance: float

    @property
    def polarity(self) -> str:
        if self.valence > 0.1:
            return "positive"
        if self.valence < -0.1:
            return "negative"
        return "neutral"


MUSIC_SENTIMENT_LEXICON: dict[str, SentimentScore] = {
    "love": SentimentScore(0.85, 0.6, 0.5),
    "joy": SentimentScore(0.9, 0.8, 0.6),
    "happy": SentimentScore(0.85, 0.7, 0.55),
    "dance": SentimentScore(0.7, 0.85, 0.6),
    "free": SentimentScore(0.75, 0.5, 0.7),
    "light": SentimentScore(0.6, 0.3, 0.5),
    "dream": SentimentScore(0.5, 0.2, 0.3),
    "hope": SentimentScore(0.7, 0.4, 0.5),
    "fire": SentimentScore(0.3, 0.9, 0.7),
    "power": SentimentScore(0.4, 0.8, 0.9),
    "dark": SentimentScore(-0.5, 0.5, 0.4),
    "pain": SentimentScore(-0.7, 0.6, 0.2),
    "cry": SentimentScore(-0.6, 0.5, 0.15),
    "alone": SentimentScore(-0.5, 0.2, 0.15),
    "lost": SentimentScore(-0.55, 0.3, 0.1),
    "broken": SentimentScore(-0.7, 0.4, 0.1),
    "hate": SentimentScore(-0.85, 0.8, 0.6),
    "death": SentimentScore(-0.8, 0.5, 0.3),
    "war": SentimentScore(-0.6, 0.9, 0.7),
    "rage": SentimentScore(-0.5, 0.95, 0.8),
    "peace": SentimentScore(0.7, 0.1, 0.5),
    "gentle": SentimentScore(0.5, 0.15, 0.3),
    "thunder": SentimentScore(-0.1, 0.9, 0.7),
    "rain": SentimentScore(-0.1, 0.3, 0.2),
    "sun": SentimentScore(0.6, 0.5, 0.5),
    "night": SentimentScore(-0.2, 0.3, 0.3),
    "heart": SentimentScore(0.4, 0.5, 0.4),
    "soul": SentimentScore(0.3, 0.4, 0.4),
    "rise": SentimentScore(0.6, 0.7, 0.7),
    "fall": SentimentScore(-0.4, 0.5, 0.2),
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere", "nor", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't"}
INTENSIFIERS = {"very": 1.3, "really": 1.2, "so": 1.2, "extremely": 1.5, "incredibly": 1.4, "absolutely": 1.4}
DIMINISHERS = {"slightly": 0.7, "somewhat": 0.75, "barely": 0.5, "hardly": 0.5, "a little": 0.6}


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def score_phrase(words: list[str]) -> SentimentScore:
    if not words:
        return SentimentScore(0.0, 0.0, 0.0)
    vals, aros, doms = [], [], []
    negated = False
    intensity = 1.0
    for w in words:
        if w in NEGATION_WORDS:
            negated = True
            continue
        if w in INTENSIFIERS:
            intensity = INTENSIFIERS[w]
            continue
        if w in DIMINISHERS:
            intensity = DIMINISHERS[w]
            continue
        entry = MUSIC_SENTIMENT_LEXICON.get(w)
        if entry:
            sign = -1.0 if negated else 1.0
            vals.append(entry.valence * sign * intensity)
            aros.append(entry.arousal * intensity)
            doms.append(entry.dominance * sign * intensity)
            negated = False
            intensity = 1.0
    if not vals:
        return SentimentScore(0.0, 0.0, 0.0)
    return SentimentScore(
        valence=float(np.clip(np.mean(vals), -1, 1)),
        arousal=float(np.clip(np.mean(aros), 0, 1)),
        dominance=float(np.clip(np.mean(doms), 0, 1)),
    )


def score_text(text: str) -> SentimentScore:
    return score_phrase(tokenize(text))


def sentiment_trajectory(
    lines: list[str],
) -> list[SentimentScore]:
    return [score_text(line) for line in lines]


def valence_arc(scores: list[SentimentScore]) -> np.ndarray:
    return np.array([s.valence for s in scores])


def arousal_arc(scores: list[SentimentScore]) -> np.ndarray:
    return np.array([s.arousal for s in scores])


def emotional_volatility(scores: list[SentimentScore]) -> float:
    if len(scores) < 2:
        return 0.0
    vals = [s.valence for s in scores]
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    return float(np.mean(diffs))


def dominant_emotion(scores: list[SentimentScore]) -> str:
    if not scores:
        return "neutral"
    avg_val = float(np.mean([s.valence for s in scores]))
    avg_aro = float(np.mean([s.arousal for s in scores]))
    if avg_val > 0.3 and avg_aro > 0.5:
        return "joyful"
    if avg_val > 0.3 and avg_aro <= 0.5:
        return "peaceful"
    if avg_val < -0.3 and avg_aro > 0.5:
        return "angry"
    if avg_val < -0.3 and avg_aro <= 0.5:
        return "melancholic"
    return "neutral"
