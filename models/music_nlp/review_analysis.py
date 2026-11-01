"""Music review and critique analysis.

Extracts structured assessments from music reviews: overall rating estimation,
aspect-based sentiment (production, performance, composition, lyrics, originality),
comparative language detection, and review summarization features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AspectSentiment:
    aspect: str
    score: float           # [-1, 1]
    mentions: int = 0
    keywords: List[str] = field(default_factory=list)


@dataclass
class ReviewAnalysis:
    overall_sentiment: float
    aspects: Dict[str, AspectSentiment]
    comparative_mentions: List[str]
    superlatives: List[str]
    recommendation_signal: float   # [-1, 1] negative to enthusiastic


ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "production": [
        "mix", "mixing", "mastering", "production", "sound", "sonic",
        "bass", "treble", "volume", "loud", "quiet", "crisp", "muddy",
        "polished", "raw", "lo-fi", "hi-fi", "compressed", "dynamic",
        "reverb", "delay", "distortion", "clean", "warm", "bright",
    ],
    "performance": [
        "vocal", "vocals", "singing", "voice", "guitar", "drums",
        "bass", "piano", "solo", "performance", "technique", "skill",
        "virtuosity", "tight", "sloppy", "energetic", "passionate",
        "bland", "flat", "pitch", "tone", "execution", "delivery",
    ],
    "composition": [
        "melody", "melodies", "chord", "chords", "harmony", "harmonies",
        "structure", "arrangement", "progression", "hook", "riff",
        "chorus", "verse", "bridge", "outro", "intro", "orchestration",
        "counterpoint", "texture", "layering", "dynamics",
    ],
    "lyrics": [
        "lyrics", "lyrical", "words", "writing", "poetry", "poetic",
        "storytelling", "narrative", "message", "theme", "metaphor",
        "imagery", "rhyme", "wordplay", "clever", "cliche", "profound",
        "shallow", "meaningful", "vapid", "honest", "authentic",
    ],
    "originality": [
        "original", "innovative", "unique", "fresh", "derivative",
        "generic", "formulaic", "predictable", "surprising", "novel",
        "experimental", "boundary", "pushing", "conventional",
        "groundbreaking", "pioneering", "copycat", "retro", "nostalgic",
    ],
}

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "brilliant", "outstanding", "superb",
    "masterpiece", "stunning", "beautiful", "captivating", "compelling",
    "engaging", "impressive", "inspired", "memorable", "perfect", "sublime",
    "wonderful", "fantastic", "remarkable", "exceptional", "breathtaking",
    "gorgeous", "magnificent", "powerful", "moving", "touching", "haunting",
    "exquisite", "flawless", "mesmerizing", "enchanting", "riveting",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "poor", "weak", "disappointing",
    "mediocre", "boring", "dull", "uninspired", "forgettable",
    "generic", "derivative", "tedious", "annoying", "pretentious",
    "overrated", "repetitive", "sloppy", "amateurish", "cringeworthy",
    "lackluster", "lifeless", "stale", "grating", "unbearable",
}

COMPARATIVE_PATTERN = re.compile(
    r'(?:sounds?\s+like|reminiscent\s+of|similar\s+to|echoes?\s+of|'
    r'in\s+the\s+vein\s+of|influenced\s+by|channels?\s|evokes?)',
    re.IGNORECASE,
)

SUPERLATIVE_PATTERN = re.compile(
    r'\b(best|worst|greatest|finest|most\s+\w+|least\s+\w+|'
    r'masterpiece|classic|landmark|definitive|essential)\b',
    re.IGNORECASE,
)

RECOMMENDATION_POSITIVE = re.compile(
    r'(must[\s-]listen|highly\s+recommend|essential\s+listening|'
    r'do\s+yourself\s+a\s+favor|don\'t\s+miss|album\s+of\s+the\s+year)',
    re.IGNORECASE,
)

RECOMMENDATION_NEGATIVE = re.compile(
    r'(skip\s+this|avoid|don\'t\s+bother|waste\s+of\s+time|'
    r'pass\s+on\s+this|disappointing\s+release)',
    re.IGNORECASE,
)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _aspect_sentiment(tokens: List[str], keywords: List[str]) -> Tuple[float, int, List[str]]:
    """Compute sentiment for an aspect based on keyword proximity."""
    matched_kw = []
    sentiment_sum = 0.0
    count = 0
    for i, token in enumerate(tokens):
        if token in keywords:
            matched_kw.append(token)
            window = tokens[max(0, i - 5):i + 6]
            pos = sum(1 for w in window if w in POSITIVE_WORDS)
            neg = sum(1 for w in window if w in NEGATIVE_WORDS)
            if pos + neg > 0:
                sentiment_sum += (pos - neg) / (pos + neg)
                count += 1
    if count == 0:
        return 0.0, len(matched_kw), matched_kw
    return sentiment_sum / count, len(matched_kw), matched_kw


def analyze_review(text: str) -> ReviewAnalysis:
    """Extract structured analysis from a music review."""
    tokens = _tokenize(text)

    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos_count + neg_count
    overall = (pos_count - neg_count) / total if total > 0 else 0.0

    aspects = {}
    for aspect_name, keywords in ASPECT_KEYWORDS.items():
        score, mentions, kw = _aspect_sentiment(tokens, keywords)
        aspects[aspect_name] = AspectSentiment(aspect_name, score, mentions, kw)

    comparatives = [m.group() for m in COMPARATIVE_PATTERN.finditer(text)]
    superlatives = [m.group() for m in SUPERLATIVE_PATTERN.finditer(text)]

    rec_pos = len(RECOMMENDATION_POSITIVE.findall(text))
    rec_neg = len(RECOMMENDATION_NEGATIVE.findall(text))
    rec_total = rec_pos + rec_neg
    rec_signal = (rec_pos - rec_neg) / rec_total if rec_total > 0 else overall * 0.5

    return ReviewAnalysis(
        overall_sentiment=overall,
        aspects=aspects,
        comparative_mentions=comparatives,
        superlatives=superlatives,
        recommendation_signal=rec_signal,
    )


def rating_estimate(text: str, scale: Tuple[float, float] = (0.0, 10.0)) -> float:
    """Estimate a numeric rating from review text.

    Maps overall sentiment [-1, 1] to the given scale (default 0-10).
    """
    analysis = analyze_review(text)
    lo, hi = scale
    mid = (lo + hi) / 2
    half_range = (hi - lo) / 2
    return float(np.clip(mid + analysis.overall_sentiment * half_range, lo, hi))


def aspect_radar(text: str) -> Dict[str, float]:
    """Return normalized [0, 1] scores per aspect for radar chart display."""
    analysis = analyze_review(text)
    return {
        name: (asp.score + 1.0) / 2.0
        for name, asp in analysis.aspects.items()
    }


def review_feature_vector(text: str) -> np.ndarray:
    """Numeric feature vector from review analysis (11 features)."""
    analysis = analyze_review(text)
    aspect_scores = [analysis.aspects[a].score for a in sorted(ASPECT_KEYWORDS)]
    return np.array([
        analysis.overall_sentiment,
        *aspect_scores,
        len(analysis.comparative_mentions),
        len(analysis.superlatives),
        analysis.recommendation_signal,
        lexicon_density(text),
    ])


def lexicon_density(text: str) -> float:
    """Fraction of tokens that are sentiment-bearing (positive or negative)."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    sentiment_tokens = sum(1 for t in tokens if t in POSITIVE_WORDS or t in NEGATIVE_WORDS)
    return sentiment_tokens / len(tokens)
