"""Music-specific sentiment lexicon with VAD (valence, arousal, dominance) mappings."""
import numpy as np

MUSIC_SENTIMENT_LEXICON: dict[str, tuple[float, float, float]] = {
    "crescendo": (0.6, 0.8, 0.7),
    "diminuendo": (0.3, 0.2, 0.3),
    "forte": (0.5, 0.9, 0.8),
    "piano": (0.5, 0.2, 0.3),
    "allegro": (0.7, 0.8, 0.6),
    "adagio": (0.4, 0.2, 0.3),
    "lament": (-0.7, 0.3, 0.2),
    "jubilant": (0.9, 0.9, 0.8),
    "melancholy": (-0.6, 0.2, 0.2),
    "triumphant": (0.9, 0.8, 0.9),
    "ethereal": (0.5, 0.2, 0.3),
    "aggressive": (-0.3, 0.9, 0.8),
    "serene": (0.6, 0.1, 0.4),
    "chaotic": (-0.2, 0.9, 0.3),
    "majestic": (0.7, 0.6, 0.9),
    "haunting": (-0.3, 0.4, 0.3),
    "playful": (0.8, 0.7, 0.5),
    "solemn": (-0.1, 0.2, 0.5),
    "passionate": (0.6, 0.8, 0.7),
    "tranquil": (0.6, 0.1, 0.4),
    "dark": (-0.5, 0.4, 0.5),
    "bright": (0.7, 0.6, 0.5),
    "warm": (0.6, 0.4, 0.5),
    "cold": (-0.3, 0.2, 0.4),
    "epic": (0.7, 0.9, 0.9),
    "intimate": (0.5, 0.2, 0.4),
    "dramatic": (0.2, 0.8, 0.7),
    "dreamy": (0.5, 0.2, 0.2),
    "energetic": (0.7, 0.9, 0.7),
    "mellow": (0.5, 0.2, 0.3),
    "intense": (0.1, 0.9, 0.7),
    "gentle": (0.6, 0.2, 0.3),
    "powerful": (0.4, 0.8, 0.9),
    "delicate": (0.5, 0.1, 0.2),
    "brooding": (-0.4, 0.5, 0.6),
    "uplifting": (0.8, 0.7, 0.6),
    "somber": (-0.5, 0.2, 0.3),
    "exuberant": (0.9, 0.9, 0.7),
    "contemplative": (0.2, 0.2, 0.5),
    "ominous": (-0.5, 0.6, 0.4),
    "joyful": (0.9, 0.8, 0.7),
    "sorrowful": (-0.8, 0.3, 0.2),
    "nostalgic": (0.1, 0.3, 0.3),
    "rebellious": (-0.1, 0.8, 0.7),
    "soothing": (0.6, 0.1, 0.3),
    "turbulent": (-0.3, 0.8, 0.4),
    "graceful": (0.6, 0.3, 0.5),
    "harsh": (-0.4, 0.8, 0.6),
    "tender": (0.7, 0.2, 0.3),
    "fierce": (-0.2, 0.9, 0.8),
    "peaceful": (0.7, 0.1, 0.4),
    "restless": (-0.2, 0.7, 0.4),
    "cheerful": (0.8, 0.7, 0.6),
    "gloomy": (-0.6, 0.3, 0.2),
    "radiant": (0.8, 0.6, 0.6),
    "weary": (-0.4, 0.1, 0.2),
    "vivacious": (0.8, 0.8, 0.7),
    "lonely": (-0.6, 0.2, 0.1),
    "festive": (0.8, 0.8, 0.7),
    "funereal": (-0.7, 0.2, 0.3),
    "sprightly": (0.7, 0.7, 0.5),
    "menacing": (-0.5, 0.7, 0.7),
    "sublime": (0.7, 0.4, 0.6),
    "tormented": (-0.7, 0.7, 0.3),
    "buoyant": (0.7, 0.6, 0.5),
    "desolate": (-0.8, 0.2, 0.1),
    "heroic": (0.7, 0.8, 0.9),
    "wistful": (0.0, 0.2, 0.2),
    "thunderous": (-0.1, 0.9, 0.8),
    "lyrical": (0.5, 0.3, 0.4),
    "frantic": (-0.3, 0.9, 0.4),
    "whimsical": (0.6, 0.5, 0.3),
    "mournful": (-0.7, 0.3, 0.2),
    "ecstatic": (0.9, 0.9, 0.8),
    "bleak": (-0.6, 0.2, 0.2),
    "vibrant": (0.7, 0.7, 0.6),
    "pensive": (0.1, 0.2, 0.4),
    "furious": (-0.5, 0.9, 0.8),
    "angelic": (0.7, 0.2, 0.4),
    "sinister": (-0.6, 0.6, 0.6),
}

_WORDS_LOWER = {k.lower(): v for k, v in MUSIC_SENTIMENT_LEXICON.items()}


def score_text_sentiment(text: str) -> tuple[float, float, float]:
    words = text.lower().split()
    vals, aros, doms = [], [], []
    for w in words:
        w_clean = w.strip(".,!?;:'\"()-")
        if w_clean in _WORDS_LOWER:
            v, a, d = _WORDS_LOWER[w_clean]
            vals.append(v)
            aros.append(a)
            doms.append(d)
    if not vals:
        return (0.0, 0.0, 0.0)
    return (float(np.mean(vals)), float(np.mean(aros)), float(np.mean(doms)))


def emotional_distance(vad_a: tuple, vad_b: tuple) -> float:
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(vad_a, vad_b))))


def closest_emotion_word(vad: tuple[float, float, float]) -> str:
    best_word = ""
    best_dist = float("inf")
    for word, wvad in MUSIC_SENTIMENT_LEXICON.items():
        d = emotional_distance(vad, wvad)
        if d < best_dist:
            best_dist = d
            best_word = word
    return best_word


def sentiment_polarity(text: str) -> float:
    v, _, _ = score_text_sentiment(text)
    return float(np.clip(v, -1.0, 1.0))


def arousal_level(text: str) -> float:
    _, a, _ = score_text_sentiment(text)
    return float(np.clip(a, 0.0, 1.0))
