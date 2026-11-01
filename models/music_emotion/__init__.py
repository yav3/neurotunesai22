from __future__ import annotations

from models.music_emotion.plutchik import (
    EmotionIntensity,
    PlutchikEmotion,
    PlutchikWheel,
)
from models.music_emotion.music_emotion_profile import (
    EmotionProfile,
    MusicEmotionMapper,
    MusicFeatureVector,
)
from models.music_emotion.russell_circumplex import (
    RussellCircumplex,
    RussellEmotion,
)
from models.music_emotion.gems import (
    GEMSFactor,
    GEMSProfile,
    MusicToGEMS,
)
from models.music_emotion.emotion_arc import (
    EmotionArc,
    EmotionArcAnalyzer,
)
from models.music_emotion.arousal_dynamics import (
    ArousalPoint,
    ArousalCurve,
    ArousalSegment,
    loudness_to_arousal,
    tempo_to_arousal,
    spectral_centroid_to_arousal,
    onset_density_to_arousal,
    harmonic_tension_to_arousal,
    compute_arousal_curve,
    segment_arousal,
    arousal_contrast,
    smooth_arousal,
)

from models.music_emotion.emotion_dynamics import (
    EmotionTrajectory, track_emotion_over_time, emotion_volatility,
    emotion_arc_type, emotional_surprise, valence_arousal_path_length,
    emotional_centroid,
)
from models.music_emotion.sentiment_lexicon import (
    MUSIC_SENTIMENT_LEXICON, score_text_sentiment, emotional_distance,
    closest_emotion_word, sentiment_polarity, arousal_level,
)
from models.music_emotion.dimensional_models import (
    DimensionalEmotionModel, MusicMoodClassifier,
    PLUTCHIK_EMOTIONS, MOOD_LABELS,
)

__all__ = [
    "EmotionIntensity",
    "PlutchikEmotion",
    "PlutchikWheel",
    "EmotionProfile",
    "MusicEmotionMapper",
    "MusicFeatureVector",
    "RussellCircumplex",
    "RussellEmotion",
    "GEMSFactor",
    "GEMSProfile",
    "MusicToGEMS",
    "EmotionArc",
    "EmotionArcAnalyzer",
    "ArousalPoint",
    "ArousalCurve",
    "ArousalSegment",
    "loudness_to_arousal",
    "tempo_to_arousal",
    "spectral_centroid_to_arousal",
    "onset_density_to_arousal",
    "harmonic_tension_to_arousal",
    "compute_arousal_curve",
    "segment_arousal",
    "arousal_contrast",
    "smooth_arousal",
    "EmotionTrajectory", "track_emotion_over_time", "emotion_volatility",
    "emotion_arc_type", "emotional_surprise", "valence_arousal_path_length",
    "emotional_centroid",
    "MUSIC_SENTIMENT_LEXICON", "score_text_sentiment", "emotional_distance",
    "closest_emotion_word", "sentiment_polarity", "arousal_level",
    "DimensionalEmotionModel", "MusicMoodClassifier",
    "PLUTCHIK_EMOTIONS", "MOOD_LABELS",
]
