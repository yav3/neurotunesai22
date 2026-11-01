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
]
