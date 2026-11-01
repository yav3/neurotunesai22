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
    "EmotionCategory", "AudioFeatures", "EmotionPrediction", "FEATURE_WEIGHTS",
    "predict_vad", "vad_to_category", "predict_emotion",
    "emotion_recognition_trajectory", "emotion_transition_matrix",
    "AVPoint", "AVTrajectory", "ArousalValenceModel",
    "av_spectral_features", "av_rhythm_features", "av_extract_features",
    "av_trajectory", "emotion_density", "quadrant_distribution",
    "emotional_arc_similarity",
    "ArcShape", "EmotionPoint", "TrajectoryStats", "TrajectoryDistance",
    "analyze_trajectory", "dtw_trajectory_distance",
    "frechet_trajectory_distance", "trajectory_distance",
    "detect_arc_shape", "emotion_momentum", "emotion_inertia",
    "transition_matrix", "vad_to_quadrant",
    "TimbreQuality", "TimbreDescriptor", "TimbreEmotionMapping",
    "descriptor_to_vector", "normalise_descriptor",
    "timbre_to_valence", "timbre_to_arousal", "timbre_to_dominance",
    "classify_timbre_qualities", "timbre_emotion_map",
    "timbre_emotion_trajectory", "timbre_distance",
    "pairwise_timbre_distance", "emotion_centroid_from_timbres",
    "timbre_emotion_variance", "spectral_centroid_from_magnitude",
    "spectral_spread_from_magnitude", "spectral_flux_from_frames",
    "roughness_from_partials", "descriptor_from_spectrum",
]

from models.music_emotion.emotion_recognition import (
    EmotionCategory, AudioFeatures, EmotionPrediction, FEATURE_WEIGHTS,
    predict_vad, vad_to_category, predict_emotion,
    emotion_trajectory as emotion_recognition_trajectory,
    emotion_transition_matrix,
)
from models.music_emotion.arousal_model import (
    AVPoint, AVTrajectory, ArousalValenceModel,
    spectral_features as av_spectral_features,
    rhythm_features as av_rhythm_features,
    extract_features as av_extract_features,
    av_trajectory, emotion_density, quadrant_distribution,
    emotional_arc_similarity,
)
from models.music_emotion.emotion_trajectory import (
    ArcShape, EmotionPoint, TrajectoryStats, TrajectoryDistance,
    analyze_trajectory, dtw_trajectory_distance,
    frechet_trajectory_distance, trajectory_distance,
    detect_arc_shape, emotion_momentum, emotion_inertia,
    transition_matrix, vad_to_quadrant,
)
from models.music_emotion.harmony_emotion import (
    AffectPoint, CHORD_AFFECT, chord_affect, progression_valence,
    progression_arousal, harmonic_tension, mode_brightness,
    emotional_trajectory as harmonic_emotional_trajectory,
    emotional_range, emotional_centroid as harmonic_emotional_centroid,
    emotional_variance, valence_arousal_quadrant, major_minor_ratio,
)
from models.music_emotion.timbre_emotion import (
    TimbreQuality, TimbreDescriptor, TimbreEmotionMapping,
    descriptor_to_vector, normalise_descriptor,
    timbre_to_valence, timbre_to_arousal, timbre_to_dominance,
    classify_timbre_qualities, timbre_emotion_map,
    timbre_emotion_trajectory, timbre_distance,
    pairwise_timbre_distance, emotion_centroid_from_timbres,
    timbre_emotion_variance, spectral_centroid_from_magnitude,
    spectral_spread_from_magnitude, spectral_flux_from_frames,
    roughness_from_partials, descriptor_from_spectrum,
)
