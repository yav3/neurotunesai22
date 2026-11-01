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
from models.music_emotion.dimensional_model import (
    VADPoint as DeepVADPoint, CircumplexPoint, ThayerPoint,
    EmotionLabel, TrajectoryFrame, RidgeModel,
    circumplex_angle, circumplex_angular_distance, circumplex_chord_distance,
    project_to_circumplex, circumplex_density, procrustes_align,
    va_to_thayer, thayer_to_va, thayer_quadrant,
    vad_distance, vad_cosine_similarity, vad_centroid, vad_dispersion,
    vad_covariance, vad_mahalanobis, nearest_emotion, vad_to_plutchik,
    plutchik_dyad, plutchik_intensity_label,
    gems_to_va, gems_to_vad, va_to_gems_approx,
    fit_ridge_vad, predict_vad_ridge, predict_vad_posterior,
    invert_vad_to_features,
    emotion_velocity, emotion_acceleration, trajectory_arc_length,
    trajectory_curvature, trajectory_torsion, frenet_serret_frame,
    emotional_surprise as deep_emotional_surprise,
    emotional_momentum as deep_emotional_momentum,
    tempo_valence_arousal, mode_valence, spectral_brightness_arousal,
    loudness_dominance, harmonic_complexity_valence, articulation_arousal,
    register_valence, feature_dict_to_vad,
    slerp_va, lerp_vad, blend_emotions,
    segment_by_emotion, emotional_contrast as deep_emotional_contrast,
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
    "PhysiologicalResponse", "MusicPhysioMapping",
    "predict_heart_rate_change", "predict_skin_conductance",
    "predict_respiration_sync", "chills_probability",
    "relaxation_index", "physio_arousal_trajectory",
    "valence_from_mode_and_tempo", "entrainment_strength",
    "autonomic_response_model", "stress_reduction_score",
    "BasicEmotion", "VADPoint", "EMOTION_VAD",
    "EmotionBlend", "normalize_weights", "blend_vad",
    "emotional_ambiguity", "emotional_tension", "create_blend",
    "NAMED_BLENDS", "named_blend",
    "MixedEmotionTrajectory", "interpolate_blends", "mixed_emotion_trajectory",
    "vad_from_features", "nearest_blend",
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
from models.music_emotion.physiological_response import (
    PhysiologicalResponse, MusicPhysioMapping,
    predict_heart_rate_change, predict_skin_conductance,
    predict_respiration_sync, chills_probability,
    relaxation_index, arousal_trajectory as physio_arousal_trajectory,
    valence_from_mode_and_tempo, entrainment_strength,
    autonomic_response_model, stress_reduction_score,
)
from models.music_emotion.mixed_emotions import (
    BasicEmotion, VADPoint, EMOTION_VAD,
    EmotionBlend, normalize_weights, blend_vad,
    emotional_ambiguity, emotional_tension, create_blend,
    NAMED_BLENDS, named_blend,
    EmotionTrajectory as MixedEmotionTrajectory,
    interpolate_blends, emotion_trajectory as mixed_emotion_trajectory,
    vad_from_features, nearest_blend,
)
