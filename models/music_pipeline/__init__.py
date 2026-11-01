"""Music processing pipeline: end-to-end feature extraction, analysis, and annotation."""
from models.music_pipeline.feature_pipeline import (
    PipelineConfig, FeatureResult, FeaturePipeline,
    extract_spectral_features, extract_rhythm_features,
    extract_harmonic_features, extract_timbral_features,
    extract_emotion_features,
)
from models.music_pipeline.annotation import (
    TrackAnnotation, AnnotationPipeline,
    classify_mood_quadrant, derive_tags,
    NOTE_NAMES, MOOD_QUADRANTS,
)
from models.music_pipeline.quality import (
    QualityReport, assess_quality,
    peak_amplitude, rms_level, rms_to_db,
    clipping_ratio, silence_ratio, estimate_snr,
    dc_offset, crest_factor, dynamic_range, spectral_quality,
)
from models.music_pipeline.normalization import (
    FeatureStatistics, ZScoreNormalizer, MinMaxNormalizer,
    RobustNormalizer, FeatureSelector,
)

__all__ = [
    "PipelineConfig", "FeatureResult", "FeaturePipeline",
    "extract_spectral_features", "extract_rhythm_features",
    "extract_harmonic_features", "extract_timbral_features",
    "extract_emotion_features",
    "TrackAnnotation", "AnnotationPipeline",
    "classify_mood_quadrant", "derive_tags",
    "NOTE_NAMES", "MOOD_QUADRANTS",
    "QualityReport", "assess_quality",
    "peak_amplitude", "rms_level", "rms_to_db",
    "clipping_ratio", "silence_ratio", "estimate_snr",
    "dc_offset", "crest_factor", "dynamic_range", "spectral_quality",
    "FeatureStatistics", "ZScoreNormalizer", "MinMaxNormalizer",
    "RobustNormalizer", "FeatureSelector",
]
