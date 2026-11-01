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

from models.music_pipeline.data_augmentation import (
    AugmentationType, AugmentationConfig, AugmentationPipeline,
    pitch_shift_semitones, time_stretch, add_noise,
    gain_change_db, polarity_invert, low_pass_simple, high_pass_simple,
    random_eq, mixup, spec_augment_mask, apply_pipeline,
)
from models.music_pipeline.inference import (
    ModelType, PredictionStatus, ModelVersion, PredictionResult,
    InferenceMetrics, ModelRegistry, PredictFn,
    predict_with_timeout, batch_predict, ensemble_predict,
    StreamingPredictor,
)

__all__ = [
    "ModelType", "PredictionStatus", "ModelVersion", "PredictionResult",
    "InferenceMetrics", "ModelRegistry", "PredictFn",
    "predict_with_timeout", "batch_predict", "ensemble_predict",
    "StreamingPredictor",
    "AugmentationType", "AugmentationConfig", "AugmentationPipeline",
    "pitch_shift_semitones", "time_stretch", "add_noise",
    "gain_change_db", "polarity_invert", "low_pass_simple", "high_pass_simple",
    "random_eq", "mixup", "spec_augment_mask", "apply_pipeline",
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
    "ProcessingStage", "BatchConfig", "TrackResult", "BatchResult",
    "validate_batch_config", "process_single", "process_batch",
    "estimate_time", "retry_failed",
]

from models.music_pipeline.batch_processor import (
    ProcessingStage, BatchConfig, TrackResult, BatchResult,
    validate_batch_config, process_single, process_batch,
    estimate_time, retry_failed,
)
