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

from models.music_pipeline.model_evaluation import (
    ClassificationReport, RegressionReport, RankingReport,
    confusion_matrix as eval_confusion_matrix,
    classification_report, regression_report,
    dcg_at_k, ndcg_at_k, mean_reciprocal_rank,
    precision_at_k, average_precision, ranking_report,
    calibration_error, brier_score,
    onset_detection_f1, pitch_accuracy, beat_tracking_f1,
    chord_recognition_accuracy, segment_boundary_f1, source_separation_sdr,
)

__all__ = [
    "ClassificationReport", "RegressionReport", "RankingReport",
    "eval_confusion_matrix",
    "classification_report", "regression_report",
    "dcg_at_k", "ndcg_at_k", "mean_reciprocal_rank",
    "precision_at_k", "average_precision", "ranking_report",
    "calibration_error", "brier_score",
    "onset_detection_f1", "pitch_accuracy", "beat_tracking_f1",
    "chord_recognition_accuracy", "segment_boundary_f1", "source_separation_sdr",
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
    "FoldResult", "CVResult",
    "k_fold_split", "stratified_k_fold", "leave_one_out", "group_k_fold",
    "accuracy", "mean_absolute_error", "f1_score_binary", "cv_confusion_matrix",
    "learning_curve_points",
    "StageType", "StageResult", "ChainResult",
    "remove_dc", "trim_silence", "mix_to_mono", "mix_to_stereo",
    "loudness_normalize", "peak_normalize", "resample_linear",
    "pre_emphasis", "de_emphasis", "apply_fade", "add_dither", "clip_guard",
    "PreprocessingChain", "default_chain", "speech_chain", "mastering_chain",
    "RingBuffer", "StreamConfig", "StreamState", "FeatureFrame",
    "create_ring_buffer", "push_samples", "pull_frame", "advance_read",
    "overlap_add_process", "streaming_stft", "streaming_features",
    "detect_onset_streaming", "adaptive_gain", "latency_report",
    "AudioChunk", "OnsetDetector", "StreamProcessor", "LatencyTracker",
    "rms_energy", "zero_crossing_rate_stream",
    "spectral_centroid_stream", "spectral_rolloff_stream", "spectral_flux_stream",
    "chunk_audio", "process_full_signal", "overlap_add_reconstruct",
    "CacheEntry", "CacheStats", "ContentAddressedCache",
    "make_cache_key", "memoize_feature",
    "ErrorSeverity", "CircuitState", "PipelineError", "RetryConfig",
    "CircuitBreaker", "HealthMetrics",
    "classify_error", "compute_backoff_delay", "should_retry",
    "update_health_metrics", "fallback_chain",
    "FeatureSchema", "FeatureRecord", "content_hash",
    "InMemoryFeatureStore", "feature_matrix", "stale_report",
]

from models.music_pipeline.dataset_builder import (
    DatasetSplit, stratified_split, FeatureCache,
    compute_dataset_statistics, DatasetStats,
    balance_classes, normalize_features,
)
from models.music_pipeline.cross_validation import (
    FoldResult, CVResult,
    k_fold_split, stratified_k_fold, leave_one_out, group_k_fold,
    accuracy, mean_absolute_error, f1_score_binary, confusion_matrix as cv_confusion_matrix,
    learning_curve_points,
)
from models.music_pipeline.batch_processor import (
    ProcessingStage, BatchConfig, TrackResult, BatchResult,
    validate_batch_config, process_single, process_batch,
    estimate_time, retry_failed,
)
from models.music_pipeline.preprocessing_chain import (
    StageType, StageResult, ChainResult,
    remove_dc, trim_silence, mix_to_mono, mix_to_stereo,
    loudness_normalize, peak_normalize, resample_linear,
    pre_emphasis, de_emphasis, apply_fade, add_dither, clip_guard,
    PreprocessingChain, default_chain, speech_chain, mastering_chain,
)
from models.music_pipeline.preprocessing_chain import (
    StageType, StageResult, ChainResult, PreprocessingChain,
    remove_dc, trim_silence, mix_to_mono, mix_to_stereo,
    loudness_normalize, peak_normalize, resample_linear,
    pre_emphasis, de_emphasis, apply_fade, add_dither, clip_guard,
    default_chain, speech_chain, mastering_chain,
)
from models.music_pipeline.quality_gate import (
    Verdict, GateCategory, GateResult, GatingReport,
    QualityGateConfig, run_quality_gates, batch_quality_gate,
    rejection_summary,
    gate_sample_rate, gate_duration, gate_clipping, gate_dc_offset,
    gate_silence_fraction, gate_snr, gate_spectral_bandwidth,
    gate_spectral_flatness, gate_loudness_range, gate_true_peak,
)
from models.music_pipeline.streaming_processor import (
    RingBuffer, StreamConfig, StreamState, FeatureFrame,
    create_ring_buffer, push_samples, pull_frame, advance_read,
    overlap_add_process, streaming_stft, streaming_features,
    detect_onset_streaming, adaptive_gain, latency_report,
)
from models.music_pipeline.streaming_pipeline import (
    AudioChunk, OnsetDetector, StreamProcessor, LatencyTracker,
    rms_energy,
    zero_crossing_rate as zero_crossing_rate_stream,
    spectral_centroid as spectral_centroid_stream,
    spectral_rolloff as spectral_rolloff_stream,
    spectral_flux as spectral_flux_stream,
    chunk_audio, process_full_signal,
    overlap_add as overlap_add_reconstruct,
)
from models.music_pipeline.feature_cache import (
    CacheEntry, CacheStats, FeatureCache as ContentAddressedCache,
    make_cache_key, memoize_feature,
)
from models.music_pipeline.error_handling import (
    ErrorSeverity, CircuitState, PipelineError, RetryConfig,
    CircuitBreaker, HealthMetrics,
    classify_error, compute_backoff_delay, should_retry,
    update_health_metrics, fallback_chain,
)
from models.music_pipeline.feature_store import (
    FeatureSchema, FeatureRecord, content_hash,
    InMemoryFeatureStore, feature_matrix, stale_report,
)
