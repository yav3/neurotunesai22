from __future__ import annotations

from models.music_data.schema import (
    TrackRecord,
    SpectralAnalysis,
    VADPoetryRecord,
    PlaylistRecord,
    TrackSchema,
)
from models.music_data.catalog import (
    MusicCatalog,
    CatalogStats,
)
from models.music_data.loaders import (
    CSVTrackLoader,
    JSONTrackLoader,
)
from models.music_data.validation import (
    ValidationResult,
    validate_audio_features,
    validate_pitch_sequence,
    validate_onset_times,
    completeness_score,
    feature_matrix_quality,
    detect_duplicates,
    detect_outliers_iqr,
)
from models.music_data.open_sources import (
    KNOWN_DATASETS,
    DatasetMeta,
    MultiSourceIngester,
    parse_msd_record,
    parse_msd_csv,
    parse_musicbrainz_recording,
    parse_musicbrainz_json,
    parse_fma_record,
    parse_fma_csv,
    parse_acousticbrainz_features,
    parse_acousticbrainz_json,
    parse_jamendo_record,
    parse_jamendo_json,
    parse_midi_metadata,
    MIDIMeta,
    detect_source_format,
    parse_record,
)

__all__ = [
    "TrackRecord",
    "SpectralAnalysis",
    "VADPoetryRecord",
    "PlaylistRecord",
    "TrackSchema",
    "MusicCatalog",
    "CatalogStats",
    "CSVTrackLoader",
    "JSONTrackLoader",
    "KNOWN_DATASETS",
    "DatasetMeta",
    "MultiSourceIngester",
    "parse_msd_record",
    "parse_msd_csv",
    "parse_musicbrainz_recording",
    "parse_musicbrainz_json",
    "parse_fma_record",
    "parse_fma_csv",
    "parse_acousticbrainz_features",
    "parse_acousticbrainz_json",
    "parse_jamendo_record",
    "parse_jamendo_json",
    "parse_midi_metadata",
    "MIDIMeta",
    "detect_source_format",
    "parse_record",
    "ValidationResult",
    "validate_audio_features",
    "validate_pitch_sequence",
    "validate_onset_times",
    "completeness_score",
    "feature_matrix_quality",
    "detect_duplicates",
    "detect_outliers_iqr",
    "AugmentationType", "AugmentationConfig", "AugmentationResult",
    "pitch_shift_stft", "time_stretch_ola", "add_noise",
    "gain_change", "polarity_invert",
    "frequency_mask", "time_mask",
    "apply_augmentation_pipeline",
    "DistributionType", "FeatureStats", "ClassBalance", "CorrelationReport",
    "DatasetSummary", "compute_feature_stats", "class_balance",
    "feature_correlation", "effective_dimensionality", "dataset_summary",
    "bootstrap_ci", "mutual_information_matrix",
    "TransformKind", "TransformRecord", "DatasetSnapshot", "LineageGraph",
    "VersionedDataset", "content_hash", "metadata_hash",
    "create_snapshot", "record_transform", "dataset_fingerprint",
    "NormMethod", "ImputeStrategy", "NormParams", "FeatureImportance",
    "fit_normalize", "apply_normalize", "impute",
    "variance_threshold", "mutual_info_classif",
    "polynomial_features", "temporal_features",
    "SplitResult", "FoldResult",
    "stratified_split", "artist_aware_split", "temporal_split",
    "stratified_kfold", "group_kfold",
    "class_weight_balanced", "split_quality_check",
]

from models.music_data.augmentation import (
    AugmentationType, AugmentationConfig, AugmentationResult,
    pitch_shift_stft, time_stretch_ola, add_noise,
    gain_change, polarity_invert,
    frequency_mask, time_mask,
    apply_augmentation_pipeline,
)
from models.music_data.dataset_statistics import (
    DistributionType, FeatureStats, ClassBalance, CorrelationReport,
    DatasetSummary, compute_feature_stats, class_balance,
    feature_correlation, effective_dimensionality, dataset_summary,
    bootstrap_ci, mutual_information_matrix,
)
from models.music_data.data_versioning import (
    TransformKind, TransformRecord, DatasetSnapshot, LineageGraph,
    VersionedDataset, content_hash, metadata_hash,
    create_snapshot, record_transform, dataset_fingerprint,
)
from models.music_data.feature_engineering import (
    NormMethod, ImputeStrategy, NormParams, FeatureImportance,
    fit_normalize, apply_normalize, impute,
    variance_threshold, mutual_info_classif,
    polynomial_features, temporal_features,
)
from models.music_data.stratified_split import (
    SplitResult, FoldResult,
    stratified_split, artist_aware_split, temporal_split,
    stratified_kfold, group_kfold,
    class_weight_balanced, split_quality_check,
)
