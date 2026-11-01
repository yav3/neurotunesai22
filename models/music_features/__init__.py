from __future__ import annotations

from models.music_features.spectral import SpectralFeatures, SpectralExtractor
from models.music_features.rhythm import RhythmFeatures, RhythmExtractor
from models.music_features.harmonic import HarmonicFeatures, HarmonicExtractor
from models.music_features.dynamics import DynamicsFeatures, DynamicsExtractor
from models.music_features.timbre import TimbreFeatures, TimbreExtractor
from models.music_features.unified_extractor import (
    MusicFeatureSet,
    UnifiedMusicExtractor,
)
from models.music_features.texture import TextureAnalyzer
from models.music_features.deep_extractor import (
    SpectralFeatureExtractor as DeepSpectralExtractor,
    RhythmFeatureExtractor as DeepRhythmExtractor,
    HarmonicFeatureExtractor as DeepHarmonicExtractor,
    TimbreFeatureExtractor as DeepTimbreExtractor,
    UnifiedDeepExtractor,
)

from models.music_features.feature_fusion import (
    zscore_normalize, minmax_normalize, early_fusion,
    late_fusion_scores, learn_fusion_weights,
    attention_weights, attention_aggregate, multi_head_attention_aggregate,
    mutual_information, mi_feature_ranking, select_top_k,
    PCAResult, pca_compress, pca_reconstruct, auto_k_pca,
    temporal_pool_stats, temporal_pool_mean, temporal_pool_max,
    temporal_pool_attention, multi_resolution_extract,
    permutation_importance, FusionPipeline,
)

__all__ = [
    "SpectralFeatures", "SpectralExtractor",
    "RhythmFeatures", "RhythmExtractor",
    "HarmonicFeatures", "HarmonicExtractor",
    "DynamicsFeatures", "DynamicsExtractor",
    "TimbreFeatures", "TimbreExtractor",
    "MusicFeatureSet", "UnifiedMusicExtractor",
    "DeepSpectralExtractor", "DeepRhythmExtractor",
    "DeepHarmonicExtractor", "DeepTimbreExtractor",
    "UnifiedDeepExtractor",
    "TextureAnalyzer",
    "zscore_normalize", "minmax_normalize", "early_fusion",
    "late_fusion_scores", "learn_fusion_weights",
    "attention_weights", "attention_aggregate", "multi_head_attention_aggregate",
    "mutual_information", "mi_feature_ranking", "select_top_k",
    "PCAResult", "pca_compress", "pca_reconstruct", "auto_k_pca",
    "temporal_pool_stats", "temporal_pool_mean", "temporal_pool_max",
    "temporal_pool_attention", "multi_resolution_extract",
    "permutation_importance", "FusionPipeline",
    "sd_spectral_centroid", "sd_spectral_bandwidth", "sd_spectral_rolloff",
    "sd_spectral_flatness", "sd_spectral_slope", "spectral_decrease",
    "spectral_crest_factor", "bark_band_energies",
    "sd_mfcc", "spectral_contrast", "spectral_feature_vector",
    "hz_to_mel", "mel_to_hz", "mf_mel_filterbank", "chroma_filterbank",
    "mel_spectrogram", "log_mel_spectrogram", "mf_mfcc",
    "delta_features", "delta_delta_features",
    "chroma_stft", "chroma_cqt", "chroma_cens", "tonnetz_features",
    "mf_spectral_contrast", "mf_spectral_bandwidth", "mf_spectral_flatness",
    "mf_zcr", "mf_rms", "MelFeatureSet", "extract_mel_features",
]

from models.music_features.rhythm_features import (
    onset_density, inter_onset_intervals, tempo_stability,
    beat_histogram, pulse_clarity, syncopation_index,
    swing_ratio as rf_swing_ratio, microtiming_deviation,
    tempo_drift, npvi, rhythmic_complexity,
)
from models.music_features.harmonic_features import (
    key_strength, mode_strength, key_clarity,
    chroma_to_tonnetz, tonal_centroid as hf_tonal_centroid, tonal_radius,
    hcdf, chord_change_rate, harmonic_complexity, harmonic_feature_vector,
    MAJOR_PROFILE, MINOR_PROFILE,
)
from models.music_features.temporal_features import (
    rms_energy as tf_rms_energy, zero_crossing_rate as tf_zcr,
    amplitude_envelope, attack_time, decay_time, sustain_level,
    temporal_centroid as tf_temporal_centroid, effective_duration,
    crest_factor as tf_crest_factor, dynamic_range, loudness_range,
    modulation_spectrum, temporal_feature_vector,
)
from models.music_features.spectral_descriptors import (
    spectral_centroid as sd_spectral_centroid,
    spectral_bandwidth as sd_spectral_bandwidth,
    spectral_rolloff as sd_spectral_rolloff,
    spectral_flatness as sd_spectral_flatness,
    spectral_slope as sd_spectral_slope,
    spectral_decrease, spectral_crest_factor,
    bark_band_energies, mel_filterbank as sd_mel_filterbank,
    mfcc as sd_mfcc, spectral_contrast, spectral_feature_vector,
)
from models.music_features.mel_features import (
    hz_to_mel, mel_to_hz,
    mel_filterbank as mf_mel_filterbank,
    chroma_filterbank,
    mel_spectrogram, log_mel_spectrogram,
    mfcc as mf_mfcc,
    delta_features, delta_delta_features,
    chroma_stft, chroma_cqt, chroma_cens,
    tonnetz_features,
    spectral_contrast as mf_spectral_contrast,
    spectral_bandwidth as mf_spectral_bandwidth,
    spectral_flatness as mf_spectral_flatness,
    zero_crossing_rate as mf_zcr,
    rms_energy as mf_rms,
    MelFeatureSet, extract_mel_features,
)
