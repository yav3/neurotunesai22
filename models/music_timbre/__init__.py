"""Timbre analysis: spectral shape descriptors, perceptual timbre space, envelope, harmonics."""

from models.music_timbre.timbre_descriptors import (
    TimbreDescriptors, AttackDescriptor, SpectralEnvelope,
    brightness, warmth, spectral_irregularity, odd_even_ratio,
    inharmonicity, log_attack_time, temporal_centroid,
    spectral_tilt, roughness_sethares, sharpness_zwicker,
    estimate_spectral_envelope, compute_timbre_descriptors,
)
from models.music_timbre.timbre_space import (
    TimbrePoint, TimbreDistance, TimbreSpace,
    grey_timbre_axes, mcadams_timbre_axes,
    descriptors_to_coordinates, timbre_similarity,
    mds_reduce, stress,
)
from models.music_timbre.descriptors import (
    spectral_centroid_from_spectrum, spectral_spread, spectral_skewness,
    spectral_kurtosis, spectral_flatness, spectral_crest,
)
from models.music_timbre.envelope import (
    ADSRParams, EnvelopeResult, amplitude_envelope, rms_envelope,
    estimate_adsr, synthesise_adsr, extract_envelope,
)
from models.music_timbre.harmonic_analysis import (
    PartialSet, HarmonicAnalysisResult, extract_partials,
    harmonic_to_noise_ratio, spectral_envelope, estimate_formants,
    harmonic_deviation, analyse_harmonics,
)
from models.music_timbre.spectral_envelope import (
    autocorrelation, levinson_durbin, lpc_envelope,
    cepstral_smoothing, true_envelope, spectral_tilt as lpc_spectral_tilt,
)
from models.music_timbre.attack_transient import (
    AttackType, TransientDescriptor, compute_attack_time, rise_slope,
    classify_attack, analyze_transient,
)

__all__ = [
    "TimbreDescriptors", "AttackDescriptor", "SpectralEnvelope",
    "brightness", "warmth", "spectral_irregularity", "odd_even_ratio",
    "inharmonicity", "log_attack_time", "temporal_centroid",
    "spectral_tilt", "roughness_sethares", "sharpness_zwicker",
    "estimate_spectral_envelope", "compute_timbre_descriptors",
    "TimbrePoint", "TimbreDistance", "TimbreSpace",
    "grey_timbre_axes", "mcadams_timbre_axes",
    "descriptors_to_coordinates", "timbre_similarity",
    "mds_reduce", "stress",
    "spectral_centroid_from_spectrum", "spectral_spread", "spectral_skewness",
    "spectral_kurtosis", "spectral_flatness", "spectral_crest",
    "ADSRParams", "EnvelopeResult", "amplitude_envelope", "rms_envelope",
    "estimate_adsr", "synthesise_adsr", "extract_envelope",
    "PartialSet", "HarmonicAnalysisResult", "extract_partials",
    "harmonic_to_noise_ratio", "spectral_envelope", "estimate_formants",
    "harmonic_deviation", "analyse_harmonics",
    "autocorrelation", "levinson_durbin", "lpc_envelope",
    "cepstral_smoothing", "true_envelope", "lpc_spectral_tilt",
    "AttackType", "TransientDescriptor", "compute_attack_time", "rise_slope",
    "classify_attack", "analyze_transient",
    "MorphMethod", "SpectralFrame", "TimbreMorphResult",
    "spectral_lerp", "morph_spectral_envelope",
    "envelope_swap", "harmonic_blend",
    "morph_trajectory", "spectral_distance", "brightness_transfer",
    "lpc_spectral_envelope_morph", "morph_spectral_crosssynth",
    "formant_transplant", "brightness_match",
    "TimbreMorphSequence", "log_spectral_distance",
    "harmonic_brightness",
    "median_filter_horizontal", "median_filter_vertical",
    "HPSSResult", "hpss", "iterative_hpss",
    "VoiceDetectionFeatures", "detect_vibrato", "formant_prominence",
    "voice_detection_features", "spectral_template_basis",
    "DrumHit", "classify_drum_hit", "detect_drum_onsets", "drum_pattern_density",
    "InstrumentClass", "InstrumentFeatures",
    "ir_extract_features", "ir_feature_vector", "ir_cosine_similarity",
    "classify_nearest", "classify_mahalanobis",
    "temporal_class_profile", "ir_feature_importance",
]

from models.music_timbre.timbre_morphing import (
    MorphMethod, SpectralFrame, TimbreMorphResult,
    spectral_lerp, spectral_envelope as morph_spectral_envelope,
    envelope_swap, harmonic_blend,
    morph_trajectory, spectral_distance, brightness_transfer,
)
from models.music_timbre.spectral_morphing import (
    spectral_envelope as lpc_spectral_envelope_morph,
    morph_spectral as morph_spectral_crosssynth,
    formant_transplant, brightness_match,
    TimbreMorphSequence, timbre_distance as log_spectral_distance,
    harmonic_brightness,
)
from models.music_timbre.instrument_templates import (
    InstrumentFamily, SpectralTemplate, IdentificationResult,
    INSTRUMENT_TEMPLATES, harmonic_spectrum_from_template,
    template_match_score, identify_instrument,
    family_probabilities, attack_time_estimate, match_by_attack,
)
from models.music_timbre.perceptual_timbre_space import (
    MPEG7Descriptor, TimbreVector, harmonic_spectral_centroid,
    harmonic_spectral_deviation, harmonic_spectral_spread,
    log_attack_time as mpeg7_log_attack_time,
    temporal_centroid as mpeg7_temporal_centroid,
    spectral_flatness_measure, compute_mpeg7_descriptors,
    sethares_roughness, pairwise_timbre_distance, mds_2d,
    timbre_centroid, timbre_interpolation,
    cluster_timbres_kmeans, silhouette_score,
    grey_dimensions, mcadams_dimensions,
)
from models.music_timbre.source_separation_features import (
    median_filter_horizontal, median_filter_vertical,
    HPSSResult, hpss, iterative_hpss,
    VoiceDetectionFeatures, detect_vibrato, formant_prominence,
    voice_detection_features, spectral_template_basis,
    DrumHit, classify_drum_hit, detect_drum_onsets, drum_pattern_density,
)
from models.music_timbre.instrument_recognition import (
    InstrumentClass, InstrumentFeatures,
    extract_features as ir_extract_features, feature_vector as ir_feature_vector,
    cosine_similarity as ir_cosine_similarity,
    classify_nearest, classify_mahalanobis,
    temporal_class_profile, feature_importance as ir_feature_importance,
)
