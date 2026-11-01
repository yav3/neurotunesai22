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
]

from models.music_timbre.timbre_morphing import (
    MorphMethod, SpectralFrame, TimbreMorphResult,
    spectral_lerp, spectral_envelope as morph_spectral_envelope,
    envelope_swap, harmonic_blend,
    morph_trajectory, spectral_distance, brightness_transfer,
)
