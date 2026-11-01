"""Music analysis: key detection, beat tracking, structure, loudness, tonality."""
from models.music_analysis.key_detection import (
    KeyProfile, KeyDetector, detect_key,
    MAJOR_PROFILE, MINOR_PROFILE, PITCH_CLASSES,
)
from models.music_analysis.beat_tracking import (
    TempoEstimator, BeatTracker, estimate_tempo, track_beats,
)
from models.music_analysis.structure import (
    StructureAnalyzer, SectionLabeler, compute_ssm, find_boundaries,
)
from models.music_analysis.loudness import (
    LoudnessAnalyzer, DynamicRangeAnalyzer, rms_loudness, lufs_approximate,
)
from models.music_analysis.tonality import (
    TonalityAnalyzer, HarmonicRhythm, chromatic_ratio, harmonic_change_rate,
)

from models.music_analysis.onset_detection import (
    OnsetMethod, Onset, OnsetDetector,
    spectral_flux as onset_spectral_flux,
    high_frequency_content, complex_domain_odf,
    energy_odf, phase_deviation_odf,
    adaptive_threshold, pick_onsets, compute_odf,
)
from models.music_analysis.structure_segmentation import (
    SegmentType, Segment, self_similarity_matrix, novelty_curve,
    detect_boundaries, label_segments, repetition_structure,
    recurrence_plot, lag_matrix,
)

__all__ = [
    "KeyProfile", "KeyDetector", "detect_key",
    "MAJOR_PROFILE", "MINOR_PROFILE", "PITCH_CLASSES",
    "TempoEstimator", "BeatTracker", "estimate_tempo", "track_beats",
    "StructureAnalyzer", "SectionLabeler", "compute_ssm", "find_boundaries",
    "LoudnessAnalyzer", "DynamicRangeAnalyzer", "rms_loudness", "lufs_approximate",
    "TonalityAnalyzer", "HarmonicRhythm", "chromatic_ratio", "harmonic_change_rate",
    "SegmentType", "Segment", "self_similarity_matrix", "novelty_curve",
    "detect_boundaries", "label_segments", "repetition_structure",
    "recurrence_plot", "lag_matrix",
    "OnsetMethod", "Onset", "OnsetDetector",
    "onset_spectral_flux", "high_frequency_content", "complex_domain_odf",
    "energy_odf", "phase_deviation_odf",
    "adaptive_threshold", "pick_onsets", "compute_odf",
    "MetricLevel", "RhythmicEvent", "GrooveProfile",
    "inter_onset_intervals", "ioi_statistics", "swing_ratio",
    "microtiming_deviations", "syncopation_lhl", "metric_weight_pattern",
    "rhythmic_density", "onset_entropy", "metric_clarity", "compute_groove_profile",
    "SpectralDescriptors", "spectral_centroid", "spectral_spread",
    "spectral_skewness", "spectral_kurtosis", "spectral_flatness",
    "spectral_crest", "spectral_rolloff", "spectral_slope",
    "spectral_decrease", "spectral_flux",
    "compute_spectral_descriptors", "Formant", "estimate_formants",
    "harmonic_to_noise_ratio",
    "TempoEstimate", "onset_strength_envelope", "autocorrelation_tempo",
    "tempo_curve", "beat_strength", "tempo_stability", "pulse_clarity",
    "analysis_stft", "analysis_istft",
    "harmonic_percussive_separation", "nmf_separate", "NMFResult",
    "nmf_reconstruct", "spectral_centroid_separation", "wiener_filter",
    "INTERVAL_TENSION", "TONAL_HIERARCHY", "TensionPoint",
    "interval_dissonance", "chord_dissonance", "tonal_distance", "harmonic_tension",
    "tension_resolution_curve", "resolution_strength",
    "detect_resolutions", "tension_profile", "cadence_strength", "prolongational_reduction",
    "cosine_similarity_matrix", "euclidean_distance_matrix",
    "distance_to_similarity", "recurrence_matrix",
    "novelty_from_ssm", "pick_boundaries",
    "label_ssm_segments", "form_string",
    "repetition_ratio", "diagonal_strength", "find_repeated_sections",
    "StructureSegment",
]

from .tempo_analysis import (
    TempoEstimate, onset_strength_envelope, autocorrelation_tempo,
    tempo_curve, beat_strength, tempo_stability, pulse_clarity,
)
from .source_separation import (
    stft as analysis_stft, istft as analysis_istft,
    harmonic_percussive_separation, nmf_separate, NMFResult,
    nmf_reconstruct, spectral_centroid_separation, wiener_filter,
)
from .rhythm_analysis import (
    MetricLevel, RhythmicEvent, GrooveProfile,
    inter_onset_intervals, ioi_statistics, swing_ratio,
    microtiming_deviations, syncopation_lhl, metric_weight_pattern,
    rhythmic_density, onset_entropy, metric_clarity, compute_groove_profile,
)
from .spectral_analysis import (
    SpectralDescriptors, spectral_centroid, spectral_spread,
    spectral_skewness, spectral_kurtosis, spectral_flatness,
    spectral_crest, spectral_rolloff, spectral_slope,
    spectral_decrease, spectral_flux,
    compute_spectral_descriptors, Formant, estimate_formants,
    harmonic_to_noise_ratio,
)
from .tension_resolution import (
    INTERVAL_TENSION, TONAL_HIERARCHY, TensionPoint,
    interval_dissonance, chord_dissonance, tonal_distance, harmonic_tension,
    tension_curve as tension_resolution_curve, resolution_strength,
    detect_resolutions, tension_profile, cadence_strength, prolongational_reduction,
)
from .self_similarity import (
    cosine_similarity_matrix, euclidean_distance_matrix,
    distance_to_similarity, recurrence_matrix,
    novelty_from_ssm, pick_boundaries,
    label_segments as label_ssm_segments, form_string,
    repetition_ratio, diagonal_strength, find_repeated_sections,
    StructureSegment,
)
from .motivic_analysis import (
    TransformationType, Motive, MotiveOccurrence, MotivicDevelopment,
    interval_sequence, contour_sequence, invert_intervals, retrograde_intervals,
    augment_durations, diminish_durations,
    mongeau_sankoff_distance, contour_similarity,
    classify_transformation, extract_motives,
    track_motive_occurrences, motivic_density_curve,
    analyze_motivic_development, motivic_coherence,
)
from .texture_analysis import (
    TextureType, TextureSegment, TextureProfile,
    estimate_voice_count, voice_density_curve,
    onset_synchrony, registral_spread, registral_spread_curve,
    voice_spacing, spacing_regularity,
    classify_texture_frame, pitch_class_entropy,
    textural_complexity, analyze_texture,
    melodic_prominence, texture_transition_matrix,
)
