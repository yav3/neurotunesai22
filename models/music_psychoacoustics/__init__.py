"""Music psychoacoustics: critical bands, masking, loudness, pitch, roughness, spatial hearing, timbre."""
from models.music_psychoacoustics.critical_bands import (
    hz_to_bark, bark_to_hz, hz_to_erb, erb_to_hz, erb_bandwidth,
    CriticalBandAnalyzer, GammatoneFilterbank,
)
from models.music_psychoacoustics.masking import (
    SimultaneousMasking, TemporalMasking, MaskingModel,
)
from models.music_psychoacoustics.loudness import (
    EqualLoudness, SpecificLoudness,
    ISO_226_FREQS, THRESHOLD_SPL,
)
from models.music_psychoacoustics.pitch_perception import (
    VirtualPitch, PitchSalience, estimate_pitch,
)
from models.music_psychoacoustics.roughness import (
    plomp_levelt_curve, critical_bandwidth,
    sethares_dissonance, hutchinson_knopoff,
    DissonanceCurve, interval_dissonance_12tet, consonance_ranking_12tet,
    RoughnessAnalyzer,
)
from models.music_psychoacoustics.timbre import (
    TimbreDescriptors as TimbreDescriptorsBasic, TimbreSpace,
    spectral_centroid, spectral_spread, spectral_flux,
    spectral_rolloff, spectral_flatness, brightness,
    attack_time, log_attack_time, temporal_centroid,
    zero_crossing_rate, compute_timbre_descriptors,
)
from models.music_psychoacoustics.spatial_hearing import (
    SpatialSource, BinauralCues, HRTFPair,
    SPEED_OF_SOUND, HEAD_RADIUS,
    interaural_time_difference, interaural_level_difference,
    head_shadow_filter, distance_attenuation, air_absorption,
    generate_synthetic_hrtf, apply_hrtf,
    interaural_coherence, estimate_azimuth, binaural_cue_analysis,
)
from models.music_psychoacoustics.timbre_perception import (
    TimbreDescriptors,
    odd_even_ratio, inharmonicity, warmth,
    roughness_dissonance, total_roughness,
    analyze_timbre, timbre_distance,
)

from models.music_psychoacoustics.temporal_perception import (
    IntegrationWindow, INTEGRATION_WINDOWS,
    weber_tempo_jnd, tempo_discrimination,
    forward_masking_threshold, backward_masking_threshold,
    gap_detection_threshold, temporal_order_judgment,
    temporal_modulation_transfer, rhythm_perception_window,
    preferred_tempo_resonance, ioi_variability, perceptual_attack_time,
)
from models.music_psychoacoustics.auditory_scene import (
    StreamSegregation, AuditoryGrouping,
)

__all__ = [
    "hz_to_bark", "bark_to_hz", "hz_to_erb", "erb_to_hz", "erb_bandwidth",
    "CriticalBandAnalyzer", "GammatoneFilterbank",
    "SimultaneousMasking", "TemporalMasking", "MaskingModel",
    "EqualLoudness", "SpecificLoudness", "ISO_226_FREQS", "THRESHOLD_SPL",
    "VirtualPitch", "PitchSalience", "estimate_pitch",
    "plomp_levelt_curve", "critical_bandwidth",
    "sethares_dissonance", "hutchinson_knopoff",
    "DissonanceCurve", "interval_dissonance_12tet", "consonance_ranking_12tet",
    "RoughnessAnalyzer",
    "TimbreDescriptorsBasic", "TimbreSpace",
    "spectral_centroid", "spectral_spread", "spectral_flux",
    "spectral_rolloff", "spectral_flatness", "brightness",
    "attack_time", "log_attack_time", "temporal_centroid",
    "zero_crossing_rate", "compute_timbre_descriptors",
    "SpatialSource", "BinauralCues", "HRTFPair",
    "SPEED_OF_SOUND", "HEAD_RADIUS",
    "interaural_time_difference", "interaural_level_difference",
    "head_shadow_filter", "distance_attenuation", "air_absorption",
    "generate_synthetic_hrtf", "apply_hrtf",
    "interaural_coherence", "estimate_azimuth", "binaural_cue_analysis",
    "TimbreDescriptors",
    "odd_even_ratio", "inharmonicity", "warmth",
    "roughness_dissonance", "total_roughness",
    "analyze_timbre", "timbre_distance",
    "IntegrationWindow", "INTEGRATION_WINDOWS",
    "weber_tempo_jnd", "tempo_discrimination",
    "forward_masking_threshold", "backward_masking_threshold",
    "gap_detection_threshold", "temporal_order_judgment",
    "temporal_modulation_transfer", "rhythm_perception_window",
    "preferred_tempo_resonance", "ioi_variability", "perceptual_attack_time",
    "StreamSegregation", "AuditoryGrouping",
    "equal_loudness_contour", "phon_to_sone", "sone_to_phon",
    "spl_to_phon", "LoudnessSpectrum",
    "a_weighting", "c_weighting", "apply_weighting",
    "loudness_compensation", "frequency_specific_loudness",
    "ConsonanceModel", "IntervalQuality", "ConsonanceRating",
    "frequency_ratio", "harmonic_series_fit", "periodicity_pitch",
    "interference_roughness", "euler_gradus_suavitatis",
    "tenney_height", "harmonic_entropy", "consonance_rating",
    "dyad_consonance", "triad_consonance", "consonance_curve",
    "interval_ranking",
    "IntervalRating", "ChordConsonance", "TensionProfile",
    "frequency_ratio_simplicity", "harmonicity_score", "periodicity_pitch_strength",
    "plomp_levelt_consonance", "composite_consonance", "JUST_INTERVALS",
    "just_interval_consonance_ranking", "chord_consonance", "cultural_weighting",
    "tension_from_dissonance_curve", "set_dissonance",
]

from models.music_psychoacoustics.consonance_model import (
    ConsonanceModel as CompositeConsonanceModel,
    IntervalRating, ChordConsonance, TensionProfile,
    frequency_ratio_simplicity, harmonicity_score, periodicity_pitch_strength,
    plomp_levelt_consonance, composite_consonance, JUST_INTERVALS,
    just_interval_consonance_ranking, chord_consonance, cultural_weighting,
    tension_from_dissonance_curve, interval_class_vector as icv_from_pcs,
    set_dissonance,
)

from models.music_psychoacoustics.equal_loudness import (
    equal_loudness_contour, phon_to_sone, sone_to_phon,
    spl_to_phon, LoudnessSpectrum,
    a_weighting, c_weighting, apply_weighting,
    loudness_compensation, frequency_specific_loudness,
)
from models.music_psychoacoustics.auditory_streaming import (
    AuditoryEvent, Stream, frequency_proximity, temporal_proximity,
    timbre_proximity, coherence_matrix, fission_probability,
    segregate_streams, onset_synchrony_groups, harmonic_fusion_score,
    stream_salience, stream_continuity,
)
from models.music_psychoacoustics.consonance_models import (
    ConsonanceModel, IntervalQuality, ConsonanceRating,
    frequency_ratio, harmonic_series_fit, periodicity_pitch,
    interference_roughness, euler_gradus_suavitatis,
    tenney_height, harmonic_entropy, consonance_rating,
    dyad_consonance, triad_consonance, consonance_curve,
    interval_ranking,
)
