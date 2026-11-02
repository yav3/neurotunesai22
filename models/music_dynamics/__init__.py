"""Music dynamics: velocity curves, dynamic markings, articulation, expression, and analysis."""

from .velocity_curves import (
    VelocityCurve, velocity_to_amplitude, amplitude_to_velocity,
    apply_velocity_curve, humanize_velocities, accent_pattern,
    velocity_gradient,
)
from .dynamic_markings import (
    DynamicLevel, DynamicMarking, DYNAMIC_VELOCITY_RANGES, DYNAMIC_NAMES,
    dynamic_to_velocity_range, dynamic_to_velocity, velocity_to_dynamic,
    interpret_hairpin, sforzando, subito_change, parse_dynamic_sequence,
)
from .articulation import (
    ArticulationType, ArticulatedNote, ARTICULATION_DURATION_RATIO,
    ARTICULATION_VELOCITY_SCALE, apply_articulation, legato_overlap,
    detect_articulation_from_ratio, batch_articulate,
)
from .expression_contour import (
    ExpressionContour, phrase_arc, crescendo_contour, diminuendo_contour,
    swell, terraced_dynamics, smooth_transition, rubato_warp,
    agogic_accent, combine_contours, build_expression_contour,
)
from .dynamic_analysis import (
    DynamicProfile, rms_envelope, rms_to_db, velocity_histogram,
    dynamic_range_midi, loudness_variation, detect_dynamic_changes,
    classify_dynamic_level, analyse_dynamics, compute_compression_ratio,
    normalise_velocities,
)

from .loudness_models import (
    LoudnessProfile, EBULoudness,
    phon_to_sone, sone_to_phon, a_weighting,
    momentary_loudness, integrated_loudness, loudness_range,
    compute_ebu_loudness, compute_loudness_profile,
)
from .compression import (
    CompressorParams, GainReduction, MultibandConfig,
    static_characteristic, expander_characteristic, limiter_characteristic,
    envelope_follower, compute_gain_reduction, compress_signal,
    multiband_split, multiband_compress, auto_makeup_gain,
)
from .expression import (
    DynamicEventType, DynamicEvent, PhraseArc, ExpressionProfile,
    detect_crescendo_diminuendo, detect_accents,
    rubato_strength, timing_jitter, extract_phrase_arcs,
    velocity_humanization, compute_expression_profile,
)

__all__ = [
    "VelocityCurve", "velocity_to_amplitude", "amplitude_to_velocity",
    "apply_velocity_curve", "humanize_velocities", "accent_pattern",
    "velocity_gradient",
    "DynamicLevel", "DynamicMarking", "DYNAMIC_VELOCITY_RANGES", "DYNAMIC_NAMES",
    "dynamic_to_velocity_range", "dynamic_to_velocity", "velocity_to_dynamic",
    "interpret_hairpin", "sforzando", "subito_change", "parse_dynamic_sequence",
    "ArticulationType", "ArticulatedNote", "ARTICULATION_DURATION_RATIO",
    "ARTICULATION_VELOCITY_SCALE", "apply_articulation", "legato_overlap",
    "detect_articulation_from_ratio", "batch_articulate",
    "ExpressionContour", "phrase_arc", "crescendo_contour", "diminuendo_contour",
    "swell", "terraced_dynamics", "smooth_transition", "rubato_warp",
    "agogic_accent", "combine_contours", "build_expression_contour",
    "DynamicProfile", "rms_envelope", "rms_to_db", "velocity_histogram",
    "dynamic_range_midi", "loudness_variation", "detect_dynamic_changes",
    "classify_dynamic_level", "analyse_dynamics", "compute_compression_ratio",
    "normalise_velocities",
    "LoudnessProfile", "EBULoudness",
    "phon_to_sone", "sone_to_phon", "a_weighting",
    "momentary_loudness", "integrated_loudness", "loudness_range",
    "compute_ebu_loudness", "compute_loudness_profile",
    "CompressorParams", "GainReduction", "MultibandConfig",
    "static_characteristic", "expander_characteristic", "limiter_characteristic",
    "envelope_follower", "compute_gain_reduction", "compress_signal",
    "multiband_split", "multiband_compress", "auto_makeup_gain",
    "DynamicEventType", "DynamicEvent", "PhraseArc", "ExpressionProfile",
    "detect_crescendo_diminuendo", "detect_accents",
    "rubato_strength", "timing_jitter", "extract_phrase_arcs",
    "velocity_humanization", "compute_expression_profile",
    "PerformanceStyle", "STYLE_PARAMS", "MicroNoteEvent", "HumanizedNote", "MicroDynamicProfile",
    "humanize_velocity", "humanize_timing", "apply_swing",
    "humanize", "downbeat_accent", "analyze_micro_dynamics",
    "MeterType", "LoudnessResult",
    "k_weighting_coefficients", "apply_k_weighting",
    "ln_momentary_loudness", "short_term_loudness", "loudness_gate",
    "ln_integrated_loudness", "ln_loudness_range",
    "true_peak", "normalize_loudness",
    "ln_rms_db", "ln_peak_db", "ln_crest_factor",
    "measure_loudness", "dynamic_range_meter",
    "TransientMode", "TransientProfile", "TransientParams",
    "td_envelope_follower", "fast_envelope", "slow_envelope",
    "separate_transient_sustain", "transient_shape",
    "detect_transients", "transient_profile",
    "percussive_enhance", "sustain_enhance", "de_esser", "noise_gate",
    "attack_time_ms", "td_crest_factor", "dynamic_eq_gain",
    "ClippingStats", "InterSamplePeak", "LoudnessWarProfile",
    "lw_crest_factor", "lw_crest_factor_db",
    "windowed_crest_factor", "peak_to_loudness_ratio", "peak_signal_ratio",
    "detect_clipping", "detect_inter_sample_peaks", "dynamic_spread",
    "loudness_fatigue_index", "classify_era", "analyze_loudness_war",
    "batch_crest_factor_trend",
    "DynamicCurve", "ADSREnvelope", "ExpressiveTimingFeatures",
    "velocity_to_loudness", "velocity_to_db",
    "perf_dynamic_range_db", "perf_dynamic_contrast",
    "crescendo_detector", "sforzando_detector",
    "rubato_measure", "agogic_accent_detector", "tempo_from_onsets",
    "perf_dynamic_envelope", "phrasing_arc_score",
    "expressive_timing_features", "build_dynamic_curve",
]

from .micro_dynamics import (
    PerformanceStyle, STYLE_PARAMS,
    NoteEvent as MicroNoteEvent, HumanizedNote, MicroDynamicProfile,
    humanize_velocity, humanize_timing, apply_swing,
    humanize, downbeat_accent, analyze_micro_dynamics,
)
from .sidechain import (
    SidechainMode, SidechainParams,
    rms_envelope as sc_rms_envelope, peak_envelope,
    compute_gain_curve, apply_sidechain,
    rhythmic_ducking, PumpingProfile, multiband_sidechain,
)
from .expressive_dynamics import (
    DynamicMarking as ExpressiveDynamicMarking,
    DynamicContourType, DynamicEvent as ExpressiveDynamicEvent,
    DynamicProfile as ExpressiveDynamicProfile,
    rms_envelope as expressive_rms_envelope, db_envelope,
    detect_crescendo as expressive_detect_crescendo,
    detect_decrescendo, detect_sforzando,
    terraced_dynamics as expressive_terraced, dynamic_range as expressive_dynamic_range,
    dynamic_variance, accent_pattern as expressive_accent_pattern,
    loudness_contour_similarity, analyze_dynamics as expressive_analyze_dynamics,
)
from .dynamic_range import (
    DynamicRangeStats, rms_level, peak_level, crest_factor,
    lufs_approximate, dynamic_range_dr, loudness_normalise, analyze_dynamic_range,
)
from .loudness_war import (
    ClippingStats, InterSamplePeak, LoudnessWarProfile,
    crest_factor as lw_crest_factor, crest_factor_db as lw_crest_factor_db,
    windowed_crest_factor, peak_to_loudness_ratio, peak_signal_ratio,
    detect_clipping, detect_inter_sample_peaks, dynamic_spread,
    loudness_fatigue_index, classify_era, analyze_loudness_war,
    batch_crest_factor_trend,
)
from .dynamic_planning import (
    DynamicLevel as PlanDynamicLevel, ArcShape, DynamicArc,
    generate_arc, velocity_to_dynamic, dynamic_to_velocity,
    crescendo_curve, phrase_dynamic_plan, dynamic_contrast,
    dynamic_variance_by_phrase, expressive_deviation,
    humanize_velocity as plan_humanize_velocity,
    structural_dynamic_template,
)
from .loudness_normalization import (
    MeterType, LoudnessResult,
    k_weighting_coefficients, apply_k_weighting,
    momentary_loudness as ln_momentary_loudness,
    short_term_loudness, loudness_gate,
    integrated_loudness as ln_integrated_loudness,
    loudness_range as ln_loudness_range,
    true_peak, normalize_loudness,
    rms_db as ln_rms_db, peak_db as ln_peak_db,
    crest_factor as ln_crest_factor,
    measure_loudness, dynamic_range_meter,
)
from .transient_design import (
    TransientMode, TransientProfile, TransientParams,
    envelope_follower as td_envelope_follower,
    fast_envelope, slow_envelope,
    separate_transient_sustain, transient_shape,
    detect_transients, transient_profile,
    percussive_enhance, sustain_enhance, de_esser, noise_gate,
    attack_time_ms, crest_factor as td_crest_factor, dynamic_eq_gain,
)
from .performance_dynamics import (
    DynamicCurve, ADSREnvelope, ExpressiveTimingFeatures,
    velocity_to_loudness, velocity_to_db,
    dynamic_range_db as perf_dynamic_range_db,
    dynamic_contrast as perf_dynamic_contrast,
    crescendo_detector, sforzando_detector,
    rubato_measure, agogic_accent_detector, tempo_from_onsets,
    dynamic_envelope as perf_dynamic_envelope,
    phrasing_arc_score, expressive_timing_features,
    build_dynamic_curve,
)
