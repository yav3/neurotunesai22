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
    DynamicProfile, dynamic_contour, detect_crescendo, detect_diminuendo,
    detect_subito, loudness_histogram, dynamic_range_score, analyze_dynamics,
)
