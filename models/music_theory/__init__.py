"""Music theory module — scales, intervals, chords, progressions, Camelot wheel, rhythm.

Pure music theory primitives for the music vertical. No ML, no medical code.
"""
from __future__ import annotations

from models.music_theory.intervals import (
    CONSONANCE_SCORES,
    ENHARMONIC_MAP,
    INTERVAL_NAMES,
    INTERVAL_TABLE,
    Interval,
    IntervalCalculator,
    PITCH_CLASSES,
    consonance_score,
    interval_between,
    interval_name,
    melodic_tension,
    normalize_note,
    note_to_index,
)
from models.music_theory.scales import (
    ALL_SCALE_TYPES,
    SCALE_INTERVALS,
    Scale,
    ScaleLibrary,
    detect_scale_from_notes,
    get_relative_major,
    get_relative_minor,
    get_scale,
    scale_degree_to_note,
)
from models.music_theory.chords import (
    CHORD_FORMULAS,
    Chord,
    ChordLibrary,
)
from models.music_theory.progressions import (
    COMMON_PROGRESSIONS,
    Progression,
    ProgressionAnalyzer,
    RomanNumeral,
)
from models.music_theory.camelot import (
    CamelotKey,
    camelot_compatibility,
    camelot_to_key,
    harmonic_mixing_score,
    key_to_camelot,
    suggest_transitions,
)
from models.music_theory.rhythm import (
    COMMON_METERS,
    GROOVE_TEMPLATES,
    RhythmAnalyzer,
    RhythmPattern,
    TimeSignature,
)
from models.music_theory.set_theory import (
    pitch_class_set, normal_form, prime_form, interval_vector,
    transpose as pc_transpose, invert as pc_invert,
    complement, is_subset, z_related,
    SetClass, FORTE_CATALOG_3, FORTE_CATALOG_4,
    lookup_forte, similarity_index as set_similarity,
)

__all__ = [
    "PITCH_CLASSES", "ENHARMONIC_MAP", "INTERVAL_NAMES", "CONSONANCE_SCORES",
    "Interval", "INTERVAL_TABLE", "IntervalCalculator",
    "normalize_note", "note_to_index", "interval_between", "interval_name",
    "consonance_score", "melodic_tension",
    "Scale", "ScaleLibrary", "SCALE_INTERVALS", "ALL_SCALE_TYPES", "get_scale",
    "get_relative_minor", "get_relative_major", "scale_degree_to_note",
    "detect_scale_from_notes",
    "Chord", "ChordLibrary", "CHORD_FORMULAS",
    "Progression", "COMMON_PROGRESSIONS", "ProgressionAnalyzer", "RomanNumeral",
    "CamelotKey", "key_to_camelot", "camelot_to_key",
    "camelot_compatibility", "harmonic_mixing_score", "suggest_transitions",
    "TimeSignature", "RhythmPattern", "COMMON_METERS", "GROOVE_TEMPLATES",
    "RhythmAnalyzer",
    "pitch_class_set", "normal_form", "prime_form", "interval_vector",
    "pc_transpose", "pc_invert", "complement", "is_subset", "z_related",
    "SetClass", "FORTE_CATALOG_3", "FORTE_CATALOG_4",
    "lookup_forte", "set_similarity",
    "VoiceLeadingRuleCV", "Voicing",
    "generate_voicings", "voice_leading_cost_cv", "smooth_voice_leading_cv",
    "check_parallel_fifths", "check_parallel_octaves",
    "voice_leading_quality", "drop2_voicing", "spread_voicing",
    "IntervalClass", "classify_interval", "is_consonant", "is_step",
    "MotionType", "motion_type",
    "CFError", "validate_cantus_firmus",
    "VoiceLeadingError", "check_first_species", "check_second_species",
    "voice_leading_score",
    "circle_of_fifths_distance", "relative_key", "parallel_key", "key_distance",
    "pivot_chords", "common_tone_count", "common_tone_modulation_score",
    "ChromaticMediant", "chromatic_mediant_relationship",
    "EnharmonicReinterpretation", "detect_enharmonic_reinterpretation",
    "ModulationPlan", "analyze_modulation",
    "Species", "MotionKind", "classify_motion", "cp_interval_class",
    "CounterpointViolation",
    "check_parallel_perfects", "check_hidden_fifths_octaves",
    "check_voice_crossing", "check_leap_resolution", "check_tritone_outline",
    "first_species_check", "counterpoint_score",
    "contrary_motion_ratio", "melodic_variety",
    "Triad", "neo_P", "neo_L", "neo_R",
    "apply_chain", "tonnetz_coordinates", "tonnetz_distance",
    "neo_shortest_path",
    "HEXATONIC_SYSTEMS", "hexatonic_system_for", "hexatonic_pole",
    "nebenverwandt", "neo_slide",
    "VoiceLeadingType", "VoiceLeading", "ChordTransition",
    "vl_classify_motion", "voice_leading_distance", "analyze_voice_leading",
    "smooth_vl", "optimal_voice_leading", "VoiceLeadingAnalyzer",
    "IL_IntervalQuality", "ConsonanceClass", "IL_Interval",
    "PERFECT_INTERVALS", "MAJOR_INTERVALS", "MINOR_INTERVALS",
    "CONSONANCE_RATING", "IL_INTERVAL_NAMES",
    "il_classify_interval", "il_interval_name",
    "invert_interval", "consonance_rating", "classify_consonance",
    "plomp_levelt_dissonance", "il_chord_dissonance",
    "frequency_ratio", "cents_from_ratio", "ratio_from_cents",
    "interval_vector_from_pitches", "all_intervals",
    "interval_histogram", "melodic_interval_profile",
    "consonance_curve", "average_consonance", "interval_entropy",
    "leap_step_ratio", "directional_change_rate",
    "ProlongationType", "StructuralLevel", "StructuralTone", "Ursatz", "ReductionResult",
    "pitch_to_degree", "degree_to_pc", "detect_urlinie", "detect_bassbrechung",
    "detect_neighbor", "detect_passing", "detect_arpeggiation",
    "reduce_foreground", "structural_weight", "analyze_structure",
    "tt_Quality", "tt_Triad", "tt_parallel", "tt_relative", "tt_leading_tone",
    "tt_nebenverwandt", "tt_slide", "tt_hexpole", "TT_OPERATIONS",
    "tt_apply_path", "tt_tonnetz_coords", "tt_tonnetz_distance",
    "plr_distance", "tt_hexatonic_system", "octatonic_pool", "weitzmann_region", "GIS",
]

from .counterpoint import (
    IntervalClass, classify_interval, is_consonant, is_step,
    MotionType, motion_type,
    CFError, validate_cantus_firmus,
    VoiceLeadingError, check_first_species, check_second_species,
    voice_leading_score,
)
from .modulation import (
    circle_of_fifths_distance, relative_key, parallel_key, key_distance,
    pivot_chords, common_tone_count, common_tone_modulation_score,
    ChromaticMediant, chromatic_mediant_relationship,
    EnharmonicReinterpretation, detect_enharmonic_reinterpretation,
    ModulationPlan, analyze_modulation,
)
from .chord_voicing import (
    VoiceLeadingRule as VoiceLeadingRuleCV, Voicing,
    generate_voicings, voice_leading_cost as voice_leading_cost_cv,
    smooth_voice_leading as smooth_voice_leading_cv,
    check_parallel_fifths, check_parallel_octaves,
    voice_leading_quality, drop2_voicing, spread_voicing,
)
from .counterpoint_rules import (
    Species, MotionKind, classify_motion,
    interval_class as cp_interval_class,
    CounterpointViolation,
    check_parallel_perfects, check_hidden_fifths_octaves,
    check_voice_crossing, check_leap_resolution, check_tritone_outline,
    first_species_check, counterpoint_score,
    contrary_motion_ratio, melodic_variety,
)
from .voice_leading import (
    VoiceLeadingType, VoiceLeading, ChordTransition,
    classify_motion as vl_classify_motion,
    voice_leading_distance, analyze_voice_leading,
    smooth_voice_leading as smooth_vl,
    optimal_voice_leading, VoiceLeadingAnalyzer,
)
from .neo_riemannian import (
    Triad, P as neo_P, L as neo_L, R as neo_R,
    apply_chain, tonnetz_coordinates, tonnetz_distance,
    shortest_path as neo_shortest_path,
    HEXATONIC_SYSTEMS, hexatonic_system_for, hexatonic_pole,
    nebenverwandt, slide as neo_slide,
)
from .interval_logic import (
    IntervalQuality as IL_IntervalQuality,
    ConsonanceClass, Interval as IL_Interval,
    PERFECT_INTERVALS, MAJOR_INTERVALS, MINOR_INTERVALS,
    CONSONANCE_RATING, INTERVAL_NAMES as IL_INTERVAL_NAMES,
    classify_interval as il_classify_interval,
    interval_name as il_interval_name,
    invert_interval, consonance_rating, classify_consonance,
    plomp_levelt_dissonance, chord_dissonance as il_chord_dissonance,
    frequency_ratio, cents_from_ratio, ratio_from_cents,
    interval_vector_from_pitches, all_intervals,
    interval_histogram, melodic_interval_profile,
    consonance_curve, average_consonance, interval_entropy,
    leap_step_ratio, directional_change_rate,
)
from .schenkerian_analysis import (
    ProlongationType, StructuralLevel, StructuralTone, Ursatz, ReductionResult,
    pitch_to_degree, degree_to_pc, detect_urlinie, detect_bassbrechung,
    detect_neighbor, detect_passing, detect_arpeggiation,
    reduce_foreground, structural_weight, analyze_structure,
)
from .transformational_theory import (
    Quality as tt_Quality, Triad as tt_Triad,
    parallel as tt_parallel, relative as tt_relative,
    leading_tone as tt_leading_tone, nebenverwandt as tt_nebenverwandt,
    slide as tt_slide, hexpole as tt_hexpole,
    OPERATIONS as TT_OPERATIONS,
    apply_path as tt_apply_path,
    tonnetz_coords as tt_tonnetz_coords, tonnetz_distance as tt_tonnetz_distance,
    plr_distance, hexatonic_system as tt_hexatonic_system,
    octatonic_pool, weitzmann_region, GIS,
)
