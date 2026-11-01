"""Music harmony: chord analysis, progressions, voice leading, functional harmony, tension, key detection."""
from models.music_harmony.chords import (
    Chord, ChordQuality, chord_from_name, chord_from_pitches,
    CHORD_TYPES, chord_similarity,
)
from models.music_harmony.progressions import (
    ChordProgression, RomanNumeral, COMMON_PROGRESSIONS,
    analyze_roman_numerals, progression_complexity, progression_tension_curve,
)
from models.music_harmony.voice_leading import (
    VoiceLeadingRules, voice_lead, voice_leading_cost,
    parallel_fifths_check, parallel_octaves_check,
)
from models.music_harmony.functional import (
    HarmonicFunction, classify_function, cadence_type,
    CADENCE_TYPES, tonicization_target,
)
from models.music_harmony.modulation import (
    Key, ModulationType, ModulationEvent, Tonicization,
    key_distance_fifths, key_distance_shared_pcs,
    find_pivot_chords, classify_modulation,
    detect_tonicizations, ModulationAnalyzer, key_distance_matrix,
)
from models.music_harmony.key_profile import (
    KeyEstimate, pitch_class_histogram, detect_key,
    key_clarity, windowed_key_detection, key_stability,
    pitch_class_entropy,
    KRUMHANSL_MAJOR, KRUMHANSL_MINOR,
    TEMPERLEY_MAJOR, TEMPERLEY_MINOR,
    AARDEN_MAJOR, AARDEN_MINOR,
    PROFILE_SETS,
)
from models.music_harmony.tension import (
    TensionProfile, interval_dissonance, chord_dissonance,
    roughness_pair, chord_roughness, chord_roughness_from_pitches,
    pitch_to_freq, tension_profile, resolution_strength,
    harmonic_rhythm_density, harmonic_rhythm_regularity,
    tension_contour_features,
)

__all__ = [
    "Chord", "ChordQuality", "chord_from_name", "chord_from_pitches",
    "CHORD_TYPES", "chord_similarity",
    "ChordProgression", "RomanNumeral", "COMMON_PROGRESSIONS",
    "analyze_roman_numerals", "progression_complexity", "progression_tension_curve",
    "VoiceLeadingRules", "voice_lead", "voice_leading_cost",
    "parallel_fifths_check", "parallel_octaves_check",
    "HarmonicFunction", "classify_function", "cadence_type",
    "CADENCE_TYPES", "tonicization_target",
    "Key", "ModulationType", "ModulationEvent", "Tonicization",
    "key_distance_fifths", "key_distance_shared_pcs",
    "find_pivot_chords", "classify_modulation",
    "detect_tonicizations", "ModulationAnalyzer", "key_distance_matrix",
    "KeyEstimate", "pitch_class_histogram", "detect_key",
    "key_clarity", "windowed_key_detection", "key_stability",
    "pitch_class_entropy",
    "KRUMHANSL_MAJOR", "KRUMHANSL_MINOR",
    "TEMPERLEY_MAJOR", "TEMPERLEY_MINOR",
    "AARDEN_MAJOR", "AARDEN_MINOR",
    "PROFILE_SETS",
    "TensionProfile", "interval_dissonance", "chord_dissonance",
    "roughness_pair", "chord_roughness", "chord_roughness_from_pitches",
    "pitch_to_freq", "tension_profile", "resolution_strength",
    "harmonic_rhythm_density", "harmonic_rhythm_regularity",
    "tension_contour_features",
    "TriadQuality", "Triad",
    "parallel", "leading_tone", "relative",
    "apply_sequence", "hexatonic_cycle", "octatonic_cycle",
    "tonnetz_coordinates", "triad_tonnetz_centroid", "tonnetz_distance",
    "voice_leading_distance", "plr_distance",
    "HarmonicPathStats", "analyze_harmonic_path",
    "IntervalQuality", "MotionType", "CantusFirmusAnalysis",
    "CounterpointAnalysis", "SuspensionPattern",
    "classify_interval", "interval_between", "classify_motion",
    "analyze_cantus_firmus", "detect_parallel_fifths_cp", "detect_parallel_octaves_cp",
    "voice_crossings", "analyze_counterpoint", "detect_suspensions",
    "voice_independence", "step_skip_ratio",
    "PCSet", "to_pc_set", "transpose", "invert", "complement",
    "JazzChordType", "JazzMode", "JazzChord", "IIVIProgression",
    "VoicingResult", "CHORD_INTERVALS", "MODE_INTERVALS",
    "detect_ii_v_i", "tritone_substitute", "is_tritone_sub",
    "shell_voicing", "rootless_voicing_a", "rootless_voicing_b",
    "drop_2_voicing", "drop_3_voicing", "chord_scale", "scale_pitches",
    "available_tensions", "jh_voice_leading_distance", "smooth_voice_lead",
    "bebop_chromatic_approach", "guide_tone_line",
    "backdoor_progression", "coltrane_changes", "turnaround", "harmonic_density",
    "normal_form", "prime_form", "interval_vector",
    "is_z_related", "is_subset", "is_superset",
    "common_tones_under_transposition", "similarity_index",
    "FORTE_CATALOG_3", "FORTE_CATALOG_4", "lookup_forte", "all_subsets",
    "fa_HarmonicFunction", "fa_CadenceType", "fa_RomanNumeral", "fa_FunctionalAnalysis",
    "fa_FUNCTION_MAP",
    "pitch_class_set_to_chord_quality", "chord_to_roman_numeral",
    "detect_cadence", "harmonic_tension", "detect_tonicization",
    "fa_tension_curve", "functional_rhythm", "analyze_progression",
    "harmonic_surprise",
]

from models.music_harmony.functional_analysis import (
    HarmonicFunction as fa_HarmonicFunction,
    CadenceType as fa_CadenceType,
    RomanNumeral as fa_RomanNumeral,
    FunctionalAnalysis as fa_FunctionalAnalysis,
    FUNCTION_MAP as fa_FUNCTION_MAP,
    pitch_class_set_to_chord_quality,
    chord_to_roman_numeral,
    detect_cadence,
    harmonic_tension,
    detect_tonicization,
    tension_curve as fa_tension_curve,
    functional_rhythm,
    analyze_progression,
    harmonic_surprise,
)

from models.music_harmony.set_theory import (
    PCSet, to_pc_set, transpose, invert, complement,
    normal_form, prime_form, interval_vector,
    is_z_related, is_subset, is_superset,
    common_tones_under_transposition, similarity_index,
    FORTE_CATALOG_3, FORTE_CATALOG_4, lookup_forte, all_subsets,
)

from models.music_harmony.neo_riemannian import (
    TriadQuality, Triad,
    parallel, leading_tone, relative,
    apply_sequence, hexatonic_cycle, octatonic_cycle,
    tonnetz_coordinates, triad_tonnetz_centroid, tonnetz_distance,
    voice_leading_distance, plr_distance,
    HarmonicPathStats, analyze_harmonic_path,
)
from models.music_harmony.voice_leading_optimization import (
    pitch_class_distance, optimal_voice_leading,
    NeoRiemannianOp, neo_riemannian_transform, neo_riemannian_distance,
    VoiceLeadingPath, analyze_voice_leading_path, parsimonious_realization,
)
from models.music_harmony.harmonic_rhythm import (
    HarmonicEvent, HarmonicRhythmProfile, chord_change_rate,
    on_beat_ratio, harmonic_acceleration, acceleration_slope,
    phrase_harmonic_density, cadential_rhythm_score, harmonic_syncopation,
    harmonic_regularity, compute_harmonic_rhythm_profile,
    harmonic_rhythm_distance, classify_harmonic_rhythm,
)
from models.music_harmony.jazz_harmony import (
    JazzChordType, JazzMode, JazzChord, IIVIProgression,
    VoicingResult, CHORD_INTERVALS, MODE_INTERVALS,
    detect_ii_v_i, tritone_substitute, is_tritone_sub,
    shell_voicing, rootless_voicing_a, rootless_voicing_b,
    drop_2_voicing, drop_3_voicing, chord_scale, scale_pitches,
    available_tensions, voice_leading_distance as jh_voice_leading_distance,
    smooth_voice_lead, bebop_chromatic_approach, guide_tone_line,
    backdoor_progression, coltrane_changes, turnaround, harmonic_density,
)
from models.music_harmony.counterpoint import (
    IntervalQuality, MotionType, CantusFirmusAnalysis,
    CounterpointAnalysis, SuspensionPattern,
    classify_interval, interval_between, classify_motion,
    analyze_cantus_firmus,
    detect_parallel_fifths as detect_parallel_fifths_cp,
    detect_parallel_octaves as detect_parallel_octaves_cp,
    voice_crossings, analyze_counterpoint, detect_suspensions,
    voice_independence, step_skip_ratio,
)
from models.music_harmony.deep_harmony import (
    plomp_levelt_curve, dyad_roughness, spectrum_roughness,
    chord_roughness_harmonic, sensory_consonance_curve,
    MAJOR_BASIC_SPACE, MINOR_BASIC_SPACE,
    tps_chord_distance, tps_sequential_tension, tps_hierarchical_tension,
    TPSAnalysis,
    melodic_attraction, harmonic_attraction,
    ProlongationType, ReductionNode, classify_prolongation,
    detect_ursatz, schenkerian_reduce,
    CadenceStrength, CadenceAnalysis, analyze_cadence,
    chromatic_mediant_type, omnibus_progression, equal_division,
    enharmonic_pivot_options,
    chord_transition_matrix, information_content,
    harmonic_entropy_rate, harmonic_surprise_curve,
    classify_augmented_sixth,
    DeepHarmonicAnalysis, analyze_harmony_deep,
)
