"""Music arrangement: orchestration, scoring, voicing, texture, reduction."""

from .orchestration import (
    InstrumentFamily, InstrumentRange, DoublingEvent, OrchestrationProfile,
    ORCHESTRA_RANGES, BLEND_COMPATIBILITY,
    fits_range, suggest_transposition,
    doubling_score, blend_score, register_overlap,
    voicing_density, detect_doublings, family_energy_distribution,
    orchestral_balance, orchestral_balance_score,
    compute_orchestration_profile,
)
from .scoring import (
    ScoreFormat, reduce_to_piano, expand_lead_sheet,
    part_extraction, transpose_part,
)
from .texture_mapping import (
    TextureType, classify_texture,
    texture_density, recommended_texture,
)
from .reharmonization import (
    tritone_substitution, backdoor_progression,
    modal_interchange, chromatic_mediant,
    reharmonize_melody_note,
)
from .voice_arrangement import (
    VoiceArrangement, voicing_spread, voicing_density,
    detect_doubling, arrangement_register, voice_crossing_count,
    spacing_quality, optimal_doubling, register_balance, open_close_voicing,
)
from .voice_leading import (
    VoiceLeadingRule, MotionType, VoiceLeadingError, SpacingReport,
    classify_motion, interval_class,
    check_parallel_motion, detect_hidden_fifths_octaves,
    voice_crossing_check, detect_voice_overlap, detect_large_leaps,
    spacing_analysis, motion_profile,
    smooth_voice_leading, voice_leading_cost_matrix,
    optimal_voicing, total_voice_motion, voice_leading_score,
)
from .arranging_styles import (
    ArrangingStyle, StyleTemplate, STYLE_TEMPLATES,
    assign_voices, double_at_octave, unison_passage, tutti_score,
)

__all__ = [
    "InstrumentFamily", "InstrumentRange", "DoublingEvent", "OrchestrationProfile",
    "ORCHESTRA_RANGES", "BLEND_COMPATIBILITY",
    "fits_range", "suggest_transposition",
    "doubling_score", "blend_score", "register_overlap",
    "voicing_density", "detect_doublings", "family_energy_distribution",
    "orchestral_balance", "orchestral_balance_score", "compute_orchestration_profile",
    "ScoreFormat", "reduce_to_piano", "expand_lead_sheet",
    "part_extraction", "transpose_part",
    "TextureType", "classify_texture", "texture_density", "recommended_texture",
    "tritone_substitution", "backdoor_progression",
    "modal_interchange", "chromatic_mediant", "reharmonize_melody_note",
    "VoiceArrangement", "voicing_spread",
    "detect_doubling", "arrangement_register", "voice_crossing_count",
    "spacing_quality", "optimal_doubling", "register_balance", "open_close_voicing",
    "VoiceLeadingRule", "MotionType", "VoiceLeadingError", "SpacingReport",
    "classify_motion", "interval_class",
    "check_parallel_motion", "detect_hidden_fifths_octaves",
    "voice_crossing_check", "detect_voice_overlap", "detect_large_leaps",
    "spacing_analysis", "motion_profile",
    "smooth_voice_leading", "voice_leading_cost_matrix",
    "optimal_voicing", "total_voice_motion", "voice_leading_score",
    "ArrangingStyle", "StyleTemplate", "STYLE_TEMPLATES",
    "assign_voices", "double_at_octave", "unison_passage", "tutti_score",
    "InstrumentProfile", "COMMON_INSTRUMENTS",
    "timbral_distance", "timbral_compatibility",
    "register_complement", "combination_score", "family_diversity",
    "suggest_complement", "blend_brightness", "total_range",
    "ScoreNote", "PianoReduction", "piano_reduce",
    "extract_melody", "extract_bass_line", "condense_parts",
    "lead_sheet", "reduction_quality", "voice_count_over_time",
    "InstrumentDoubling", "COMMON_DOUBLINGS",
    "VoicingConstraint", "VOICING_RULES", "POWER_DB",
    "score_voicing", "suggest_doublings",
    "orchestral_balance_detailed", "REGISTER_CURVES", "register_effectiveness",
    "BrassInstrument", "BRASS_RANGES", "VoicingTechnique", "BrassVoicing",
    "close_voicing", "drop2", "drop24", "spread_voicing",
    "assign_brass_instruments", "brass_range_check",
    "voicing_tension", "parallel_motion",
    "ReductionType", "NoteEvent", "ReducedScore",
    "sr_extract_melody", "sr_extract_bass",
    "piano_reduction", "lead_sheet_extraction",
    "voice_importance", "sr_texture_density", "adaptive_thinning",
    "OEInstrumentFamily", "InstrumentSpec",
    "oe_in_range", "range_utilization", "tessitura_comfort",
    "timbre_compatibility", "assign_voices", "oe_doubling_score",
    "voice_crossing_count", "spacing_score", "orchestration_score",
]

from .instrument_combination import (
    InstrumentFamily, InstrumentProfile, COMMON_INSTRUMENTS,
    timbral_distance, timbral_compatibility, range_overlap,
    register_complement, combination_score, family_diversity,
    suggest_complement, blend_brightness, total_range,
)
from .reduction import (
    ScoreNote, PianoReduction, piano_reduce,
    extract_melody, extract_bass_line, condense_parts,
    lead_sheet, reduction_quality, voice_count_over_time,
)
from .orchestration_rules import (
    InstrumentDoubling, COMMON_DOUBLINGS,
    VoicingConstraint, VOICING_RULES, POWER_DB,
    score_voicing, suggest_doublings,
    orchestral_balance as orchestral_balance_detailed,
    REGISTER_CURVES, register_effectiveness,
)
from .brass_voicing import (
    BrassInstrument, BRASS_RANGES, VoicingTechnique, BrassVoicing,
    close_voicing, drop2, drop24, spread_voicing,
    assign_instruments as assign_brass_instruments,
    range_check as brass_range_check,
    voicing_tension, parallel_motion,
)
from .score_reduction import (
    ReductionType, NoteEvent, ReducedScore,
    extract_melody as sr_extract_melody,
    extract_bass as sr_extract_bass,
    piano_reduction, lead_sheet_extraction,
    voice_importance, texture_density as sr_texture_density,
    adaptive_thinning,
)
from .orchestration_engine import (
    InstrumentFamily as OEInstrumentFamily, InstrumentSpec,
    in_range as oe_in_range, range_utilization, tessitura_comfort,
    timbre_compatibility, assign_voices, doubling_score as oe_doubling_score,
    voice_crossing_count, spacing_score, orchestration_score,
)
from .texture_evolution import (
    TextureCategory, RegisterBand, REGISTER_BOUNDS, RK_IDEAL_BALANCE,
    TextureSnapshot, CollaParteEvent, TextureEvolution,
    classify_texture, note_to_register, register_distribution,
    rk_balance_score, muddy_interval_count, detect_colla_parte,
    instrumentation_density_curve, register_weight_curve,
    texture_change_points, analyze_texture_evolution,
    rk_balance_summary, texture_variety_index, colla_parte_fraction,
)
