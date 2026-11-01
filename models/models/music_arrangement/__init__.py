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
