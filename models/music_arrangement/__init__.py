"""Music arrangement: orchestration, scoring, voicing, texture, reduction."""

from .orchestration import (
    InstrumentRange, ORCHESTRA_RANGES,
    fits_range, suggest_transposition,
    doubling_score, orchestral_balance,
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
    VoiceLeadingRule, check_parallel_motion, voice_crossing_check,
    smooth_voice_leading, voice_leading_cost_matrix,
    optimal_voicing, total_voice_motion,
)
from .arranging_styles import (
    ArrangingStyle, StyleTemplate, STYLE_TEMPLATES,
    assign_voices, double_at_octave, unison_passage, tutti_score,
)

__all__ = [
    "InstrumentRange", "ORCHESTRA_RANGES",
    "fits_range", "suggest_transposition", "doubling_score", "orchestral_balance",
    "ScoreFormat", "reduce_to_piano", "expand_lead_sheet",
    "part_extraction", "transpose_part",
    "TextureType", "classify_texture", "texture_density", "recommended_texture",
    "tritone_substitution", "backdoor_progression",
    "modal_interchange", "chromatic_mediant", "reharmonize_melody_note",
    "VoiceArrangement", "voicing_spread", "voicing_density",
    "detect_doubling", "arrangement_register", "voice_crossing_count",
    "spacing_quality", "optimal_doubling", "register_balance", "open_close_voicing",
    "VoiceLeadingRule", "check_parallel_motion", "voice_crossing_check",
    "smooth_voice_leading", "voice_leading_cost_matrix",
    "optimal_voicing", "total_voice_motion",
    "ArrangingStyle", "StyleTemplate", "STYLE_TEMPLATES",
    "assign_voices", "double_at_octave", "unison_passage", "tutti_score",
    "InstrumentFamily", "InstrumentProfile", "COMMON_INSTRUMENTS",
    "timbral_distance", "timbral_compatibility", "range_overlap",
    "register_complement", "combination_score", "family_diversity",
    "suggest_complement", "blend_brightness", "total_range",
]

from .instrument_combination import (
    InstrumentFamily, InstrumentProfile, COMMON_INSTRUMENTS,
    timbral_distance, timbral_compatibility, range_overlap,
    register_complement, combination_score, family_diversity,
    suggest_complement, blend_brightness, total_range,
)
