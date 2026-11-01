"""Music improvisation: scale navigation, motivic development, call-response, jazz voicings."""

from .scale_navigation import (
    ScaleMode, scale_degrees, chord_tones_in_scale,
    available_tensions, approach_note_options,
    target_note_gravity,
)
from .motivic_development import (
    Motif, transpose_motif, invert_motif, retrograde_motif,
    augment_motif, diminish_motif, sequence_motif,
    motivic_similarity,
)
from .call_response import (
    call_response_score, rhythmic_complement,
    melodic_answer, phrase_balance,
)
from .jazz_voicings import (
    VoicingType, voice_chord, rootless_voicing,
    drop2_voicing, shell_voicing, quartal_voicing,
    voice_leading_cost,
)

from .phrase_generation import (
    PhraseContour, classify_contour, generate_tension_arc,
    rhythmic_variation, pitch_displacement, phrase_density,
    intervallic_variety, build_phrase,
)

from .jazz_improv import (
    CHORD_SCALE_MAP, AVAILABLE_TENSIONS, GUIDE_TONES,
    ChordContext, available_pitches, guide_tone_line,
    sequence, fragment, augment_rhythm, diminish_rhythm,
    invert_motive, retrograde,
    CallResponse, generate_response,
    tension_profile, phrase_contour,
)

__all__ = [
    "ScaleMode", "scale_degrees", "chord_tones_in_scale",
    "available_tensions", "approach_note_options", "target_note_gravity",
    "Motif", "transpose_motif", "invert_motif", "retrograde_motif",
    "augment_motif", "diminish_motif", "sequence_motif", "motivic_similarity",
    "call_response_score", "rhythmic_complement", "melodic_answer", "phrase_balance",
    "VoicingType", "voice_chord", "rootless_voicing",
    "drop2_voicing", "shell_voicing", "quartal_voicing", "voice_leading_cost",
    "CHORD_SCALE_MAP", "AVAILABLE_TENSIONS", "GUIDE_TONES",
    "ChordContext", "available_pitches", "guide_tone_line",
    "sequence", "fragment", "augment_rhythm", "diminish_rhythm",
    "invert_motive", "retrograde",
    "CallResponse", "generate_response",
    "tension_profile", "phrase_contour",
    "PhraseContour", "classify_contour", "generate_tension_arc",
    "rhythmic_variation", "pitch_displacement", "phrase_density",
    "intervallic_variety", "build_phrase",
    "Mode", "MODE_INTERVALS", "mode_pitch_classes", "common_tones",
    "mode_distance", "InterchangeOption", "find_interchange_options",
    "borrowed_chord_tones", "pivot_note", "interchange_tension",
    "InterchangeSequence", "build_interchange_sequence",
    "RhythmFeel", "CLAVE_PATTERNS", "RhythmicEvent",
    "swing_offset", "apply_swing", "syncopation_score",
    "metric_displacement", "clave_alignment",
    "generate_clave_pattern", "tresillo_pattern",
    "anticipation_pattern", "density_curve",
    "OrnamentType", "Ornament", "trill", "mordent", "turn", "grace_note",
    "appoggiatura", "acciaccatura", "tremolo", "glissando_pitches",
    "ornament_duration", "apply_ornament_timing",
    "sensory_dissonance", "voice_leading_distance", "tonal_distance",
    "TensionPoint", "tension_curve", "tension_release_ratio",
    "peak_tension_position", "mean_tension", "tension_variance",
    "LickTemplate", "LickMatch",
    "BEBOP_LICKS", "BLUES_LICKS", "COMMON_PATTERNS", "ALL_TEMPLATES",
    "match_score", "find_licks", "vocabulary_richness", "suggest_next_lick",
    "note_choice_distribution", "ApproachEvent", "detect_approaches",
    "approach_type_distribution", "detect_enclosures",
    "RegisterProfile", "register_profile",
    "rhythmic_density_curve", "interval_histogram",
    "SoloSummary", "summarise_solo",
]

from .modal_interchange_improv import (
    Mode, MODE_INTERVALS, mode_pitch_classes, common_tones,
    mode_distance, InterchangeOption, find_interchange_options,
    borrowed_chord_tones, pivot_note, interchange_tension,
    InterchangeSequence, build_interchange_sequence,
)
from .ornaments import (
    OrnamentType, Ornament, trill, mordent, turn, grace_note,
    appoggiatura, acciaccatura, tremolo, glissando_pitches,
    ornament_duration, apply_ornament_timing,
)
from .rhythm_patterns import (
    RhythmFeel, CLAVE_PATTERNS, RhythmicEvent,
    swing_offset, apply_swing, syncopation_score,
    metric_displacement, clave_alignment,
    generate_clave_pattern, tresillo_pattern,
    anticipation_pattern, density_curve,
)
from .tension_curve import (
    sensory_dissonance, voice_leading_distance, tonal_distance,
    TensionPoint, tension_curve, tension_release_ratio,
    peak_tension_position, mean_tension, tension_variance,
)
from .lick_library import (
    LickTemplate, LickMatch,
    BEBOP_LICKS, BLUES_LICKS, COMMON_PATTERNS, ALL_TEMPLATES,
    match_score, find_licks, vocabulary_richness, suggest_next_lick,
)
from .solo_analysis import (
    note_choice_distribution, ApproachEvent, detect_approaches,
    approach_type_distribution, detect_enclosures,
    RegisterProfile, register_profile,
    rhythmic_density_curve, interval_histogram,
    SoloSummary, summarise_solo,
)
from .jazz_harmony import (
    ChordTone, CHORD_INTERVALS, chord_tones as jazz_chord_tones,
    available_tensions as jazz_available_tensions,
    guide_tone_line as jazz_guide_tone_line,
    approach_note as jazz_approach_note,
    ii_V_I_voicings, tritone_substitution,
    backdoor_progression, coltrane_changes, tension_resolution_score,
)
