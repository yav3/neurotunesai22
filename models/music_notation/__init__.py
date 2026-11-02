"""Music notation: ABC/MusicXML parsing, score representation, analysis, staff layout, transposition."""
from .abc_parser import (
    ABCHeader,
    ABCNote,
    ABCTune,
    ABCParser,
    ABCWriter,
)
from .musicxml import (
    MXLNote,
    MXLMeasure,
    MXLPart,
    MXLScore,
    MXLParser,
    MXLWriter,
)
from .score import (
    PITCH_CLASSES,
    Note,
    Rest,
    Measure,
    Score,
    transpose_score,
)
from .score_analysis import (
    ScoreStatistics,
    VoiceLeading,
    MelodicAnalysis,
    HarmonicRhythm,
)
from .staff_layout import (
    Clef,
    StaffLine,
    NoteHead,
    Beam,
    StaffLayout,
)
from .transposition import (
    Transposer,
    InstrumentTransposition,
    KeySignatureTransposer,
    EnharmonicSpeller,
)

__all__ = [
    "ABCHeader", "ABCNote", "ABCTune", "ABCParser", "ABCWriter",
    "MXLNote", "MXLMeasure", "MXLPart", "MXLScore", "MXLParser", "MXLWriter",
    "PITCH_CLASSES", "Note", "Rest", "Measure", "Score", "transpose_score",
    "ScoreStatistics", "VoiceLeading", "MelodicAnalysis", "HarmonicRhythm",
    "Clef", "StaffLine", "NoteHead", "Beam", "StaffLayout",
    "Transposer", "InstrumentTransposition", "KeySignatureTransposer",
    "EnharmonicSpeller",
    "LilyArticulation", "LilyNote", "LilyMeasure", "LilyScore",
    "PITCH_TO_LILY", "LILY_TO_PITCH",
    "midi_to_lily_note", "notes_to_score",
    "parse_lily_pitch", "transpose_lily",
    "ClefType", "NoteValue", "AdvNoteHead", "BeamGroup", "AdvStaffLayout",
    "pitch_to_staff_y", "adv_stem_direction",
    "optimal_stem_length", "beam_angle", "adv_note_spacing",
    "adv_collision_detect", "resolve_collisions",
    "accidental_layout", "ledger_lines", "measure_width", "justify_measure",
    "engrave_system",
    "NoteEvent", "BeamGroupNI", "SpacingResult",
    "quantise_duration", "duration_to_note_value", "needs_tie",
    "split_tied_note", "infer_ties", "beam_group_notes", "assign_beam_groups",
    "voice_separate", "proportional_spacing", "optimal_clef",
    "detect_clef_changes", "note_collision_detect_ni",
    "stem_direction_ni", "accidental_needed",
    "ChordQuality", "ChordSymbol", "parse_chord_symbol", "chord_to_text",
    "cs_chord_pitches", "chord_to_midi", "cs_transpose_chord",
    "enharmonic_root", "cs_chord_complexity", "cs_common_tones",
    "cs_interval_content", "cs_classify_function",
    "InstrumentTuning", "TabNote", "TabMeasure",
    "midi_to_fret_options", "fret_distance", "hand_stretch", "position_penalty",
    "optimize_fingering", "chord_fingering", "tab_to_ascii",
    "playability_score", "capo_transpose", "suggest_capo",
    "string_usage_distribution", "fret_histogram", "tab_difficulty_estimate",
    "ArticulationType", "ArticulationEffect", "NoteWithArticulation",
    "apply_articulation", "trill_expansion", "mordent_expansion",
    "turn_expansion", "tremolo_expansion", "articulation_envelope",
    "detect_articulation", "articulation_density", "combined_effect",
]

from .notation_intelligence import (
    NoteEvent, BeamGroup as BeamGroupNI, SpacingResult,
    quantise_duration, duration_to_note_value, needs_tie,
    split_tied_note, infer_ties, beam_group_notes, assign_beam_groups,
    voice_separate, proportional_spacing, optimal_clef,
    detect_clef_changes, note_collision_detect as note_collision_detect_ni,
    stem_direction as stem_direction_ni, accidental_needed,
)

from .lilypond import (
    Articulation as LilyArticulation,
    LilyNote, LilyMeasure, LilyScore,
    PITCH_TO_LILY, LILY_TO_PITCH,
    midi_to_lily_note, notes_to_score,
    parse_lily_pitch, transpose_lily,
)
from .engraving import (
    StaffMetrics, compute_note_spacing, stem_direction, beam_slope,
    collision_detect, lyric_hyphenation_points, page_break_penalty,
    accidental_stacking, tie_arc_height, optimal_system_breaks,
)
from .advanced_engraving import (
    ClefType, NoteValue, NoteHead as AdvNoteHead, BeamGroup,
    StaffLayout as AdvStaffLayout,
    pitch_to_staff_y, stem_direction as adv_stem_direction,
    optimal_stem_length, beam_angle, note_spacing as adv_note_spacing,
    collision_detect as adv_collision_detect, resolve_collisions,
    accidental_layout, ledger_lines, measure_width, justify_measure,
    engrave_system,
)
from .chord_symbols import (
    ChordQuality, ChordSymbol, parse_chord_symbol, chord_to_text,
    chord_pitches as cs_chord_pitches, chord_to_midi,
    transpose_chord as cs_transpose_chord, enharmonic_root,
    chord_complexity as cs_chord_complexity,
    common_tones as cs_common_tones, interval_content as cs_interval_content,
    classify_function as cs_classify_function,
)
from .tablature import (
    InstrumentTuning, TabNote, TabMeasure,
    midi_to_fret_options, fret_distance, hand_stretch, position_penalty,
    optimize_fingering, chord_fingering, tab_to_ascii,
    playability_score, capo_transpose, suggest_capo,
    string_usage_distribution, fret_histogram,
    difficulty_estimate as tab_difficulty_estimate,
)
from .articulation_marks import (
    ArticulationType, ArticulationEffect, NoteWithArticulation,
    apply_articulation, trill_expansion, mordent_expansion,
    turn_expansion, tremolo_expansion, articulation_envelope,
    detect_articulation, articulation_density, combined_effect,
)
