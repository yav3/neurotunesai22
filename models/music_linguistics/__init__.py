"""Music linguistics: phonetics, prosody, poetic forms, lyric structure, cross-lingual analysis."""
from models.music_linguistics.phonetics import (
    IPA_VOWELS, IPA_CONSONANTS, PhonemeInventory, INVENTORIES,
    count_syllables_english, count_syllables_spanish, count_syllables_japanese,
    singability_score,
)
from models.music_linguistics.prosody import (
    Foot, RhymeType, MeterResult, RhymeSchemeResult,
    stress_pattern, classify_feet, analyze_meter,
    classify_rhyme, detect_rhyme_scheme,
    detect_enjambment, detect_caesura,
)
from models.music_linguistics.poetic_forms import (
    MeterConstraint, StanzaTemplate, PoeticForm, FORM_REGISTRY,
    get_form, list_forms, forms_by_origin,
    validate_line_count, validate_syllable_counts,
    SONNET_SHAKESPEAREAN, SONNET_PETRARCHAN, HAIKU, TANKA, GHAZAL,
    VILLANELLE, SESTINA, LIMERICK, TERZA_RIMA, PANTOUM, BLUES_STANZA,
)
from models.music_linguistics.lyric_structure import (
    SectionType, LyricSection, LyricStructure,
    detect_sections, find_repeated_lines, hook_density,
    SyllableAlignment, compute_syllable_alignment,
    classify_song_form, COMMON_FORMS,
)
from models.music_linguistics.cross_lingual import (
    ScriptFamily, detect_script, detect_scripts, dominant_script,
    LanguageProfile, LANGUAGE_PROFILES,
    transliterate, ROMANIZATION_TABLES,
    SingabilityProfile, estimate_singing_difficulty,
    TextStats, analyze_text,
)

__all__ = [
    "IPA_VOWELS", "IPA_CONSONANTS", "PhonemeInventory", "INVENTORIES",
    "count_syllables_english", "count_syllables_spanish", "count_syllables_japanese",
    "singability_score",
    "Foot", "RhymeType", "MeterResult", "RhymeSchemeResult",
    "stress_pattern", "classify_feet", "analyze_meter",
    "classify_rhyme", "detect_rhyme_scheme",
    "detect_enjambment", "detect_caesura",
    "MeterConstraint", "StanzaTemplate", "PoeticForm", "FORM_REGISTRY",
    "get_form", "list_forms", "forms_by_origin",
    "validate_line_count", "validate_syllable_counts",
    "SONNET_SHAKESPEAREAN", "SONNET_PETRARCHAN", "HAIKU", "TANKA", "GHAZAL",
    "VILLANELLE", "SESTINA", "LIMERICK", "TERZA_RIMA", "PANTOUM", "BLUES_STANZA",
    "SectionType", "LyricSection", "LyricStructure",
    "detect_sections", "find_repeated_lines", "hook_density",
    "SyllableAlignment", "compute_syllable_alignment",
    "classify_song_form", "COMMON_FORMS",
    "ScriptFamily", "detect_script", "detect_scripts", "dominant_script",
    "LanguageProfile", "LANGUAGE_PROFILES",
    "transliterate", "ROMANIZATION_TABLES",
    "SingabilityProfile", "estimate_singing_difficulty",
    "TextStats", "analyze_text",
    "MUSIC_SENTIMENT_LEXICON", "SentimentScore", "score_text",
    "sentiment_trajectory", "valence_arc", "arousal_arc",
    "emotional_volatility", "dominant_emotion",
    "NarrativePhase", "NarrativeArc", "SemanticShift", "ThematicProfile",
    "estimate_narrative_arc", "semantic_shift_between", "pairwise_semantic_shifts",
    "thematic_profile", "repetition_density", "lexical_density", "imagery_score",
    "RhymeQuality", "RhymeScore", "SchemeAnalysis",
    "score_rhyme", "analyze_rhyme_scheme", "internal_rhyme_density",
    "multisyllabic_rhyme_score",
    "AdaptationDimension", "AdaptationScore", "SyllableMap", "VowelProfile",
    "syllable_map", "syllable_fidelity", "vowel_openness_similarity",
    "stress_alignment_score", "rhyme_preservation_score", "semantic_fidelity_score",
    "score_adaptation", "back_translation_consistency", "note_density_ratio",
    "PaintingGesture", "PaintingSuggestion", "RepetitionPainting",
    "detect_painting_words", "direction_to_contour", "dynamic_curve_from_gestures",
    "suggest_painting", "painting_density",
    "melodic_interval_from_direction", "tempo_adjustment",
    "detect_word_repetition", "onomatopoeia_gestures",
    "DIRECTION_WORDS", "SPEED_WORDS", "TEXTURE_WORDS", "DYNAMIC_WORDS",
    "FigureType", "FigureInstance",
    "SIMILE_MARKERS", "PERSONIFICATION_VERBS", "SYNESTHESIA_PAIRS", "HYPERBOLE_MARKERS",
    "detect_simile", "detect_personification", "detect_hyperbole",
    "domain_membership", "detect_cross_domain",
    "analyze_lyrics", "figurative_density", "figure_type_distribution",
    "na_NarrativePhase", "na_SectionType", "na_LyricSection", "na_NarrativeArc",
    "INTENSITY_WORDS",
    "na_section_intensity", "na_detect_repetition_ratio",
    "assign_narrative_phases", "classify_arc_type",
    "build_narrative_arc", "thematic_coherence",
    "StressLevel", "SyllableInfo", "StressPattern",
    "estimate_syllable_count", "lookup_stress", "line_stress_pattern",
    "metric_alignment_score", "generate_metric_grid",
    "stress_mismatch_positions", "prosodic_naturalness",
]

from models.music_linguistics.sentiment_lexicon import (
    MUSIC_SENTIMENT_LEXICON, SentimentScore, score_text,
    sentiment_trajectory, valence_arc, arousal_arc,
    emotional_volatility, dominant_emotion,
)
from models.music_linguistics.text_music_alignment import (
    AlignmentType, SyllableNote, AlignmentScore,
    syllable_count, simple_stress_pattern, stress_match_score,
    melismatic_ratio, detect_melismas, phrase_boundary_alignment,
    compute_alignment_score, word_painting_candidates,
)
from models.music_linguistics.discourse_analysis import (
    NarrativePhase, NarrativeArc, SemanticShift, ThematicProfile,
    estimate_narrative_arc, semantic_shift_between, pairwise_semantic_shifts,
    thematic_profile, repetition_density, lexical_density, imagery_score,
)
from models.music_linguistics.rhyme_quality import (
    RhymeQuality, RhymeScore, SchemeAnalysis,
    score_rhyme, analyze_rhyme_scheme, internal_rhyme_density,
    multisyllabic_rhyme_score,
)
from models.music_linguistics.translation_scoring import (
    AdaptationDimension, AdaptationScore, SyllableMap, VowelProfile,
    syllable_map, syllable_fidelity, vowel_openness_similarity,
    stress_alignment_score, rhyme_preservation_score, semantic_fidelity_score,
    score_adaptation, back_translation_consistency, note_density_ratio,
)
from models.music_linguistics.prosody_analysis import (
    SyllableStress, ProsodicFit,
    extract_stress_pattern, stress_beat_alignment,
    melismatic_classification, vowel_openness_map,
    text_rhythm_ioi, prosodic_fit_score,
)
from models.music_linguistics.word_painting import (
    PaintingGesture, PaintingSuggestion, RepetitionPainting,
    detect_painting_words, direction_to_contour, dynamic_curve_from_gestures,
    suggest_painting, painting_density,
    melodic_interval_from_direction, tempo_adjustment,
    detect_word_repetition, onomatopoeia_gestures,
    DIRECTION_WORDS, SPEED_WORDS, TEXTURE_WORDS, DYNAMIC_WORDS,
)
from models.music_linguistics.metaphor_detection import (
    FigureType, FigureInstance,
    SIMILE_MARKERS, PERSONIFICATION_VERBS, SYNESTHESIA_PAIRS, HYPERBOLE_MARKERS,
    detect_simile, detect_personification, detect_hyperbole,
    domain_membership, detect_cross_domain,
    analyze_lyrics, figurative_density, figure_type_distribution,
)
from models.music_linguistics.narrative_arc import (
    NarrativePhase as na_NarrativePhase,
    SectionType as na_SectionType,
    LyricSection as na_LyricSection,
    NarrativeArc as na_NarrativeArc,
    INTENSITY_WORDS,
    section_intensity as na_section_intensity,
    detect_repetition_ratio as na_detect_repetition_ratio,
    assign_narrative_phases, classify_arc_type,
    build_narrative_arc, thematic_coherence,
)
from models.music_linguistics.syllable_stress import (
    StressLevel, SyllableInfo, StressPattern,
    estimate_syllable_count, lookup_stress, line_stress_pattern,
    metric_alignment_score, generate_metric_grid,
    stress_mismatch_positions, prosodic_naturalness,
)
