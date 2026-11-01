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
]

from models.music_linguistics.sentiment_lexicon import (
    MUSIC_SENTIMENT_LEXICON, SentimentScore, score_text,
    sentiment_trajectory, valence_arc, arousal_arc,
    emotional_volatility, dominant_emotion,
)
