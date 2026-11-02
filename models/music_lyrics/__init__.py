"""Multilingual lyrics and poetry corpus: tokenization, rhyme, meter, sentiment, parsing."""
from models.music_lyrics.schema import LyricRecord, PoemRecord, CorpusStats
from models.music_lyrics.tokenizer import (
    word_tokenize,
    syllable_estimate,
    detect_language,
    normalize_text,
)
from models.music_lyrics.rhyme import (
    phonetic_suffix,
    detect_rhyme_scheme,
    end_rhyme_score,
    alliteration_score,
)
from models.music_lyrics.meter import (
    stress_pattern,
    detect_meter,
    scansion_regularity,
)
from models.music_lyrics.sentiment import (
    word_vad_lookup,
    line_vad,
    document_vad,
    emotion_arc,
)
from models.music_lyrics.parsers import (
    parse_gutenberg_poem,
    parse_lrc_file,
    parse_srt_subtitle,
    parse_plain_lyrics,
)
from models.music_lyrics.corpus_builder import CorpusBuilder
from models.music_lyrics.multilingual import (
    CorpusSource,
    OPEN_CORPUS_SOURCES,
    METER_TYPES,
    ProsodyAnalysis,
    MultilingualTokenizer,
    MultilingualPoetryAnalyzer,
)
from models.music_lyrics.poetry_analysis import (
    MeterType as MeterTypePA, StanzaForm, RhymeAnalysis, MeterAnalysis,
    PoetryAnalysis, estimate_syllables, syllable_count_line,
    end_sound, detect_rhyme_scheme as detect_rhyme_scheme_pa,
    estimate_stress, detect_meter as detect_meter_pa,
    classify_stanza_form, vocabulary_richness, analyze_poetry,
)
from models.music_lyrics.lyrics_melody_alignment import (
    SyllableInfo, NoteInfo, AlignmentPair, AlignmentResult,
    extract_syllables, extract_notes,
    align_lyrics_to_melody, prosodic_fitness,
    phrase_boundary_score, pitch_contour_text_affect,
)
from models.music_lyrics.rhyme_generation import (
    RhymeType, RhymeCandidate, RhymeGraph,
    rhyme_similarity, multi_syllable_rhyme_score,
    build_rhyme_graph, find_rhyme_candidates, rhyme_chain,
    get_scheme, generate_rhyme_assignments, scheme_complexity, suggest_scheme,
)

from models.music_lyrics.word_painting import (
    SemanticCategory, PaintingSuggestion,
    classify_word, suggest_painting, paint_line,
    aggregate_dynamics, aggregate_tempo,
    pitch_contour_from_text, word_painting_density, semantic_arc,
)

__all__ = [
    "SemanticCategory", "PaintingSuggestion",
    "classify_word", "suggest_painting", "paint_line",
    "aggregate_dynamics", "aggregate_tempo",
    "pitch_contour_from_text", "word_painting_density", "semantic_arc",
    "LyricRecord", "PoemRecord", "CorpusStats",
    "word_tokenize", "syllable_estimate", "detect_language", "normalize_text",
    "phonetic_suffix", "detect_rhyme_scheme", "end_rhyme_score", "alliteration_score",
    "stress_pattern", "detect_meter", "scansion_regularity",
    "word_vad_lookup", "line_vad", "document_vad", "emotion_arc",
    "parse_gutenberg_poem", "parse_lrc_file", "parse_srt_subtitle", "parse_plain_lyrics",
    "CorpusBuilder",
    "CorpusSource", "OPEN_CORPUS_SOURCES", "METER_TYPES",
    "ProsodyAnalysis", "MultilingualTokenizer", "MultilingualPoetryAnalyzer",
    "MeterTypePA", "StanzaForm", "RhymeAnalysis", "MeterAnalysis",
    "PoetryAnalysis", "estimate_syllables", "syllable_count_line",
    "end_sound", "detect_rhyme_scheme_pa",
    "estimate_stress", "detect_meter_pa",
    "classify_stanza_form", "vocabulary_richness", "analyze_poetry",
    "SyllableInfo", "NoteInfo", "AlignmentPair", "AlignmentResult",
    "extract_syllables", "extract_notes",
    "align_lyrics_to_melody", "prosodic_fitness",
    "phrase_boundary_score", "pitch_contour_text_affect",
    "RhymeType", "RhymeCandidate", "RhymeGraph",
    "rhyme_similarity", "multi_syllable_rhyme_score",
    "build_rhyme_graph", "find_rhyme_candidates", "rhyme_chain",
    "get_scheme", "generate_rhyme_assignments", "scheme_complexity", "suggest_scheme",
    "Syllable", "ProsodyAlignment", "MelodyContour",
    "syllabify", "align_syllables_to_beats", "prosody_score",
    "contour_similarity", "melisma_ratio", "text_setting_quality",
    "optimal_text_underlay",
]

from models.music_lyrics.prosody_alignment import (
    Syllable, ProsodyAlignment, MelodyContour,
    syllabify, stress_pattern as prosody_stress_pattern,
    align_syllables_to_beats, prosody_score,
    contour_similarity, melisma_ratio, text_setting_quality,
    optimal_text_underlay,
)
from models.music_lyrics.prosody import (
    Syllable as ProsodySyllable, MetricPosition, ProsodyAlignment as ProsodyAlignmentDetailed,
    syllabify as prosody_syllabify, estimate_stress as prosody_estimate_stress,
    metric_strength_profile, align_syllables_to_beats as prosody_align,
    prosody_score as prosody_fitness_score,
    mismatch_count, mismatch_ratio, syllable_density,
    melisma_ratio as prosody_melisma_ratio,
    classify_text_setting, stress_contour, contour_correlation,
)
from models.music_lyrics.lyric_generation import (
    SyllableTemplate, COMMON_TEMPLATES, RhymeScheme, RHYME_SCHEMES,
    PhonemeCluster, GeneratedLine, GeneratedVerse, SongStructure,
    COMMON_STRUCTURES, count_syllables_approx, line_syllable_count,
    stress_from_syllable_count, enforce_syllable_count,
    end_phoneme, words_rhyme, pick_rhyming_word, rhyme_distance,
    vowel_similarity, MarkovLyricModel, generate_verse,
    song_section_count, total_lines, lyric_density, repetition_ratio,
)
