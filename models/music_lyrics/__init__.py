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

__all__ = [
    "LyricRecord", "PoemRecord", "CorpusStats",
    "word_tokenize", "syllable_estimate", "detect_language", "normalize_text",
    "phonetic_suffix", "detect_rhyme_scheme", "end_rhyme_score", "alliteration_score",
    "stress_pattern", "detect_meter", "scansion_regularity",
    "word_vad_lookup", "line_vad", "document_vad", "emotion_arc",
    "parse_gutenberg_poem", "parse_lrc_file", "parse_srt_subtitle", "parse_plain_lyrics",
    "CorpusBuilder",
    "CorpusSource", "OPEN_CORPUS_SOURCES", "METER_TYPES",
    "ProsodyAnalysis", "MultilingualTokenizer", "MultilingualPoetryAnalyzer",
]
