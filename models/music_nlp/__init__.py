from __future__ import annotations

from models.music_nlp.lyrics_features import LyricsAnalyzer, LyricsFeatures
from models.music_nlp.lyrics_emotion import LyricsEmotionProfile, LyricsEmotionAnalyzer
from models.music_nlp.multilingual import MultilingualProcessor
from models.music_nlp.prosody import ProsodyFeatures, ProsodyAnalyzer
from models.music_nlp.scraper_registry import DataSource

__all__ = [
    "LyricsAnalyzer", "LyricsFeatures",
    "LyricsEmotionProfile", "LyricsEmotionAnalyzer",
    "MultilingualProcessor",
    "ProsodyFeatures", "ProsodyAnalyzer",
    "DataSource",
    "Syllable", "PhoneticWord", "NoteSlot", "AlignmentResult",
    "count_syllables", "syllabify",
    "align_syllables_to_notes", "stress_alignment_score",
    "vowel_on_high_note", "lyric_density", "phrasing_breath_points",
    "MarkovOrder", "LyricTemplate", "MarkovChain",
    "build_markov_chain", "generate_line", "syllable_count_estimate",
    "fits_template", "rhyme_score", "generate_verse", "perplexity",
    "vocabulary_richness",
    "EntityType", "MusicEntity", "NERResult",
    "GENRE_GAZETTEER", "INSTRUMENT_GAZETTEER", "TEMPO_MARKINGS",
    "gazetteer_match", "extract_quoted_titles", "extract_key_signatures",
    "extract_tempo_markings", "capitalized_phrase_candidates",
    "classify_capitalized_phrase", "extract_entities",
    "entity_cooccurrence_matrix", "entity_frequency",
    "Vocabulary", "TopicModelResult",
    "tokenize_music_text", "build_corpus", "tfidf_matrix",
    "lda_gibbs", "nmf_topics", "top_words_per_topic",
    "topic_coherence", "compute_perplexity",
    "fit_lda", "fit_nmf",
    "AspectSentiment", "ReviewAnalysis",
    "analyze_review", "rating_estimate", "aspect_radar",
    "review_feature_vector", "lexicon_density",
]

from models.music_nlp.phonetic_alignment import (
    Syllable, PhoneticWord, NoteSlot, AlignmentResult,
    count_syllables, syllabify,
    align_syllables_to_notes, stress_alignment_score,
    vowel_on_high_note, lyric_density, phrasing_breath_points,
)
from models.music_nlp.lyric_generation import (
    MarkovOrder, LyricTemplate, MarkovChain,
    build_markov_chain, generate_line, syllable_count_estimate,
    fits_template, rhyme_score, generate_verse, perplexity,
    vocabulary_richness,
)
from models.music_nlp.music_ner import (
    EntityType, MusicEntity, NERResult,
    GENRE_GAZETTEER, INSTRUMENT_GAZETTEER, TEMPO_MARKINGS,
    gazetteer_match, extract_quoted_titles, extract_key_signatures,
    extract_tempo_markings, capitalized_phrase_candidates,
    classify_capitalized_phrase, extract_entities,
    entity_cooccurrence_matrix, entity_frequency,
)
from models.music_nlp.topic_modeling import (
    Vocabulary, TopicModelResult,
    tokenize_music_text, build_corpus, tfidf_matrix,
    lda_gibbs, nmf_topics, top_words_per_topic,
    topic_coherence, compute_perplexity,
    fit_lda, fit_nmf,
)
from models.music_nlp.review_analysis import (
    AspectSentiment, ReviewAnalysis,
    analyze_review, rating_estimate, aspect_radar,
    review_feature_vector, lexicon_density,
)
