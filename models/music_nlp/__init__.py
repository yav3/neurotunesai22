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
    "SectionRole", "RhetoricalRelation", "DiscourseUnit", "RhetoricalEdge", "DiscourseTree",
    "classify_sentence", "segment_document", "build_discourse_tree",
    "argument_chain", "section_proportions", "discourse_feature_vector",
    "LexiconEntry", "MUSIC_LEXICON",
    "sentiment_tokenize", "score_token", "score_text",
    "sentiment_trajectory", "lexicon_coverage",
    "StyleDimension", "StyleProfile", "GENRE_PROFILES",
    "FORMAL_SUBSTITUTIONS", "INFORMAL_SUBSTITUTIONS",
    "measure_formality", "measure_emotionality", "measure_concreteness",
    "measure_complexity", "measure_repetitiveness",
    "compute_style_profile", "style_distance", "closest_genre", "formality_shift",
    "as_ScoredLine", "SongSummary", "textrank_sentences",
    "position_weighted_score", "extract_themes",
    "summarize_song", "chorus_detection", "information_density",
    "ke_Keyword", "MUSIC_VOCABULARY", "ke_STOPWORDS",
    "ke_tokenize", "ke_term_frequency", "ke_idf_scores",
    "ke_tfidf_keywords", "ke_textrank_keywords",
    "ke_extract_ngrams", "ke_music_term_density",
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
from models.music_nlp.discourse_structure import (
    SectionRole, RhetoricalRelation, DiscourseUnit, RhetoricalEdge, DiscourseTree,
    classify_sentence, segment_document, build_discourse_tree,
    argument_chain, section_proportions, discourse_feature_vector,
)
from models.music_nlp.sentiment_lexicon import (
    LexiconEntry, MUSIC_LEXICON,
    tokenize as sentiment_tokenize, score_token, score_text,
    sentiment_trajectory, lexicon_coverage,
)
from models.music_nlp.style_transfer import (
    StyleDimension, StyleProfile, GENRE_PROFILES,
    FORMAL_SUBSTITUTIONS, INFORMAL_SUBSTITUTIONS,
    measure_formality, measure_emotionality, measure_concreteness,
    measure_complexity, measure_repetitiveness,
    compute_style_profile, style_distance, closest_genre, formality_shift,
)
from models.music_nlp.abstractive_summary import (
    ScoredLine as as_ScoredLine,
    SongSummary, textrank_sentences,
    position_weighted_score, extract_themes,
    summarize_song, chorus_detection, information_density,
)
from models.music_nlp.keyword_extraction import (
    Keyword as ke_Keyword,
    MUSIC_VOCABULARY, STOPWORDS as ke_STOPWORDS,
    tokenize as ke_tokenize, term_frequency as ke_term_frequency,
    idf_scores as ke_idf_scores,
    tfidf_keywords as ke_tfidf_keywords, textrank_keywords as ke_textrank_keywords,
    extract_ngrams as ke_extract_ngrams, music_term_density as ke_music_term_density,
)
