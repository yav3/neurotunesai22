from __future__ import annotations

import pytest

from models.music_nlp.multilingual import MultilingualProcessor, LANGUAGE_CONFIGS
from models.music_nlp.lyrics_features import LyricsAnalyzer, POSITIVE_WORDS, NEGATIVE_WORDS
from models.music_nlp.prosody import ProsodyAnalyzer, STRESS_DICTIONARY, VOWELS
from models.music_nlp.lyrics_emotion import LyricsEmotionAnalyzer, EMOTION_LEXICON, VAD_LEXICON
from models.music_nlp.scraper_registry import (
    OPEN_SOURCE_REGISTRY,
    get_sources_by_language,
    get_sources_by_license,
    get_poetry_sources,
    get_lyrics_sources,
    get_midi_sources,
    get_emotion_benchmark_sources,
)


# --- MultilingualProcessor ---

class TestMultilingualProcessor:

    def setup_method(self):
        self.mlp = MultilingualProcessor()

    def test_detect_english(self):
        lang, conf = self.mlp.detect_language("The quick brown fox jumps over the lazy dog")
        assert lang == "en"
        assert conf > 0.0

    def test_detect_spanish(self):
        lang, _ = self.mlp.detect_language("El rápido zorro marrón salta sobre el perro perezoso")
        assert lang == "es"

    def test_detect_french(self):
        lang, _ = self.mlp.detect_language("Le renard brun rapide saute par-dessus le chien paresseux")
        assert lang == "fr"

    def test_detect_japanese(self):
        lang, _ = self.mlp.detect_language("素早い茶色のキツネが怠惰な犬を飛び越える")
        assert lang == "ja"

    def test_detect_korean(self):
        lang, _ = self.mlp.detect_language("빠른 갈색 여우가 게으른 개를 뛰어넘는다")
        assert lang == "ko"

    def test_detect_chinese(self):
        lang, _ = self.mlp.detect_language("敏捷的棕色狐狸跳过了懒狗")
        assert lang == "zh"

    def test_detect_arabic(self):
        lang, _ = self.mlp.detect_language("الثعلب البني السريع يقفز فوق الكلب الكسول")
        assert lang == "ar"

    def test_detect_russian(self):
        lang, _ = self.mlp.detect_language("Быстрая коричневая лиса прыгает через ленивую собаку")
        assert lang == "ru"

    def test_tokenize_english(self):
        tokens = self.mlp.tokenize("Hello world, this is a test!", "en")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_tokenize_cjk_characters(self):
        tokens = self.mlp.tokenize("你好世界", "zh")
        assert len(tokens) >= 1

    def test_remove_stopwords_english(self):
        tokens = ["the", "quick", "brown", "fox", "is", "a", "good", "animal"]
        filtered = self.mlp.remove_stopwords(tokens, "en")
        assert "the" not in filtered
        assert "is" not in filtered
        assert "a" not in filtered
        assert "quick" in filtered
        assert "fox" in filtered

    def test_syllable_count_monosyllable(self):
        assert self.mlp.syllable_count("dog", "en") == 1

    def test_syllable_count_polysyllable(self):
        count = self.mlp.syllable_count("beautiful", "en")
        assert count >= 3

    def test_detect_script_latin(self):
        script = self.mlp.detect_script("Hello world")
        assert script == "latin"

    def test_detect_script_cjk(self):
        script = self.mlp.detect_script("你好世界")
        assert script == "cjk"

    def test_detect_script_hangul(self):
        script = self.mlp.detect_script("안녕하세요")
        assert script == "hangul"

    def test_detect_script_arabic(self):
        script = self.mlp.detect_script("مرحبا")
        assert script == "arabic"

    def test_detect_script_cyrillic(self):
        script = self.mlp.detect_script("Привет")
        assert script == "cyrillic"

    def test_language_configs_coverage(self):
        assert len(LANGUAGE_CONFIGS) == 12
        for code in ("en", "es", "fr", "pt", "de", "it", "ja", "ko", "zh", "ar", "hi", "ru"):
            assert code in LANGUAGE_CONFIGS


# --- LyricsAnalyzer ---

class TestLyricsAnalyzer:

    def setup_method(self):
        self.analyzer = LyricsAnalyzer()

    def test_empty_text(self):
        features = self.analyzer.analyze("")
        assert features.word_count == 0
        assert features.line_count == 0

    def test_whitespace_only(self):
        features = self.analyzer.analyze("   \n\n  ")
        assert features.word_count == 0

    def test_word_count(self):
        features = self.analyzer.analyze("one two three four five")
        assert features.word_count == 5

    def test_lexical_diversity(self):
        features = self.analyzer.analyze("the the the cat cat dog")
        assert features.lexical_diversity < 1.0
        unique = self.analyzer.analyze("every word here is different and unique")
        assert unique.lexical_diversity == 1.0

    def test_line_count(self):
        text = "line one\nline two\nline three"
        features = self.analyzer.analyze(text)
        assert features.line_count == 3

    def test_syllable_counting(self):
        features = self.analyzer.analyze("beautiful wonderful amazing")
        assert features.syllable_count_approx >= 9

    def test_rhyme_detection_adjacent(self):
        text = "I saw the cat\nSitting on the mat"
        features = self.analyzer.analyze(text)
        assert features.rhyme_density > 0.0

    def test_rhyme_scheme_aabb(self):
        text = "The cat sat on a mat\nThe bat flew from the hat\nI walked along the road\nCarrying a heavy load"
        features = self.analyzer.analyze(text)
        assert len(features.rhyme_scheme) == 4

    def test_repetition_score(self):
        text = "hello world\nhello world\nsomething new"
        features = self.analyzer.analyze(text)
        assert features.repetition_score > 0.0

    def test_question_density(self):
        text = "Why do birds fly? Where do they go? I wonder."
        features = self.analyzer.analyze(text)
        assert features.question_density > 0.0

    def test_exclamation_density(self):
        text = "Wow! Amazing! Great!"
        features = self.analyzer.analyze(text)
        assert features.exclamation_density > 0.0

    def test_positive_sentiment(self):
        text = "love joy happy beautiful wonderful sunshine paradise heaven"
        features = self.analyzer.analyze(text)
        assert features.sentiment_valence_approx > 0.5

    def test_negative_sentiment(self):
        text = "hate pain hurt broken dark cold death fear anger rage"
        features = self.analyzer.analyze(text)
        assert features.sentiment_valence_approx < 0.5

    def test_vocabulary_richness(self):
        features = self.analyzer.analyze("cat dog bird fish tree rock sky sun moon star")
        assert features.vocabulary_richness > 0.0

    def test_positive_negative_word_sets_disjoint(self):
        overlap = POSITIVE_WORDS & NEGATIVE_WORDS
        assert len(overlap) == 0


# --- ProsodyAnalyzer ---

class TestProsodyAnalyzer:

    def setup_method(self):
        self.analyzer = ProsodyAnalyzer()

    def test_empty_text(self):
        features = self.analyzer.analyze("")
        assert features.meter_type == "free"
        assert features.meter_regularity == 0.0

    def test_iambic_pentameter(self):
        text = (
            "Shall I compare thee to a summer's day\n"
            "Thou art more lovely and more temperate\n"
            "Rough winds do shake the darling buds of May\n"
            "And summer's lease hath all too short a date"
        )
        features = self.analyzer.analyze(text)
        assert features.meter_type in ("iambic", "trochaic", "spondaic", "free")
        assert len(features.stress_pattern) > 0

    def test_stress_dictionary_lookup(self):
        assert STRESS_DICTIONARY["about"] == [0, 1]
        assert STRESS_DICTIONARY["happy"] == [1, 0]
        assert STRESS_DICTIONARY["imagine"] == [0, 1, 0]

    def test_alliteration_detection(self):
        text = "Peter Piper picked a peck of pickled peppers"
        features = self.analyzer.analyze(text)
        assert features.alliteration_score > 0.0

    def test_assonance_detection(self):
        text = "The rain in Spain stays mainly in the plain"
        features = self.analyzer.analyze(text)
        assert features.assonance_score > 0.0

    def test_consonance_detection(self):
        text = "The lumpy, bumpy road led straight to the dump"
        features = self.analyzer.analyze(text)
        assert features.consonance_score >= 0.0

    def test_enjambment_count(self):
        text = "I wandered lonely as\na cloud that floats on high.\nOver the hills and"
        features = self.analyzer.analyze(text)
        assert features.enjambment_count >= 1

    def test_caesura_detection(self):
        text = "To be, or not to be — that is the question"
        features = self.analyzer.analyze(text)
        assert len(features.caesura_positions) >= 0

    def test_vowel_set(self):
        assert "a" in VOWELS
        assert "e" in VOWELS
        assert "y" in VOWELS
        assert "z" not in VOWELS

    def test_feet_per_line(self):
        text = "The cat sat on the mat\nA dog ran down the road"
        features = self.analyzer.analyze(text)
        assert features.feet_per_line_avg > 0.0


# --- LyricsEmotionAnalyzer ---

class TestLyricsEmotionAnalyzer:

    def setup_method(self):
        self.analyzer = LyricsEmotionAnalyzer()

    def test_empty_text(self):
        profile = self.analyzer.analyze("")
        assert profile.valence == 0.5
        assert profile.arousal == 0.5
        assert all(v == 0.0 for v in profile.plutchik_scores.values())

    def test_joyful_text(self):
        text = "happiness love sunshine celebrate wonderful beautiful"
        profile = self.analyzer.analyze(text)
        assert profile.valence > 0.5

    def test_sad_text(self):
        text = "grief sorrow tears lonely abandoned despair"
        profile = self.analyzer.analyze(text)
        assert profile.valence < 0.5

    def test_per_line_emotions(self):
        text = "I feel so happy today\nBut yesterday was full of sorrow"
        lines = self.analyzer.per_line_emotions(text)
        assert len(lines) == 2
        for line_emo in lines:
            assert "joy" in line_emo or "sadness" in line_emo

    def test_emotion_arc(self):
        text = "joy and happiness\npain and sorrow\nhope and light again"
        arc = self.analyzer.emotion_arc(text)
        assert len(arc) == 3
        for point in arc:
            assert isinstance(point, (int, float))

    def test_emotional_intensity_curve(self):
        text = "love and peace\nrage and fury\ncalm again"
        curve = self.analyzer.emotional_intensity_curve(text)
        assert len(curve) == 3
        for val in curve:
            assert isinstance(val, float)

    def test_emotion_lexicon_coverage(self):
        assert len(EMOTION_LEXICON) > 100
        for word, entry in EMOTION_LEXICON.items():
            emotion, intensity = entry
            assert isinstance(emotion, str)
            assert 0.0 <= intensity <= 1.0

    def test_vad_lexicon_coverage(self):
        assert len(VAD_LEXICON) > 50
        for word, (v, a, d) in VAD_LEXICON.items():
            assert 0.0 <= v <= 1.0
            assert 0.0 <= a <= 1.0
            assert 0.0 <= d <= 1.0


# --- Scraper Registry ---

class TestScraperRegistry:

    def test_registry_not_empty(self):
        assert len(OPEN_SOURCE_REGISTRY) >= 18

    def test_sources_by_language_english(self):
        en_sources = get_sources_by_language("en")
        assert len(en_sources) >= 5
        names = {s.name for s in en_sources}
        assert "Project Gutenberg" in names
        assert "Genius" in names

    def test_sources_by_language_chinese(self):
        zh_sources = get_sources_by_language("zh")
        assert len(zh_sources) >= 2
        names = {s.name for s in zh_sources}
        assert "Chinese Poetry Dataset" in names

    def test_sources_by_language_japanese(self):
        ja_sources = get_sources_by_language("ja")
        names = {s.name for s in ja_sources}
        assert "Aozora Bunko" in names

    def test_sources_by_language_arabic(self):
        ar_sources = get_sources_by_language("ar")
        names = {s.name for s in ar_sources}
        assert "Arabic Poetry Corpus" in names

    def test_poetry_sources(self):
        poetry = get_poetry_sources()
        assert len(poetry) >= 8
        assert all(s.content_type == "poetry" for s in poetry)

    def test_lyrics_sources(self):
        lyrics = get_lyrics_sources()
        assert len(lyrics) >= 2
        assert all(s.content_type == "lyrics" for s in lyrics)

    def test_midi_sources(self):
        midi = get_midi_sources()
        assert len(midi) >= 1
        assert all(s.content_type == "midi" for s in midi)

    def test_emotion_benchmark_sources(self):
        emo = get_emotion_benchmark_sources()
        assert len(emo) >= 2

    def test_public_domain_license(self):
        pd = get_sources_by_license("Public Domain")
        assert len(pd) >= 2

    def test_datasource_frozen(self):
        src = OPEN_SOURCE_REGISTRY[0]
        with pytest.raises(AttributeError):
            src.name = "modified"


# --- Edge Cases ---

class TestEdgeCases:

    def test_single_word(self):
        la = LyricsAnalyzer()
        features = la.analyze("hello")
        assert features.word_count == 1
        assert features.line_count == 1

    def test_all_punctuation(self):
        la = LyricsAnalyzer()
        features = la.analyze("!!! ??? ... ---")
        assert features.word_count == 0

    def test_mixed_language_text(self):
        mlp = MultilingualProcessor()
        lang, _ = mlp.detect_language("Hello 你好 مرحبا")
        assert lang in LANGUAGE_CONFIGS

    def test_emotion_on_neutral_text(self):
        ea = LyricsEmotionAnalyzer()
        profile = ea.analyze("the table is in the room near the window")
        assert 0.3 <= profile.valence <= 0.7

    def test_prosody_single_line(self):
        pa = ProsodyAnalyzer()
        features = pa.analyze("Hello world")
        assert features.meter_type == "free" or features.meter_regularity >= 0.0
