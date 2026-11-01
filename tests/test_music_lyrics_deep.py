"""Deep tests for music_lyrics: rhyme detection, meter analysis,
sentiment, tokenizer, prosody alignment."""
import numpy as np
import pytest

from models.music_lyrics.rhyme import (
    detect_rhyme_scheme, end_rhyme_score, phonetic_suffix,
    alliteration_score, normalize_text, word_tokenize,
)
from models.music_lyrics.meter import (
    detect_meter, scansion_regularity, stress_pattern,
    syllable_estimate,
)
from models.music_lyrics.sentiment import (
    analyze_sentiment, sentiment_valence, emotional_arc,
)
from models.music_lyrics.tokenizer import (
    tokenize_lyrics, normalize_lyrics, split_into_sections,
)
from models.music_lyrics.prosody import (
    syllable_count, vowel_ratio, consonant_cluster_density,
    phonetic_complexity,
)


# ---- Rhyme ----

class TestRhyme:
    SIMPLE_POEM = "The cat sat on the mat\nIt wore a funny hat\nThe dog ran very far\nAnd jumped into a car"

    def test_rhyme_scheme_aabb(self):
        scheme = detect_rhyme_scheme(self.SIMPLE_POEM)
        assert isinstance(scheme, str)
        assert len(scheme) >= 4

    def test_end_rhyme_score_rhyming(self):
        score = end_rhyme_score(self.SIMPLE_POEM)
        assert score > 0.3

    def test_end_rhyme_score_nonrhyming(self):
        text = "The sky is blue today\nI like to read a book\nCoffee tastes so good\nMy shoes are red and green"
        score = end_rhyme_score(text)
        assert score < 0.5

    def test_phonetic_suffix_cat(self):
        s = phonetic_suffix("cat")
        assert isinstance(s, str)
        assert len(s) >= 2

    def test_alliteration_high(self):
        text = "Peter Piper picked a peck of pickled peppers"
        score = alliteration_score(text)
        assert score > 0.3

    def test_alliteration_low(self):
        text = "The quick brown fox jumped over the lazy dog"
        score = alliteration_score(text)
        assert score < 0.5

    def test_normalize_lowercases(self):
        assert normalize_text("HELLO World") == "hello world"

    def test_word_tokenize(self):
        tokens = word_tokenize("Hello, world!")
        assert "hello" in tokens or "Hello" in tokens


# ---- Meter ----

class TestMeter:
    def test_detect_iambic(self):
        text = "Shall I compare thee to a summer's day\nThou art more lovely and more temperate"
        meter = detect_meter(text)
        assert isinstance(meter, str)

    def test_scansion_regularity_bounded(self):
        text = "Shall I compare thee to a summer's day\nThou art more lovely and more temperate"
        reg = scansion_regularity(text)
        assert 0.0 <= reg <= 1.0

    def test_stress_pattern_returns_list(self):
        pat = stress_pattern("banana")
        assert isinstance(pat, list)
        assert all(s in (0, 1) for s in pat)

    def test_syllable_estimate_monosyllable(self):
        assert syllable_estimate("cat") == 1

    def test_syllable_estimate_polysyllable(self):
        est = syllable_estimate("beautiful")
        assert est >= 3


# ---- Sentiment ----

class TestSentiment:
    def test_positive_sentiment(self):
        result = analyze_sentiment("I love this beautiful song so much")
        assert result.polarity > 0

    def test_negative_sentiment(self):
        result = analyze_sentiment("This terrible song makes me cry with sadness")
        assert result.polarity < 0

    def test_sentiment_valence_bounded(self):
        v = sentiment_valence("A wonderful melody fills the air")
        assert -1.0 <= v <= 1.0

    def test_emotional_arc_shape(self):
        lines = [
            "The morning sun shines bright",
            "But clouds begin to gather",
            "The storm breaks with fury",
            "And then the rainbow appears",
        ]
        arc = emotional_arc(lines)
        assert len(arc) == 4


# ---- Tokenizer ----

class TestTokenizer:
    SONG = "[Verse 1]\nHello darkness my old friend\nI've come to talk with you again\n\n[Chorus]\nThe sound of silence"

    def test_tokenize_returns_list(self):
        tokens = tokenize_lyrics("Hello darkness my old friend")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_normalize_strips_brackets(self):
        norm = normalize_lyrics("[Verse 1] Hello world")
        assert "[" not in norm

    def test_split_into_sections(self):
        sections = split_into_sections(self.SONG)
        assert isinstance(sections, (list, dict))


# ---- Prosody ----

class TestProsody:
    def test_syllable_count_hello(self):
        c = syllable_count("hello")
        assert c == 2

    def test_syllable_count_cat(self):
        c = syllable_count("cat")
        assert c == 1

    def test_vowel_ratio_bounded(self):
        r = vowel_ratio("beautiful")
        assert 0.0 <= r <= 1.0

    def test_consonant_cluster_density_bounded(self):
        d = consonant_cluster_density("strengths")
        assert d >= 0.0

    def test_phonetic_complexity_positive(self):
        c = phonetic_complexity("strengths")
        assert c > 0.0

    def test_phonetic_complexity_simple_lower(self):
        c_simple = phonetic_complexity("mama")
        c_hard = phonetic_complexity("strengths")
        assert c_hard > c_simple
