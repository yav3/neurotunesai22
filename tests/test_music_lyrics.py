"""Tests for music_lyrics module: tokenizer, rhyme, meter, sentiment, parsers, corpus builder."""
import pytest
import json


class TestTokenizer:
    def test_word_tokenize_english(self):
        from models.music_lyrics.tokenizer import word_tokenize
        tokens = word_tokenize("Hello, world! How are you?")
        assert "Hello" in tokens
        assert "world" in tokens
        assert len(tokens) >= 4

    def test_word_tokenize_cjk(self):
        from models.music_lyrics.tokenizer import word_tokenize
        tokens = word_tokenize("你好世界")
        assert len(tokens) == 4

    def test_word_tokenize_mixed(self):
        from models.music_lyrics.tokenizer import word_tokenize
        tokens = word_tokenize("Hello 你好")
        assert "Hello" in tokens
        assert len(tokens) >= 3

    def test_syllable_english(self):
        from models.music_lyrics.tokenizer import syllable_estimate
        assert syllable_estimate("hello") == 2
        assert syllable_estimate("beautiful") == 3
        assert syllable_estimate("I") == 1
        assert syllable_estimate("") == 0

    def test_syllable_cjk(self):
        from models.music_lyrics.tokenizer import syllable_estimate
        assert syllable_estimate("你好") == 2
        assert syllable_estimate("世") == 1

    def test_detect_language_english(self):
        from models.music_lyrics.tokenizer import detect_language
        lang = detect_language("The quick brown fox jumps over the lazy dog")
        assert lang == "en"

    def test_detect_language_chinese(self):
        from models.music_lyrics.tokenizer import detect_language
        lang = detect_language("你好世界这是一个测试")
        assert lang == "zh"

    def test_detect_language_arabic(self):
        from models.music_lyrics.tokenizer import detect_language
        lang = detect_language("مرحبا بالعالم هذا اختبار")
        assert lang == "ar"

    def test_normalize_text(self):
        from models.music_lyrics.tokenizer import normalize_text
        result = normalize_text("Hello WORLD")
        assert result == "hello world"

    def test_normalize_strip_diacritics(self):
        from models.music_lyrics.tokenizer import normalize_text
        result = normalize_text("café résumé", strip_diacritics=True)
        assert "e" in result


class TestRhyme:
    def test_phonetic_suffix(self):
        from models.music_lyrics.rhyme import phonetic_suffix
        s1 = phonetic_suffix("night")
        s2 = phonetic_suffix("light")
        assert s1 == s2

    def test_detect_rhyme_scheme_aabb(self):
        from models.music_lyrics.rhyme import detect_rhyme_scheme
        text = "I saw a cat\nWearing a hat\nIt ran away\nOn a sunny day"
        scheme = detect_rhyme_scheme(text)
        assert len(scheme) == 4
        assert scheme[0] == scheme[1]
        assert scheme[2] == scheme[3]

    def test_detect_rhyme_scheme_abab(self):
        from models.music_lyrics.rhyme import detect_rhyme_scheme
        text = "Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you"
        scheme = detect_rhyme_scheme(text)
        assert len(scheme) == 4

    def test_end_rhyme_score(self):
        from models.music_lyrics.rhyme import end_rhyme_score
        rhyming = "The cat in the hat\nSat on the mat\nAnd that was that\nWith a baseball bat"
        non_rhyming = "I went to the store\nBought some bread\nCame back home\nAnd went to sleep"
        assert end_rhyme_score(rhyming) > end_rhyme_score(non_rhyming)

    def test_alliteration_score(self):
        from models.music_lyrics.rhyme import alliteration_score
        alliterative = "Peter Piper picked a peck of pickled peppers"
        normal = "The cat sat on the mat"
        assert alliteration_score(alliterative) > alliteration_score(normal)

    def test_empty_text(self):
        from models.music_lyrics.rhyme import detect_rhyme_scheme, end_rhyme_score
        assert detect_rhyme_scheme("") == ""
        assert end_rhyme_score("") == 0.0


class TestMeter:
    def test_stress_pattern(self):
        from models.music_lyrics.meter import stress_pattern
        p = stress_pattern("hello")
        assert len(p) == 2
        assert sum(p) >= 1

    def test_stress_monosyllable(self):
        from models.music_lyrics.meter import stress_pattern
        assert stress_pattern("the") == [0]
        assert stress_pattern("love") == [1]

    def test_detect_meter_iambic(self):
        from models.music_lyrics.meter import detect_meter
        iambic = "Shall I compare thee to a summer's day\nThou art more lovely and more temperate\nRough winds do shake the darling buds of May\nAnd summer's lease hath all too short a date"
        meter = detect_meter(iambic)
        assert meter in ("iambic", "trochaic", "amphibrachic", "free verse")

    def test_detect_meter_free_verse(self):
        from models.music_lyrics.meter import detect_meter
        free = "I\nAm\nHere"
        meter = detect_meter(free)
        assert isinstance(meter, str)

    def test_scansion_regularity(self):
        from models.music_lyrics.meter import scansion_regularity
        regular = "Da dum da dum da dum da dum\nDa dum da dum da dum da dum\nDa dum da dum da dum da dum\nDa dum da dum da dum da dum"
        irregular = "Short\nA very very very long line with many syllables\nMedium line here"
        assert scansion_regularity(regular) > scansion_regularity(irregular)


class TestSentiment:
    def test_word_vad_lookup(self):
        from models.music_lyrics.sentiment import word_vad_lookup
        vad = word_vad_lookup("love")
        assert vad is not None
        assert vad[0] > 0.8  # high valence

    def test_word_vad_negative(self):
        from models.music_lyrics.sentiment import word_vad_lookup
        vad = word_vad_lookup("death")
        assert vad is not None
        assert vad[0] < 0.2  # low valence

    def test_word_vad_unknown(self):
        from models.music_lyrics.sentiment import word_vad_lookup
        vad = word_vad_lookup("xyzzyplugh")
        assert vad is None

    def test_line_vad(self):
        from models.music_lyrics.sentiment import line_vad
        v, a, d = line_vad("I love the beautiful sunshine")
        assert v > 0.7

    def test_document_vad(self):
        from models.music_lyrics.sentiment import document_vad
        text = "Love and joy\nHappy beautiful day\nSmile and dance"
        v, a, d = document_vad(text)
        assert v > 0.7

    def test_document_vad_chorus_weight(self):
        from models.music_lyrics.sentiment import document_vad
        text = "[Chorus]\nSad lonely tears\nDark empty pain"
        v, a, d = document_vad(text)
        assert v < 0.4

    def test_emotion_arc(self):
        from models.music_lyrics.sentiment import emotion_arc
        text = "Love and joy\nBeautiful day\nSunshine bright\nDark night\nSad tears\nPain alone\nHope returns\nLight again"
        arc = emotion_arc(text, n_segments=4)
        assert len(arc) == 4
        assert all(len(point) == 3 for point in arc)
        assert arc[0][0] > arc[2][0]  # positive start > negative middle

    def test_emotion_arc_empty(self):
        from models.music_lyrics.sentiment import emotion_arc
        arc = emotion_arc("", n_segments=4)
        assert len(arc) == 4


class TestParsers:
    def test_parse_lrc(self):
        from models.music_lyrics.parsers import parse_lrc_file
        lrc = "[ti:Test Song]\n[ar:Test Artist]\n[00:05.00]First line\n[00:10.50]Second line\n[00:15.00]Third line"
        rec = parse_lrc_file(lrc)
        assert rec.title == "Test Song"
        assert rec.artist == "Test Artist"
        assert "First line" in rec.text
        assert len(rec.timestamps) == 3
        assert rec.timestamps[0][0] == 5.0

    def test_parse_srt(self):
        from models.music_lyrics.parsers import parse_srt_subtitle
        srt = "1\n00:00:05,000 --> 00:00:10,000\nFirst line\n\n2\n00:00:10,000 --> 00:00:15,000\nSecond line"
        rec = parse_srt_subtitle(srt, title="Test")
        assert rec.title == "Test"
        assert "First line" in rec.text
        assert len(rec.timestamps) == 2

    def test_parse_plain_lyrics(self):
        from models.music_lyrics.parsers import parse_plain_lyrics
        text = "[Verse]\nFirst verse line\nSecond verse line\n\n[Chorus]\nChorus line one\nChorus line two"
        rec = parse_plain_lyrics(text, title="Song", artist="Artist")
        assert rec.title == "Song"
        assert len(rec.sections) >= 2
        assert rec.sections[0]["type"] == "verse"
        assert rec.sections[1]["type"] == "chorus"

    def test_parse_gutenberg(self):
        from models.music_lyrics.parsers import parse_gutenberg_poem
        text = "*** START OF THE PROJECT GUTENBERG EBOOK ***\nSonnet\nby Shakespeare\n\nShall I compare thee\nTo a summer's day\n\n*** END OF THE PROJECT GUTENBERG EBOOK ***"
        rec = parse_gutenberg_poem(text)
        assert rec.source == "gutenberg"
        assert rec.licence == "public domain"
        assert "compare" in rec.text


class TestCorpusBuilder:
    def test_add_lyric(self):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        from models.music_lyrics.schema import LyricRecord
        cb = CorpusBuilder()
        rec = LyricRecord(text="This is a test song with some lyrics and words", title="Test")
        assert cb.add_lyric(rec, enrich=False)
        assert cb.total == 1

    def test_dedup(self):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        from models.music_lyrics.schema import LyricRecord
        cb = CorpusBuilder()
        rec1 = LyricRecord(text="This is a test song with many words and lines", title="Test1")
        rec2 = LyricRecord(text="This is a test song with many words and lines", title="Test2")
        cb.add_lyric(rec1, enrich=False)
        assert not cb.add_lyric(rec2, enrich=False)
        assert cb.total == 1

    def test_add_poem(self):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        from models.music_lyrics.schema import PoemRecord
        cb = CorpusBuilder()
        rec = PoemRecord(text="Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you", title="Classic")
        assert cb.add_poem(rec, enrich=False)
        assert cb.total == 1

    def test_statistics(self):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        from models.music_lyrics.schema import LyricRecord
        cb = CorpusBuilder()
        cb.add_lyric(LyricRecord(text="English song lyrics here for testing", language="en", source="test"), enrich=False)
        cb.add_lyric(LyricRecord(text="Chanson francaise avec des paroles", language="fr", source="test"), enrich=False)
        stats = cb.statistics()
        assert stats.total_records == 2
        assert stats.languages_count == 2
        assert "en" in stats.by_language

    def test_filter_by_language(self):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        from models.music_lyrics.schema import LyricRecord
        cb = CorpusBuilder()
        cb.add_lyric(LyricRecord(text="English song lyrics here for testing", language="en"), enrich=False)
        cb.add_lyric(LyricRecord(text="Chanson francaise avec des mots et paroles", language="fr"), enrich=False)
        en_lyrics, en_poems = cb.filter_by_language("en")
        assert len(en_lyrics) == 1

    def test_add_from_records(self):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        cb = CorpusBuilder()
        records = [
            {"text": "First song with enough text to pass filter", "title": "Song1", "artist": "A"},
            {"text": "Second song with enough text to pass filter", "title": "Song2", "artist": "B"},
        ]
        added = cb.add_from_records(records, record_type="lyric")
        assert added == 2

    def test_export_jsonl(self, tmp_path):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        from models.music_lyrics.schema import LyricRecord
        cb = CorpusBuilder()
        cb.add_lyric(LyricRecord(text="Test song with enough words for export", title="T"), enrich=False)
        path = tmp_path / "corpus.jsonl"
        count = cb.export_jsonl(path)
        assert count == 1
        with open(path) as f:
            data = json.loads(f.readline())
        assert data["type"] == "lyric"
        assert data["title"] == "T"

    def test_reject_short(self):
        from models.music_lyrics.corpus_builder import CorpusBuilder
        from models.music_lyrics.schema import LyricRecord
        cb = CorpusBuilder()
        assert not cb.add_lyric(LyricRecord(text="hi"), enrich=False)
        assert cb.total == 0
