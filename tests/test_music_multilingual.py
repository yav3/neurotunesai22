"""Tests for multilingual lyrics/poetry: corpus sources, tokenizer, prosody, cross-lingual."""
import pytest

from models.music_lyrics.multilingual import (
    CorpusSource,
    OPEN_CORPUS_SOURCES,
    METER_TYPES,
    ProsodyAnalysis,
    MultilingualTokenizer,
    MultilingualPoetryAnalyzer,
)


# --- Corpus source registry ---

class TestCorpusSourceRegistry:
    def test_source_count(self):
        assert len(OPEN_CORPUS_SOURCES) >= 25

    def test_sources_frozen(self):
        with pytest.raises(AttributeError):
            OPEN_CORPUS_SOURCES[0].name = "mutated"  # type: ignore[misc]

    def test_all_have_licence(self):
        for src in OPEN_CORPUS_SOURCES:
            assert src.licence, f"{src.name} missing licence"

    def test_all_have_iso639(self):
        for src in OPEN_CORPUS_SOURCES:
            assert 2 <= len(src.iso639) <= 3, f"{src.name} bad iso639: {src.iso639}"

    def test_unique_names(self):
        names = [s.name for s in OPEN_CORPUS_SOURCES]
        assert len(names) == len(set(names))

    def test_language_diversity(self):
        langs = {s.language for s in OPEN_CORPUS_SOURCES}
        assert len(langs) >= 15

    def test_includes_non_western(self):
        traditions = {s.tradition for s in OPEN_CORPUS_SOURCES}
        for t in ("Chinese Classical", "South Asian", "Middle Eastern"):
            assert any(t in tr for tr in traditions), f"Missing tradition: {t}"


# --- Meter types ---

class TestMeterTypes:
    def test_meter_count(self):
        assert len(METER_TYPES) >= 18

    def test_haiku_mora(self):
        h = METER_TYPES["haiku_mora"]
        assert h["mora"] == [5, 7, 5]
        assert h["tradition"] == "Japanese"

    def test_tanka_mora(self):
        assert METER_TYPES["tanka_mora"]["mora"] == [5, 7, 5, 7, 7]

    def test_iambic_pentameter(self):
        ip = METER_TYPES["iambic_pentameter"]
        assert ip["feet"] == 5
        assert ip["pattern"] == "01"

    def test_alexandrine(self):
        assert METER_TYPES["alexandrine"]["feet"] == 6

    def test_luc_bat(self):
        lb = METER_TYPES["luc_bat"]
        assert lb["syllables"] == [6, 8]
        assert lb["tradition"] == "Vietnamese"

    def test_ghazal(self):
        assert "Arabic" in METER_TYPES["ghazal_bahr"]["tradition"]

    def test_shi_regulated(self):
        assert METER_TYPES["shi_regulated"]["characters"] == [5, 7]


# --- MultilingualTokenizer ---

class TestMultilingualTokenizer:
    @pytest.fixture
    def tok(self):
        return MultilingualTokenizer()

    def test_english_tokenize(self, tok):
        tokens = tok.tokenize("The quick brown fox", "en")
        assert tokens == ["The", "quick", "brown", "fox"]

    def test_chinese_tokenize(self, tok):
        tokens = tok.tokenize("床前明月光", "zh")
        assert tokens == ["床", "前", "明", "月", "光"]

    def test_hangul_tokenize(self, tok):
        tokens = tok.tokenize("나는 학생입니다", "ko")
        assert len(tokens) == 2
        assert tokens[0] == "나는"

    def test_detect_script_latin(self, tok):
        assert tok.detect_script("Hello world") == "lat"

    def test_detect_script_chinese(self, tok):
        assert tok.detect_script("静夜思") == "zh"

    def test_detect_script_korean(self, tok):
        assert tok.detect_script("한국어") == "ko"

    def test_detect_script_arabic(self, tok):
        assert tok.detect_script("مرحبا") == "ar"

    def test_detect_script_devanagari(self, tok):
        assert tok.detect_script("नमस्ते") == "hi"

    def test_detect_script_japanese(self, tok):
        assert tok.detect_script("ありがとう") == "ja"

    def test_detect_script_empty(self, tok):
        assert tok.detect_script("") == "en"

    def test_auto_detect(self, tok):
        tokens = tok.tokenize("春眠不觉晓")
        assert len(tokens) == 5

    def test_syllable_count_english(self, tok):
        assert tok.syllable_count("hello world", "en") == 3

    def test_syllable_count_chinese(self, tok):
        assert tok.syllable_count("床前明月光", "zh") == 5

    def test_syllable_count_japanese(self, tok):
        count = tok.syllable_count("ふるいけや", "ja")
        assert count == 5

    def test_syllable_count_korean(self, tok):
        count = tok.syllable_count("한국어", "ko")
        assert count == 3

    def test_cjk_mixed_text(self, tok):
        tokens = tok.tokenize("Hello世界", "zh")
        assert "世" in tokens
        assert "界" in tokens

    def test_punctuation_stripped_cjk(self, tok):
        tokens = tok.tokenize("春眠不觉晓，处处闻啼鸟。", "zh")
        assert "，" not in tokens
        assert "。" not in tokens
        assert len(tokens) == 10

    def test_english_apostrophe(self, tok):
        tokens = tok.tokenize("don't isn't won't", "en")
        assert "don't" in tokens


# --- ProsodyAnalysis ---

class TestProsodyAnalysis:
    def test_default_values(self):
        pa = ProsodyAnalysis()
        assert pa.meter_type == "unknown"
        assert pa.syllable_counts == []
        assert pa.stress_regularity == 0.0

    def test_fields_assignable(self):
        pa = ProsodyAnalysis(meter_type="iambic_pentameter", syllable_counts=[10, 10, 10])
        assert pa.meter_type == "iambic_pentameter"
        assert len(pa.syllable_counts) == 3


# --- MultilingualPoetryAnalyzer ---

class TestMultilingualPoetryAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return MultilingualPoetryAnalyzer()

    def test_analyze_empty(self, analyzer):
        pa = analyzer.analyze_prosody("", "en")
        assert pa.meter_type == "unknown"

    def test_english_sonnet_regularity(self, analyzer):
        sonnet = (
            "Shall I compare thee to a summer's day?\n"
            "Thou art more lovely and more temperate:\n"
            "Rough winds do shake the darling buds of May,\n"
            "And summer's lease hath all too short a date."
        )
        pa = analyzer.analyze_prosody(sonnet, "en")
        assert pa.stress_regularity > 0.3
        assert len(pa.syllable_counts) == 4

    def test_haiku_classification(self, analyzer):
        haiku = "ふるいけや\nかわずとびこむ\nみずのおと"
        pa = analyzer.analyze_prosody(haiku, "ja")
        assert pa.meter_type == "haiku"
        assert pa.syllable_counts == [5, 7, 5]

    def test_tanka_classification(self, analyzer):
        tanka = "あきのたの\nかりほのいほの\nとまをあらみ\nわがころもでは\nつゆにぬれつつ"
        pa = analyzer.analyze_prosody(tanka, "ja")
        assert pa.meter_type in ("tanka", "mixed_meter")
        assert len(pa.syllable_counts) == 5

    def test_chinese_wuyan(self, analyzer):
        poem = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
        pa = analyzer.analyze_prosody(poem, "zh")
        assert pa.meter_type == "wuyan (five-character)"
        assert all(s == 5 for s in pa.syllable_counts)

    def test_chinese_qiyan(self, analyzer):
        poem = "白日依山尽黄河\n入海流欲穷千里\n目更上一层楼好\n诗句传天下无穷"
        pa = analyzer.analyze_prosody(poem, "zh")
        assert pa.meter_type == "qiyan (seven-character)"

    def test_free_verse_classification(self, analyzer):
        text = (
            "I saw the best minds of my generation\n"
            "destroyed by\n"
            "madness\n"
            "starving hysterical naked"
        )
        pa = analyzer.analyze_prosody(text, "en")
        assert pa.meter_type in ("free_verse", "mixed_meter")

    def test_rhyme_density_couplets(self, analyzer):
        couplet = "The cat sat on the mat\nThe bat flew to the hat\nHe found a welcome mat"
        pa = analyzer.analyze_prosody(couplet, "en")
        assert pa.rhyme_density > 0.0

    def test_alliteration_density(self, analyzer):
        text = "Peter Piper picked a peck of pickled peppers"
        pa = analyzer.analyze_prosody(text, "en")
        assert pa.alliteration_density > 0.0

    def test_enjambment_ratio(self, analyzer):
        enjambed = "The river flows\nthrough the valley wide\nand deep below"
        pa = analyzer.analyze_prosody(enjambed, "en")
        assert pa.enjambment_ratio > 0.0

    def test_no_enjambment(self, analyzer):
        end_stopped = "The cat sat on the mat.\nThe dog lay on the rug.\nThe bird sang in the tree."
        pa = analyzer.analyze_prosody(end_stopped, "en")
        assert pa.enjambment_ratio == 0.0

    def test_caesura_positions(self, analyzer):
        text = "To be, or not to be\nThat is the question—whether"
        pa = analyzer.analyze_prosody(text, "en")
        assert len(pa.caesura_positions) > 0
        assert all(0.0 < p < 1.0 for p in pa.caesura_positions)

    def test_cross_lingual_similarity_same_text(self, analyzer):
        text = "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate."
        sim = analyzer.cross_lingual_similarity(text, "en", text, "en")
        assert abs(sim - 1.0) < 0.01

    def test_cross_lingual_similarity_different(self, analyzer):
        eng = "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate."
        chi = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
        sim = analyzer.cross_lingual_similarity(eng, "en", chi, "zh")
        assert 0.0 <= sim <= 1.0

    def test_cross_lingual_similarity_empty(self, analyzer):
        sim = analyzer.cross_lingual_similarity("", "en", "", "zh")
        assert sim == 0.0

    def test_meter_distribution(self, analyzer):
        texts = [
            ("Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.", "en"),
            ("床前明月光\n疑是地上霜\n举头望明月\n低头思故乡", "zh"),
        ]
        dist = analyzer.meter_distribution(texts)
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 2

    def test_luc_bat_classification(self, analyzer):
        text = ("trăm năm trong cõi người ta\n"
                "chữ tài chữ mệnh khéo là ghét nhau\n"
                "trải qua một cuộc bể dâu\n"
                "mà đau lòng thấy trông buồn tình thê")
        pa = analyzer.analyze_prosody(text, "vi")
        assert pa.syllable_counts[0] == 6
        assert pa.syllable_counts[1] == 8
        if all(s in (6, 8) for s in pa.syllable_counts):
            assert pa.meter_type == "luc_bat"
        else:
            assert pa.meter_type in ("luc_bat", "mixed_meter")

    def test_syllable_counts_populated(self, analyzer):
        text = "hello world today\ngoodbye cruel world"
        pa = analyzer.analyze_prosody(text, "en")
        assert len(pa.syllable_counts) == 2
        assert all(s > 0 for s in pa.syllable_counts)


# --- Import from __init__ ---

class TestMultilingualExports:
    def test_import_from_package(self):
        from models.music_lyrics import (
            CorpusSource,
            OPEN_CORPUS_SOURCES,
            METER_TYPES,
            ProsodyAnalysis,
            MultilingualTokenizer,
            MultilingualPoetryAnalyzer,
        )
        assert callable(MultilingualTokenizer)
        assert callable(MultilingualPoetryAnalyzer)
        assert isinstance(OPEN_CORPUS_SOURCES, list)
        assert isinstance(METER_TYPES, dict)
