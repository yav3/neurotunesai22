"""Tests for the music_ir module: fingerprinting, chroma, onset, similarity search."""
import numpy as np
import pytest

from models.music_ir.fingerprint import (
    AudioFingerprinter,
    ConstellationMap,
    FingerprintHash,
    SpectrogramPeaks,
)
from models.music_ir.chromagram import (
    CHORD_TEMPLATES,
    KRUMHANSL_KESSLER_MAJOR,
    KRUMHANSL_KESSLER_MINOR,
    PITCH_CLASSES,
    ChordRecognizer,
    ChromaDistance,
    ChromaExtractor,
    KeyDetector,
)
from models.music_ir.onset import (
    BeatTracker,
    OnsetDetector,
    SegmentBoundaryDetector,
    TempoHistogram,
)
from models.music_ir.similarity_search import (
    CoverSongDetector,
    FeatureIndex,
    LocalitySensitiveHash,
    SequenceAlignment,
)


# ---------- Fingerprint ----------

class TestSpectrogramPeaks:
    def test_find_peaks_basic(self):
        spec = np.random.rand(64, 100).astype(np.float32)
        peaks = SpectrogramPeaks.find_peaks(spec, threshold=0.5)
        assert isinstance(peaks, list)
        for t, f in peaks:
            assert 0 <= t < 100
            assert 0 <= f < 64

    def test_find_peaks_empty(self):
        spec = np.zeros((0, 0), dtype=np.float32)
        assert SpectrogramPeaks.find_peaks(spec) == []

    def test_find_peaks_uniform(self):
        spec = np.ones((10, 10), dtype=np.float32) * 5.0
        peaks = SpectrogramPeaks.find_peaks(spec, threshold=4.0, neighborhood_size=3)
        assert len(peaks) > 0


class TestFingerprintHash:
    def test_hash_deterministic(self):
        h1 = FingerprintHash.hash_pair((10, 200), (15, 300))
        h2 = FingerprintHash.hash_pair((10, 200), (15, 300))
        assert h1 == h2

    def test_hash_different_pairs(self):
        h1 = FingerprintHash.hash_pair((10, 200), (15, 300))
        h2 = FingerprintHash.hash_pair((10, 200), (20, 400))
        assert h1 != h2

    def test_hash_is_int(self):
        h = FingerprintHash.hash_pair((0, 0), (5, 100))
        assert isinstance(h, int)


class TestConstellationMap:
    def test_build(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float32)
        cmap = ConstellationMap()
        peaks = cmap.build(signal, sr=22050, n_peaks=50)
        assert len(peaks) <= 50
        assert all(isinstance(p, tuple) and len(p) == 2 for p in peaks)

    def test_visualize(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float32)
        cmap = ConstellationMap()
        cmap.build(signal, sr=22050, n_peaks=20)
        vis = cmap.visualize()
        assert vis.ndim == 2
        assert vis.dtype == np.float32

    def test_empty_signal(self):
        cmap = ConstellationMap()
        peaks = cmap.build(np.array([]), sr=22050, n_peaks=10)
        assert len(peaks) <= 10


class TestAudioFingerprinter:
    def test_fingerprint(self):
        rng = np.random.default_rng(99)
        signal = rng.standard_normal(44100).astype(np.float32)
        fp = AudioFingerprinter(fan_out=5)
        hashes = fp.fingerprint(signal, sr=22050)
        assert isinstance(hashes, list)
        assert all(isinstance(h, int) for h in hashes)

    def test_match_self(self):
        rng = np.random.default_rng(99)
        signal = rng.standard_normal(44100).astype(np.float32)
        fp = AudioFingerprinter(fan_out=5)
        hashes = fp.fingerprint(signal, sr=22050)
        db = {"track_a": hashes}
        results = fp.match(hashes, db)
        assert len(results) == 1
        assert results[0][0] == "track_a"
        assert results[0][1] == 1.0

    def test_match_empty(self):
        fp = AudioFingerprinter()
        assert fp.match([], {}) == []
        assert fp.match([1, 2, 3], {}) == []
        assert fp.match([], {"a": [1]}) == []


# ---------- Chromagram ----------

class TestChromaExtractor:
    def test_extract_shape(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float32)
        ext = ChromaExtractor(n_fft=2048, hop=512)
        chroma = ext.extract(signal, sr=22050)
        assert chroma.shape[0] == 12
        assert chroma.shape[1] > 0

    def test_extract_normalized(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float32)
        ext = ChromaExtractor()
        chroma = ext.extract(signal, sr=22050)
        assert chroma.max() <= 1.0 + 1e-6

    def test_extract_empty(self):
        ext = ChromaExtractor()
        chroma = ext.extract(np.array([]), sr=22050)
        assert chroma.shape[0] == 12
        assert chroma.shape[1] == 0


class TestChromaDistance:
    def test_cosine_identical(self):
        a = np.random.rand(12, 20).astype(np.float32)
        assert ChromaDistance.cosine(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_euclidean_zero(self):
        a = np.random.rand(12, 20).astype(np.float32)
        assert ChromaDistance.euclidean(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_correlation(self):
        a = np.random.rand(12, 20).astype(np.float32)
        b = np.random.rand(12, 20).astype(np.float32)
        corr = ChromaDistance.correlation(a, b)
        assert -1.0 <= corr <= 1.0


class TestKeyDetector:
    def test_detect_returns_valid(self):
        rng = np.random.default_rng(42)
        chroma = rng.random((12, 50)).astype(np.float32)
        det = KeyDetector()
        key, mode, conf = det.detect(chroma)
        assert key in PITCH_CLASSES
        assert mode in ("major", "minor")
        assert 0.0 <= conf <= 1.0

    def test_detect_c_major_biased(self):
        chroma = np.zeros((12, 20), dtype=np.float32)
        chroma[0, :] = 1.0  # C
        chroma[4, :] = 0.8  # E
        chroma[7, :] = 0.7  # G
        det = KeyDetector()
        key, mode, conf = det.detect(chroma)
        assert key == "C"
        assert mode == "major"

    def test_detect_local(self):
        rng = np.random.default_rng(42)
        chroma = rng.random((12, 40)).astype(np.float32)
        det = KeyDetector()
        results = det.detect_local(chroma, window=10)
        assert len(results) == 4
        for key, mode, conf in results:
            assert key in PITCH_CLASSES

    def test_profiles_shape(self):
        assert KRUMHANSL_KESSLER_MAJOR.shape == (12,)
        assert KRUMHANSL_KESSLER_MINOR.shape == (12,)


class TestChordRecognizer:
    def test_recognize_c_major(self):
        frame = np.zeros(12)
        frame[0] = 1.0  # C
        frame[4] = 1.0  # E
        frame[7] = 1.0  # G
        rec = ChordRecognizer()
        chord, conf = rec.recognize_frame(frame)
        assert chord == "C major"
        assert conf > 0.9

    def test_recognize_a_minor(self):
        frame = np.zeros(12)
        frame[9] = 1.0   # A
        frame[0] = 1.0   # C
        frame[4] = 1.0   # E
        rec = ChordRecognizer()
        chord, conf = rec.recognize_frame(frame)
        assert "minor" in chord or "major" in chord
        assert conf > 0.5

    def test_recognize_sequence(self):
        chroma = np.random.rand(12, 10).astype(np.float32)
        rec = ChordRecognizer()
        seq = rec.recognize_sequence(chroma)
        assert len(seq) == 10

    def test_templates_count(self):
        assert len(CHORD_TEMPLATES) == 24


# ---------- Onset ----------

class TestOnsetDetector:
    def test_spectral_flux(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float32)
        det = OnsetDetector()
        flux = det.spectral_flux(signal)
        assert flux.ndim == 1
        assert len(flux) > 0

    def test_energy(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float32)
        det = OnsetDetector()
        e = det.energy(signal)
        assert e.ndim == 1

    def test_complex_domain(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float32)
        det = OnsetDetector()
        cd = det.complex_domain(signal)
        assert cd.ndim == 1

    def test_pick_onsets(self):
        odf = np.array([0, 1, 5, 2, 0, 0, 8, 3, 1, 0], dtype=float)
        det = OnsetDetector()
        peaks = det.pick_onsets(odf, threshold=3.0, min_spacing=2)
        assert 2 in peaks
        assert 6 in peaks


class TestBeatTracker:
    def test_track(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(44100).astype(np.float32)
        bt = BeatTracker()
        beats = bt.track(signal, sr=22050)
        assert isinstance(beats, list)
        assert all(isinstance(b, float) for b in beats)

    def test_estimate_tempo(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(44100).astype(np.float32)
        bt = BeatTracker()
        tempo = bt.estimate_tempo(signal, sr=22050)
        assert isinstance(tempo, float)
        assert tempo >= 0


class TestTempoHistogram:
    def test_build(self):
        onsets = [0.0, 0.5, 1.0, 1.5, 2.0]
        th = TempoHistogram()
        hist, bins = th.build(onsets)
        assert len(hist) == 270
        assert len(bins) == 271

    def test_dominant_tempo(self):
        onsets = [i * 0.5 for i in range(20)]
        th = TempoHistogram()
        tempo = th.dominant_tempo(onsets)
        assert 100 < tempo < 140

    def test_empty(self):
        th = TempoHistogram()
        assert th.dominant_tempo([]) == 0.0
        assert th.dominant_tempo([1.0]) == 0.0


# ---------- Similarity Search ----------

class TestFeatureIndex:
    def test_add_and_query(self):
        idx = FeatureIndex(dim=16)
        rng = np.random.default_rng(42)
        for i in range(10):
            idx.add(f"track_{i}", rng.standard_normal(16))
        results = idx.query(rng.standard_normal(16), k=3)
        assert len(results) == 3
        assert all(isinstance(r[0], str) for r in results)

    def test_self_query(self):
        idx = FeatureIndex(dim=8)
        v = np.ones(8)
        idx.add("x", v)
        results = idx.query(v, k=1)
        assert results[0][0] == "x"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_len(self):
        idx = FeatureIndex(dim=4)
        assert len(idx) == 0
        idx.add("a", np.ones(4))
        assert len(idx) == 1


class TestLSH:
    def test_add_query(self):
        lsh = LocalitySensitiveHash(dim=16, n_tables=5, n_bits=4)
        rng = np.random.default_rng(42)
        for i in range(20):
            lsh.add(f"t_{i}", rng.standard_normal(16))
        v = rng.standard_normal(16)
        lsh.add("target", v)
        results = lsh.query(v, k=5)
        ids = [r[0] for r in results]
        assert "target" in ids

    def test_len(self):
        lsh = LocalitySensitiveHash(dim=8)
        assert len(lsh) == 0
        lsh.add("a", np.ones(8))
        assert len(lsh) == 1


class TestSequenceAlignment:
    def test_dtw_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert SequenceAlignment.dtw(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_dtw_different(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert SequenceAlignment.dtw(a, b) > 0

    def test_dtw_path(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        path = SequenceAlignment.dtw_path(a, b)
        assert len(path) == 3
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    def test_smith_waterman(self):
        a = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
        score, path = SequenceAlignment.smith_waterman(a, b, threshold=0.5)
        assert score > 0
        assert len(path) > 0

    def test_dtw_empty(self):
        assert SequenceAlignment.dtw(np.array([]), np.array([1.0])) == float("inf")


class TestCoverSongDetector:
    def test_compare_identical(self):
        rng = np.random.default_rng(42)
        chroma = rng.random((12, 30)).astype(np.float32)
        det = CoverSongDetector()
        result = det.compare(chroma, chroma)
        assert "dtw_distance" in result
        assert "oti_score" in result
        assert "cover_score" in result
        assert result["oti_score"] > 0.5

    def test_compare_empty(self):
        det = CoverSongDetector()
        result = det.compare(np.array([]), np.random.rand(12, 10))
        assert result["dtw_distance"] == float("inf")
        assert result["cover_score"] == 0.0


# ---------- Search ----------

from models.music_ir.search import InvertedIndex, TrackIndex, QueryExpander, SearchRanker


class TestInvertedIndex:
    def test_add_and_search(self):
        idx = InvertedIndex()
        idx.add("t1", ["rock", "guitar", "drums"])
        idx.add("t2", ["jazz", "piano", "smooth"])
        idx.add("t3", ["rock", "piano", "heavy"])
        results = idx.search(["rock"], top_k=5)
        ids = [r[0] for r in results]
        assert "t1" in ids
        assert "t3" in ids
        assert "t2" not in ids

    def test_tfidf_ranking(self):
        idx = InvertedIndex()
        idx.add("t1", ["rock", "rock", "guitar"])
        idx.add("t2", ["rock", "jazz"])
        results = idx.search(["rock"], top_k=2)
        assert results[0][0] == "t1"
        assert results[0][1] > results[1][1]

    def test_empty_query(self):
        idx = InvertedIndex()
        idx.add("t1", ["rock"])
        assert idx.search([]) == []

    def test_empty_index(self):
        idx = InvertedIndex()
        assert idx.search(["rock"]) == []

    def test_case_insensitive(self):
        idx = InvertedIndex()
        idx.add("t1", ["Rock", "GUITAR"])
        results = idx.search(["rock"])
        assert len(results) == 1
        assert results[0][0] == "t1"

    def test_num_docs(self):
        idx = InvertedIndex()
        assert idx.num_docs == 0
        idx.add("t1", ["rock"])
        assert idx.num_docs == 1

    def test_vocab_size(self):
        idx = InvertedIndex()
        idx.add("t1", ["rock", "guitar"])
        assert idx.vocab_size == 2


class TestTrackIndex:
    def test_text_search(self):
        ti = TrackIndex()
        ti.index_track("t1", {"title": "chill vibes", "genre": "ambient"})
        ti.index_track("t2", {"title": "party dance", "genre": "pop"})
        results = ti.text_search("chill", top_k=2)
        assert results[0][0] == "t1"

    def test_feature_search(self):
        ti = TrackIndex()
        ti.index_track("t1", {"title": "a", "bpm": 120, "energy": 0.8})
        ti.index_track("t2", {"title": "b", "bpm": 80, "energy": 0.3})
        results = ti.feature_search({"bpm": 120, "energy": 0.8}, top_k=2)
        assert results[0][0] == "t1"

    def test_num_tracks(self):
        ti = TrackIndex()
        assert ti.num_tracks == 0
        ti.index_track("t1", {"title": "test"})
        assert ti.num_tracks == 1


class TestQueryExpander:
    def test_expand_known_term(self):
        expanded = QueryExpander.expand(["rock"])
        assert "rock" in expanded
        assert len(expanded) > 1

    def test_expand_unknown_term(self):
        expanded = QueryExpander.expand(["xyzzy_unknown"])
        assert expanded == ["xyzzy_unknown"]

    def test_expand_empty(self):
        assert QueryExpander.expand([]) == []


class TestSearchRanker:
    def test_rerank_basic(self):
        ranker = SearchRanker()
        text_results = [("t1", 0.8), ("t2", 0.6), ("t3", 0.9)]
        feat_results = [("t1", 0.1), ("t2", 0.5), ("t3", 0.2)]
        ranked = ranker.rerank(text_results, feat_results, top_k=2)
        assert len(ranked) == 2

    def test_rerank_feature_tiebreak(self):
        ranker = SearchRanker(text_weight=0.5, feature_weight=0.5)
        text_results = [("t1", 0.5), ("t2", 0.5)]
        feat_results = [("t1", 0.0), ("t2", 1.0)]
        ranked = ranker.rerank(text_results, feat_results, top_k=2)
        assert ranked[0][0] == "t1"

    def test_rerank_empty(self):
        ranker = SearchRanker()
        ranked = ranker.rerank([], [])
        assert ranked == []


# ---------- Tagging ----------

from models.music_ir.tagging import AutoTagger, GenreClassifier, MoodTagger, InstrumentDetector


class TestAutoTagger:
    def test_tag_electronic(self):
        tagger = AutoTagger()
        tags = tagger.tag({"acousticness": 0.1, "energy": 0.8})
        assert "electronic" in tags

    def test_tag_acoustic(self):
        tagger = AutoTagger()
        tags = tagger.tag({"acousticness": 0.9})
        assert "acoustic" in tags

    def test_tag_empty_features(self):
        tagger = AutoTagger()
        tags = tagger.tag({})
        assert isinstance(tags, list)

    def test_tag_confidence(self):
        tagger = AutoTagger()
        results = tagger.tag_confidence({"acousticness": 0.85})
        assert len(results) > 0
        assert all(0 < conf <= 1.0 for _, conf in results)
        assert results[0][1] >= results[-1][1]


class TestGenreClassifier:
    def test_classify_pop(self):
        gc = GenreClassifier()
        results = gc.classify({"bpm": 120, "valence": 0.65, "energy": 0.65,
                               "danceability": 0.7, "acousticness": 0.3})
        assert results[0][0] == "Pop"

    def test_classify_metal(self):
        gc = GenreClassifier()
        results = gc.classify({"bpm": 140, "valence": 0.35, "energy": 0.95,
                               "danceability": 0.3, "acousticness": 0.15})
        assert results[0][0] == "Metal"

    def test_classify_top_k(self):
        gc = GenreClassifier()
        results = gc.classify({"bpm": 120, "energy": 0.5}, top_k=5)
        assert len(results) == 5

    def test_classify_all_similarities_positive(self):
        gc = GenreClassifier()
        results = gc.classify({"bpm": 100}, top_k=10)
        assert all(sim > 0 for _, sim in results)


class TestMoodTagger:
    def test_happy_mood(self):
        mt = MoodTagger()
        results = mt.tag(valence=0.8, arousal=0.7, dominance=0.6)
        assert results[0][0] == "Happy"

    def test_sad_mood(self):
        mt = MoodTagger()
        results = mt.tag(valence=0.2, arousal=0.2, dominance=0.3)
        assert results[0][0] == "Sad"

    def test_top_k(self):
        mt = MoodTagger()
        results = mt.tag(valence=0.5, arousal=0.5, top_k=5)
        assert len(results) == 5

    def test_similarities_positive(self):
        mt = MoodTagger()
        results = mt.tag(valence=0.5, arousal=0.5, top_k=15)
        assert all(sim > 0 for _, sim in results)


class TestInstrumentDetector:
    def test_detect_drums(self):
        det = InstrumentDetector()
        results = det.detect({"spectral_centroid": 0.6, "spectral_flatness": 0.7})
        instruments = [r[0] for r in results]
        assert "drums" in instruments

    def test_detect_bass(self):
        det = InstrumentDetector()
        results = det.detect({"spectral_centroid": 0.08, "spectral_flatness": 0.25})
        instruments = [r[0] for r in results]
        assert "bass" in instruments

    def test_detect_threshold(self):
        det = InstrumentDetector()
        results_low = det.detect({"spectral_centroid": 0.5, "spectral_flatness": 0.5}, threshold=0.1)
        results_high = det.detect({"spectral_centroid": 0.5, "spectral_flatness": 0.5}, threshold=0.9)
        assert len(results_low) >= len(results_high)


# ---------- Segmentation ----------

from models.music_ir.segmentation import StructuralSegmenter, ChromaAnalyzer


class TestStructuralSegmenter:
    def test_segment_basic(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(44100).astype(np.float64)
        seg = StructuralSegmenter()
        segments = seg.segment(signal, sr=22050, n_segments=3)
        assert len(segments) >= 1
        assert all("start" in s and "end" in s and "label" in s for s in segments)

    def test_segment_labels_unique(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(44100).astype(np.float64)
        seg = StructuralSegmenter()
        segments = seg.segment(signal, sr=22050, n_segments=4)
        labels = [s["label"] for s in segments]
        assert labels[0] == "A"

    def test_segment_short_signal(self):
        seg = StructuralSegmenter()
        segments = seg.segment(np.zeros(100), sr=22050)
        assert len(segments) == 1
        assert segments[0]["label"] == "A"

    def test_self_similarity_shape(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float64)
        seg = StructuralSegmenter()
        ssm = seg.self_similarity(signal, sr=22050)
        assert ssm.ndim == 2
        assert ssm.shape[0] == ssm.shape[1]

    def test_self_similarity_diagonal(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float64)
        seg = StructuralSegmenter()
        ssm = seg.self_similarity(signal, sr=22050)
        diag = np.diag(ssm)
        assert np.all(diag >= 0.99)


class TestChromaAnalyzer:
    def test_chromagram_shape(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float64)
        ca = ChromaAnalyzer()
        cg = ca.chromagram(signal, sr=22050)
        assert cg.shape[0] == 12
        assert cg.shape[1] > 0

    def test_chromagram_normalized(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float64)
        ca = ChromaAnalyzer()
        cg = ca.chromagram(signal, sr=22050)
        col_norms = np.linalg.norm(cg, axis=0)
        assert np.allclose(col_norms, 1.0, atol=1e-4)

    def test_key_profile_sums_to_one(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float64)
        ca = ChromaAnalyzer()
        profile = ca.key_profile(signal, sr=22050)
        assert len(profile) == 12
        assert abs(sum(profile.values()) - 1.0) < 1e-6

    def test_estimate_key(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(22050).astype(np.float64)
        ca = ChromaAnalyzer()
        note, mode = ca.estimate_key(signal, sr=22050)
        assert note in ChromaAnalyzer.NOTE_NAMES
        assert mode in ("major", "minor")

    def test_key_profile_silent(self):
        ca = ChromaAnalyzer()
        profile = ca.key_profile(np.zeros(4096), sr=22050)
        assert all(v == 0.0 for v in profile.values())
