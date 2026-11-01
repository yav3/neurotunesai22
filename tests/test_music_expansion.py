"""Tests for expanded modules: transforms, copyright, semantics, cognition, composition."""
from __future__ import annotations
import pytest
import numpy as np

# ── Transforms: harmonic ──────────────────────────────────────────────
from models.music_transforms.harmonic import (
    hpss, harmonic_product_spectrum, harmonic_to_noise_ratio,
    subharmonic_summation, autocorrelation_pitch, zero_crossing_rate,
)

class TestHarmonicTransforms:
    def test_hpss_returns_two_arrays(self):
        sig = np.random.default_rng(42).standard_normal(8000)
        h, p = hpss(sig, frame_size=512, hop=256, kernel_size=7)
        assert len(h) == len(p)
        assert len(h) > 0

    def test_hps_detects_fundamental(self):
        sr = 16000.0
        t = np.arange(16000) / sr
        sig = (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
               + 0.3 * np.sin(2 * np.pi * 600 * t) + 0.2 * np.sin(2 * np.pi * 800 * t))
        f0, _ = harmonic_product_spectrum(sig, sr=sr, n_fft=16000)
        assert 190 < f0 < 210

    def test_hnr_shape(self):
        sig = np.random.default_rng(1).standard_normal(4096)
        hnr = harmonic_to_noise_ratio(sig, sr=16000, frame_size=1024, hop=512)
        expected_frames = max(1, (4096 - 1024) // 512 + 1)
        assert len(hnr) == expected_frames

    def test_subharmonic_summation(self):
        spectrum = np.zeros(2049)
        spectrum[100] = 1.0
        spectrum[200] = 0.8
        spectrum[300] = 0.5
        f0, shs = subharmonic_summation(spectrum, sr=44100, n_fft=4096)
        assert f0 > 0

    def test_autocorrelation_pitch_sine(self):
        sr = 8000.0
        t = np.arange(4000) / sr
        sig = np.sin(2 * np.pi * 200 * t)
        f0 = autocorrelation_pitch(sig, sr=sr)
        assert 190 < f0 < 210

    def test_zcr_shape(self):
        sig = np.random.default_rng(7).standard_normal(4096)
        zcr = zero_crossing_rate(sig, frame_size=1024, hop=512)
        assert len(zcr) > 0
        assert all(0 <= z <= 0.5 for z in zcr)


# ── Transforms: wavelet ──────────────────────────────────────────────
from models.music_transforms.wavelet import (
    cwt_morlet, scalogram, dwt_haar, idwt_haar, morlet_wavelet,
    wavelet_energy_bands, wavelet_entropy, multi_resolution_analysis,
    wavelet_packet_decomposition,
)

class TestWaveletTransforms:
    def test_morlet_shape(self):
        w = morlet_wavelet(128)
        assert len(w) == 128
        assert np.iscomplexobj(w)

    def test_cwt_shape(self):
        sig = np.random.default_rng(0).standard_normal(64)
        scales = np.arange(1, 9, dtype=np.float64)
        c = cwt_morlet(sig, scales)
        assert c.shape == (8, 64)

    def test_scalogram_nonnegative(self):
        sig = np.random.default_rng(1).standard_normal(128)
        s = scalogram(sig, np.arange(1, 9, dtype=np.float64))
        assert np.all(s >= 0)

    def test_dwt_idwt_roundtrip(self):
        sig = np.random.default_rng(2).standard_normal(64)
        coeffs = dwt_haar(sig, levels=3)
        recon = idwt_haar(coeffs)
        np.testing.assert_allclose(recon, sig, atol=1e-10)

    def test_energy_bands_sum_to_one(self):
        sig = np.random.default_rng(3).standard_normal(256)
        bands = wavelet_energy_bands(sig, levels=4)
        assert abs(np.sum(bands) - 1.0) < 1e-6

    def test_entropy_nonnegative(self):
        sig = np.random.default_rng(4).standard_normal(128)
        ent = wavelet_entropy(sig, levels=4)
        assert ent >= 0

    def test_multi_resolution_keys(self):
        sig = np.random.default_rng(5).standard_normal(64)
        mra = multi_resolution_analysis(sig, levels=3)
        assert "approximation" in mra
        assert "detail_level_1" in mra

    def test_wavelet_packets(self):
        sig = np.random.default_rng(6).standard_normal(32)
        packets = wavelet_packet_decomposition(sig, max_level=2)
        assert "0_0" in packets
        assert "2_0" in packets


# ── Transforms: filterbank ───────────────────────────────────────────
from models.music_transforms.filterbank import (
    mel_filterbank as mel_fb, bark_filterbank, erb_filterbank,
    gammatone_filterbank, octave_band_edges, third_octave_band_edges,
    windowed_sinc_lowpass, apply_filter,
)

class TestFilterbankTransforms:
    def test_mel_fb_shape(self):
        fb = mel_fb(n_filters=40, n_fft=2048)
        assert fb.shape == (40, 1025)
        assert np.all(fb >= 0)

    def test_bark_fb_shape(self):
        fb = bark_filterbank(n_filters=24, n_fft=2048)
        assert fb.shape == (24, 1025)

    def test_erb_fb_shape(self):
        fb = erb_filterbank(n_filters=32, n_fft=2048)
        assert fb.shape == (32, 1025)

    def test_gammatone_shape(self):
        fb = gammatone_filterbank(n_filters=16, filter_length=512)
        assert fb.shape == (16, 512)

    def test_octave_bands(self):
        bands = octave_band_edges(n_bands=10)
        assert bands.shape == (10, 2)
        assert all(bands[i, 0] < bands[i, 1] for i in range(10))

    def test_third_octave_bands(self):
        bands = third_octave_band_edges(n_bands=30)
        assert bands.shape == (30, 2)

    def test_lowpass_sums_to_one(self):
        h = windowed_sinc_lowpass(1000, 44100, n_taps=101)
        assert abs(np.sum(h) - 1.0) < 0.01

    def test_apply_filter_preserves_length(self):
        sig = np.random.default_rng(0).standard_normal(1000)
        h = windowed_sinc_lowpass(2000, 44100, n_taps=51)
        out = apply_filter(sig, h)
        assert len(out) == len(sig)


# ── Copyright: fingerprint ───────────────────────────────────────────
from models.music_copyright.fingerprint import (
    FingerprintDatabase, find_spectral_peaks, generate_fingerprints,
    fingerprint_similarity, SpectralPeak, FingerprintHash,
)

class TestFingerprint:
    def test_find_peaks(self):
        spec = np.random.default_rng(0).random((128, 50))
        peaks = find_spectral_peaks(spec, n_peaks_per_frame=3)
        assert len(peaks) > 0
        assert all(isinstance(p, SpectralPeak) for p in peaks)

    def test_generate_fingerprints(self):
        peaks = [SpectralPeak(i, i * 3, 0.5) for i in range(20)]
        hashes = generate_fingerprints(peaks, fan_out=5, max_delta=50)
        assert len(hashes) > 0

    def test_database_insert_and_query(self):
        db = FingerprintDatabase()
        hashes = [FingerprintHash(10, 20, 5, t) for t in range(10)]
        db.insert("track_a", hashes)
        results = db.query(hashes, min_matches=3)
        assert len(results) >= 1
        assert results[0][0] == "track_a"

    def test_similarity_identical(self):
        hashes = [FingerprintHash(i, i + 5, 3, i) for i in range(10)]
        assert fingerprint_similarity(hashes, hashes) == pytest.approx(1.0)

    def test_similarity_disjoint(self):
        a = [FingerprintHash(1, 2, 3, 0)]
        b = [FingerprintHash(100, 200, 300, 0)]
        assert fingerprint_similarity(a, b) == pytest.approx(0.0)


# ── Copyright: rights model ──────────────────────────────────────────
from models.music_copyright.rights_model import (
    RightsHolder, RightsChain, RightType, calculate_split,
    mechanical_royalty_rate, streaming_royalty_estimate, STANDARD_SPLITS,
)

class TestRightsModel:
    def test_holder_validate(self):
        h = RightsHolder("Alice", 0.5, RightType.COMPOSITION)
        assert h.validate()
        h2 = RightsHolder("Bob", 1.5, RightType.LYRICS)
        assert not h2.validate()

    def test_chain_total_share(self):
        chain = RightsChain()
        chain.add_holder(RightsHolder("A", 0.6, RightType.COMPOSITION))
        chain.add_holder(RightsHolder("B", 0.4, RightType.COMPOSITION))
        assert chain.total_share(RightType.COMPOSITION) == pytest.approx(1.0)

    def test_chain_is_complete(self):
        chain = RightsChain()
        chain.add_holder(RightsHolder("A", 1.0, RightType.LYRICS))
        assert chain.is_complete(RightType.LYRICS)

    def test_chain_validate_errors(self):
        chain = RightsChain()
        chain.add_holder(RightsHolder("A", 0.7, RightType.MASTER_RECORDING))
        chain.add_holder(RightsHolder("B", 0.7, RightType.MASTER_RECORDING))
        errors = chain.validate()
        assert len(errors) > 0

    def test_calculate_split(self):
        split = calculate_split({"guitar": 3, "drums": 1, "bass": 1})
        assert split["guitar"] == pytest.approx(0.6)

    def test_mechanical_rate(self):
        assert mechanical_royalty_rate(2024) == 0.12

    def test_streaming_estimate(self):
        est = streaming_royalty_estimate(1_000_000, 0.004, 0.5)
        assert est == pytest.approx(2000.0)

    def test_standard_splits_exist(self):
        assert len(STANDARD_SPLITS) >= 4


# ── Semantics: mood tags ─────────────────────────────────────────────
from models.music_semantics.mood_tags import (
    MoodCategory, MoodTag, vad_to_mood, TagProfile, tag_similarity,
    MOOD_VAD_MAP, MUSIC_TAG_TAXONOMY,
)

class TestMoodTags:
    def test_mood_vad_map_complete(self):
        assert len(MOOD_VAD_MAP) == len(MoodCategory)

    def test_mood_tag_vad(self):
        tag = MoodTag(MoodCategory.HAPPY, confidence=1.0, intensity=1.0)
        v, a, d = tag.vad
        assert v > 0.5

    def test_vad_to_mood_happy(self):
        results = vad_to_mood(0.8, 0.6, 0.7, top_k=3)
        assert results[0][0] == MoodCategory.HAPPY

    def test_tag_profile(self):
        tp = TagProfile()
        tp.add_tag("genre", "rock", 0.9)
        tp.add_tag("genre", "blues", 0.5)
        top = tp.top_tags("genre", k=1)
        assert top[0][0] == "rock"

    def test_tag_similarity_identical(self):
        tp = TagProfile()
        tp.add_tag("genre", "rock")
        assert tag_similarity(tp, tp) == pytest.approx(1.0)

    def test_taxonomy_genres(self):
        assert "genre" in MUSIC_TAG_TAXONOMY
        assert len(MUSIC_TAG_TAXONOMY["genre"]) >= 15


# ── Semantics: knowledge graph ───────────────────────────────────────
from models.music_semantics.music_knowledge_graph import (
    MusicKnowledgeGraph, Entity, Relation, EntityType, RelationType,
)

class TestKnowledgeGraph:
    def test_add_entity(self):
        g = MusicKnowledgeGraph()
        e = Entity("e1", EntityType.ARTIST, "Bach")
        g.add_entity(e)
        assert "e1" in g.entities

    def test_add_relation_and_neighbors(self):
        g = MusicKnowledgeGraph()
        g.add_entity(Entity("a1", EntityType.ARTIST, "Mozart"))
        g.add_entity(Entity("g1", EntityType.GENRE, "Classical"))
        g.add_relation(Relation("a1", "g1", RelationType.BELONGS_TO_GENRE))
        nbrs = g.neighbors("a1")
        assert len(nbrs) == 1

    def test_path_exists(self):
        g = MusicKnowledgeGraph()
        for i in range(5):
            g.add_entity(Entity(f"n{i}", EntityType.ARTIST, f"Artist {i}"))
        g.add_relation(Relation("n0", "n1", RelationType.INFLUENCES))
        g.add_relation(Relation("n1", "n2", RelationType.INFLUENCES))
        assert g.path_exists("n0", "n2")
        assert not g.path_exists("n0", "n4")

    def test_count_by_type(self):
        g = MusicKnowledgeGraph()
        g.add_entity(Entity("a1", EntityType.ARTIST, "X"))
        g.add_entity(Entity("a2", EntityType.ARTIST, "Y"))
        g.add_entity(Entity("g1", EntityType.GENRE, "Rock"))
        counts = g.count_by_type()
        assert counts[EntityType.ARTIST] == 2


# ── Cognition: auditory scene ────────────────────────────────────────
from models.music_cognition.auditory_scene import (
    SceneAnalyzer, gestalt_proximity, gestalt_similarity,
    attention_salience, RhythmicExpectation, rhythmic_complexity,
)

class TestAuditoryScene:
    def test_segregate_two_streams(self):
        sa = SceneAnalyzer(proximity_threshold=4)
        pitches = [60, 72, 62, 74, 64, 76]
        onsets = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        streams = sa.segregate_by_pitch(pitches, onsets)
        assert len(streams) == 2

    def test_stream_coherence(self):
        sa = SceneAnalyzer()
        assert sa.stream_coherence([60, 62, 64, 65]) > 0.5

    def test_gestalt_proximity(self):
        onsets = [0, 0.1, 0.2, 1.0, 1.1, 1.2]
        groups = gestalt_proximity(onsets, threshold=0.3)
        assert len(groups) == 2

    def test_gestalt_similarity(self):
        features = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]])
        groups = gestalt_similarity(features, threshold=0.5)
        assert len(groups) == 2

    def test_attention_salience_shape(self):
        loudness = [0.5, 0.8, 0.3, 0.9]
        novelty = [0.1, 0.5, 0.2, 0.9]
        sal = attention_salience(loudness, novelty)
        assert len(sal) == 4

    def test_rhythmic_expectation(self):
        re = RhythmicExpectation(beat_period=0.5, phase=0.0)
        predicted = re.predict_next_beat(0.4)
        assert predicted == pytest.approx(0.5)

    def test_rhythmic_complexity_simple(self):
        onsets = [0, 0.5, 1.0, 1.5, 2.0]
        c = rhythmic_complexity(onsets, beat_period=0.5)
        assert c < 0.3

    def test_rhythmic_complexity_syncopated(self):
        onsets = [0, 0.33, 0.77, 1.1, 1.55]
        c = rhythmic_complexity(onsets, beat_period=0.5)
        assert c > 0.1


# ── Composition: melodic development ─────────────────────────────────
from models.music_composition.melodic_development import (
    Motif, transpose, invert, retrograde, retrograde_inversion,
    augmentation, diminution, sequence, fragment,
    contour_similarity, interval_similarity, develop_theme,
)

class TestMelodicDevelopment:
    @pytest.fixture
    def motif(self):
        return Motif(pitches=(60, 64, 67, 65, 62), durations=(0.5, 0.5, 0.25, 0.25, 1.0))

    def test_transpose(self, motif):
        t = transpose(motif, 5)
        assert t.pitches == (65, 69, 72, 70, 67)

    def test_invert(self, motif):
        inv = invert(motif)
        assert inv.pitches[0] == motif.pitches[0]
        assert inv.pitches[1] < motif.pitches[0]

    def test_retrograde(self, motif):
        r = retrograde(motif)
        assert r.pitches == tuple(reversed(motif.pitches))

    def test_retrograde_inversion(self, motif):
        ri = retrograde_inversion(motif)
        assert len(ri.pitches) == len(motif.pitches)

    def test_augmentation(self, motif):
        aug = augmentation(motif, 2.0)
        assert aug.durations[0] == 1.0

    def test_diminution(self, motif):
        dim = diminution(motif, 2.0)
        assert dim.durations[0] == 0.25

    def test_sequence(self, motif):
        seq = sequence(motif, steps=3, interval=2)
        assert len(seq) == 3
        assert seq[1].pitches[0] == 62

    def test_fragment(self, motif):
        f = fragment(motif, 0, 3)
        assert len(f.pitches) == 3

    def test_contour_similarity_identical(self, motif):
        assert contour_similarity(motif, motif) == pytest.approx(1.0)

    def test_interval_similarity_identical(self, motif):
        assert interval_similarity(motif, motif) == pytest.approx(1.0)

    def test_develop_theme(self, motif):
        developed = develop_theme(motif, ["transpose", "invert", "retrograde"])
        assert len(developed) == 4

    def test_motif_intervals(self, motif):
        assert motif.intervals == (4, 3, -2, -3)

    def test_motif_contour(self, motif):
        assert motif.contour == (1, 1, -1, -1)
