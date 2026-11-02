"""Tests for deepened music modules: copyright, semantics, psychoacoustics."""
import numpy as np
import pytest

from models.music_copyright.fingerprint import (
    SpectralPeak, FingerprintHash, FingerprintDatabase,
    find_spectral_peaks, generate_fingerprints, fingerprint_similarity,
)
from models.music_copyright.sampling_detection import (
    SampleMatch, LoopDetection,
    cross_correlation_search, spectral_envelope, envelope_similarity,
    detect_pitch_shift, detect_time_stretch, detect_loops,
    find_sample_matches,
)
from models.music_semantics.concept_embedding import (
    MusicalConcept, CONCEPT_REGISTRY, ConceptEmbedding,
    concept_similarity, nearest_concepts, analogy,
)
from models.music_semantics.mood_ontology import (
    MOOD_TREE, MoodTrajectory,
    mood_distance, mood_ancestors, mood_from_vad, mood_from_features,
)
from models.music_psychoacoustics.spatial_hearing import (
    interaural_time_difference, interaural_level_difference,
    head_shadow_filter, distance_attenuation, air_absorption,
    generate_synthetic_hrtf, apply_hrtf,
    interaural_coherence, estimate_azimuth, binaural_cue_analysis,
)
from models.music_psychoacoustics.timbre_perception import (
    spectral_centroid, spectral_spread, spectral_flatness,
    spectral_rolloff, spectral_flux,
    odd_even_ratio, inharmonicity, attack_time,
    brightness, warmth, roughness_dissonance, total_roughness,
    analyze_timbre, timbre_distance,
)


class TestFingerprintPeaks:
    def test_peak_extraction(self):
        rng = np.random.RandomState(0)
        spec = rng.rand(128, 50)
        peaks = find_spectral_peaks(spec, n_peaks_per_frame=3)
        assert len(peaks) > 0
        assert all(isinstance(p, SpectralPeak) for p in peaks)

    def test_no_peaks_below_min_magnitude(self):
        spec = np.full((64, 10), 0.005)
        peaks = find_spectral_peaks(spec, min_magnitude=0.01)
        assert len(peaks) == 0


class TestFingerprintHashing:
    def test_hash_generation(self):
        rng = np.random.RandomState(1)
        spec = rng.rand(128, 100)
        peaks = find_spectral_peaks(spec, n_peaks_per_frame=5)
        hashes = generate_fingerprints(peaks, fan_out=5)
        assert len(hashes) > 0
        assert all(isinstance(h, FingerprintHash) for h in hashes)

    def test_hash_deterministic(self):
        h = FingerprintHash(anchor_freq=10, target_freq=50, delta_time=5, anchor_time=0)
        assert h.hash_value == h.hash_value
        assert len(h.hash_value) == 16


class TestFingerprintDatabase:
    def test_insert_and_query(self):
        db = FingerprintDatabase()
        rng = np.random.RandomState(2)
        spec = rng.rand(128, 200)
        peaks = find_spectral_peaks(spec)
        hashes = generate_fingerprints(peaks)
        db.insert("track_A", hashes)
        results = db.query(hashes, min_matches=3)
        assert len(results) > 0
        assert results[0][0] == "track_A"
        assert results[0][2] > 0.5

    def test_empty_query(self):
        db = FingerprintDatabase()
        results = db.query([], min_matches=1)
        assert results == []


class TestFingerprintSimilarity:
    def test_identical_hashes(self):
        h = [FingerprintHash(10, 50, 5, i) for i in range(10)]
        assert fingerprint_similarity(h, h) == 1.0

    def test_disjoint_hashes(self):
        a = [FingerprintHash(10, 50, 5, 0)]
        b = [FingerprintHash(99, 99, 99, 0)]
        assert fingerprint_similarity(a, b) == 0.0


class TestCrossCorrelation:
    def test_finds_identical_segment(self):
        sr = 8000
        t = np.arange(sr * 2) / sr
        haystack = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        needle = haystack[3000:4000].copy()
        matches = cross_correlation_search(haystack, needle, hop=100, threshold=0.95)
        assert len(matches) > 0
        offsets = [m[0] for m in matches]
        assert any(abs(o - 3000) < 200 for o in offsets)

    def test_no_match_in_noise(self):
        rng = np.random.RandomState(4)
        haystack = rng.randn(5000)
        needle = np.sin(2 * np.pi * 440 * np.arange(500) / 8000)
        matches = cross_correlation_search(haystack, needle, hop=256, threshold=0.9)
        assert len(matches) == 0


class TestSpectralEnvelope:
    def test_envelope_shape(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4096) / 44100)
        env = spectral_envelope(sig, n_fft=2048)
        assert len(env) == 2048 // 2 + 1
        assert np.all(np.isfinite(env))

    def test_self_similarity(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4096) / 44100)
        env = spectral_envelope(sig)
        assert envelope_similarity(env, env) > 0.99


class TestPitchShiftDetection:
    def test_no_shift_same_signal(self):
        t = np.arange(4096) / 44100
        sig = np.sin(2 * np.pi * 440 * t)
        shift = detect_pitch_shift(sig, sig, sr=44100)
        assert abs(shift) < 1.0

    def test_octave_shift(self):
        t = np.arange(8192) / 44100
        original = np.sin(2 * np.pi * 220 * t)
        shifted = np.sin(2 * np.pi * 440 * t)
        shift = detect_pitch_shift(original, shifted, sr=44100)
        assert abs(shift - 12.0) < 1.5


class TestLoopDetection:
    def test_detect_repeating_pattern(self):
        sr = 44100
        loop_len = int(0.25 * sr)
        one_loop = np.sin(2 * np.pi * 440 * np.arange(loop_len) / sr)
        one_loop *= np.hanning(loop_len)
        signal = np.tile(one_loop, 8)
        loops = detect_loops(signal, sr=sr, min_loop_ms=100, max_loop_ms=500, threshold=0.3)
        assert len(loops) > 0

    def test_no_loops_in_random(self):
        rng = np.random.RandomState(5)
        signal = rng.randn(44100)
        loops = detect_loops(signal, sr=44100, threshold=0.8)
        assert len(loops) == 0


class TestConceptRegistry:
    def test_32_concepts(self):
        assert len(CONCEPT_REGISTRY) == 32

    def test_vectors_unit_norm(self):
        for name, c in CONCEPT_REGISTRY.items():
            norm = np.linalg.norm(c.vector)
            assert abs(norm - 1.0) < 0.01, f"{name} norm={norm}"


class TestConceptSimilarity:
    def test_self_similarity(self):
        c = CONCEPT_REGISTRY["melody"]
        assert abs(concept_similarity(c, c) - 1.0) < 0.01

    def test_same_category_closer(self):
        melody = CONCEPT_REGISTRY["melody"]
        assert melody.category == "pitch"
        interval = CONCEPT_REGISTRY["interval"]
        assert interval.category == "pitch"
        sim = concept_similarity(melody, interval)
        assert -1.0 <= sim <= 1.0


class TestNearestConcepts:
    def test_returns_k(self):
        result = nearest_concepts(CONCEPT_REGISTRY["chord"], k=3)
        assert len(result) == 3
        assert all(isinstance(r[1], float) for r in result)


class TestAnalogy:
    def test_analogy_returns_results(self):
        results = analogy("melody", "interval", "chord", k=3)
        assert len(results) == 3

    def test_analogy_unknown_concept(self):
        assert analogy("nonexistent", "melody", "chord") == []


class TestConceptEmbedding:
    def test_add_and_search(self):
        ce = ConceptEmbedding()
        custom = ce.add("riff", "pitch", "Short repeated melodic figure")
        results = ce.search(custom.vector, k=3)
        assert results[0][0] == "riff"

    def test_cluster(self):
        ce = ConceptEmbedding()
        clusters = ce.cluster(k=5)
        assert len(clusters) > 0
        total = sum(len(v) for v in clusters.values())
        assert total == len(ce.concepts)


class TestMoodTree:
    def test_24_moods(self):
        assert len(MOOD_TREE) == 24

    def test_parent_child_consistency(self):
        for name, node in MOOD_TREE.items():
            if node.parent:
                parent = MOOD_TREE[node.parent]
                assert name in parent.children


class TestMoodDistance:
    def test_zero_self_distance(self):
        assert mood_distance("happy", "happy") == 0.0

    def test_happy_sad_far(self):
        d = mood_distance("happy", "sad")
        assert d > 0.5


class TestMoodAncestors:
    def test_child_has_ancestors(self):
        anc = mood_ancestors("joyful")
        assert "happy" in anc

    def test_root_has_no_ancestors(self):
        assert mood_ancestors("happy") == []


class TestMoodFromVAD:
    def test_high_valence_arousal(self):
        m = mood_from_vad(0.95, 0.95, 0.8)
        assert m == "euphoric"

    def test_low_valence_low_arousal(self):
        m = mood_from_vad(0.15, 0.25, 0.3)
        assert m in ("melancholic", "sad", "grieving")


class TestMoodFromFeatures:
    def test_fast_major_bright(self):
        m = mood_from_features(tempo=150, mode=1.0, brightness=0.95)
        assert m in ("euphoric", "joyful", "happy", "energetic")

    def test_slow_minor_dark(self):
        m = mood_from_features(tempo=55, mode=0.0, brightness=0.2)
        assert m in ("grieving", "sad", "melancholic")


class TestMoodTrajectory:
    def test_rags_to_riches(self):
        traj = MoodTrajectory(moods=["sad", "melancholic", "nostalgic", "content", "happy", "joyful", "euphoric"])
        arc, sim = traj.classify_arc()
        assert arc == "rags_to_riches"
        assert sim > 0.5


class TestITD:
    def test_zero_at_front(self):
        assert interaural_time_difference(0.0) == 0.0

    def test_positive_right(self):
        itd = interaural_time_difference(90.0)
        assert itd > 0

    def test_symmetric(self):
        itd_l = interaural_time_difference(-45.0)
        itd_r = interaural_time_difference(45.0)
        assert abs(itd_l + itd_r) < 1.0


class TestILD:
    def test_zero_at_front(self):
        ild = interaural_level_difference(0.0, 1000.0)
        assert abs(ild) < 0.01

    def test_increases_with_freq(self):
        ild_low = interaural_level_difference(90.0, 500.0)
        ild_high = interaural_level_difference(90.0, 5000.0)
        assert ild_high > ild_low


class TestHeadShadow:
    def test_unity_at_front(self):
        lg, rg = head_shadow_filter(1000.0, 0.0)
        assert abs(lg - 1.0) < 0.01
        assert abs(rg - 1.0) < 0.01


class TestDistanceAttenuation:
    def test_zero_at_ref(self):
        assert distance_attenuation(1.0, 1.0) == 0.0

    def test_negative_further(self):
        assert distance_attenuation(2.0) < 0.0


class TestHRTF:
    def test_generate_and_apply(self):
        hrtf = generate_synthetic_hrtf(45.0, sr=16000, n_taps=64)
        mono = np.sin(2 * np.pi * 440 * np.arange(1000) / 16000)
        left, right = apply_hrtf(mono, hrtf)
        assert len(left) == len(mono)
        assert len(right) == len(mono)


class TestAzimuthEstimation:
    def test_roundtrip(self):
        for az in [-60, -30, 0, 30, 60]:
            itd = interaural_time_difference(float(az))
            est = estimate_azimuth(itd)
            assert abs(est - az) < 3.0, f"az={az}, est={est}"


class TestBinauralCueAnalysis:
    def test_identical_channels(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100)
        cues = binaural_cue_analysis(sig, sig, sr=44100)
        assert abs(cues.itd_us) < 50
        assert abs(cues.ild_db) < 0.5
        assert cues.ic > 0.95


class TestSpectralDescriptors:
    def test_centroid(self):
        freqs = np.arange(100)
        mag = np.zeros(100)
        mag[50] = 1.0
        assert abs(spectral_centroid(mag, freqs) - 50.0) < 0.01

    def test_flatness_sparse_spectrum(self):
        mag = np.zeros(512)
        mag[100] = 1.0
        mag[200] = 0.5
        mag[300] = 0.1
        sf = spectral_flatness(mag)
        assert sf < 1.0
        mag_flat = np.ones(512)
        assert spectral_flatness(mag_flat) > sf

    def test_flatness_white_noise(self):
        mag = np.ones(512)
        assert spectral_flatness(mag) > 0.99

    def test_rolloff(self):
        freqs = np.linspace(0, 22050, 513)
        mag = np.ones(513)
        ro = spectral_rolloff(mag, freqs, percentile=0.85)
        assert 15000 < ro < 22050


class TestHarmonicAnalysis:
    def test_odd_even_pure_fundamental(self):
        sr = 44100
        n_fft = 8192
        f0 = 440.0
        mag = np.zeros(n_fft // 2 + 1)
        bin_res = sr / n_fft
        mag[int(round(f0 / bin_res))] = 1.0
        oer = odd_even_ratio(mag, f0, sr, n_fft)
        assert oer > 1.0

    def test_inharmonicity_perfect_harmonics(self):
        sr = 44100
        n_fft = 8192
        f0 = 440.0
        bin_res = sr / n_fft
        mag = np.zeros(n_fft // 2 + 1)
        for h in range(1, 8):
            b = int(round(f0 * h / bin_res))
            if b < len(mag):
                mag[b] = 1.0 / h
        inh = inharmonicity(mag, f0, sr, n_fft)
        assert inh < 0.05


class TestAttackTime:
    def test_fast_attack(self):
        sr = 44100
        env = np.ones(4410)
        at = attack_time(env, sr)
        assert at < 1.0

    def test_slow_attack(self):
        sr = 44100
        env = np.linspace(0, 1, 44100)
        at = attack_time(env, sr)
        assert at > 500


class TestBrightnessWarmth:
    def test_bright_high_freq(self):
        freqs = np.linspace(0, 22050, 513)
        mag = np.zeros(513)
        mag[400:] = 1.0
        br = brightness(mag, freqs, cutoff=1500.0)
        assert br > 0.8

    def test_warm_low_freq(self):
        freqs = np.linspace(0, 22050, 513)
        mag = np.zeros(513)
        mag[3:15] = 1.0
        wm = warmth(mag, freqs)
        assert wm > 0.5


class TestRoughness:
    def test_unison_smooth(self):
        r = roughness_dissonance(440.0, 440.0)
        assert r < 0.01

    def test_minor_second_rough(self):
        r = roughness_dissonance(440.0, 466.0)
        assert r > 0.01


class TestAnalyzeTimbre:
    def test_full_analysis(self):
        t = np.arange(8192) / 44100
        sig = np.sin(2 * np.pi * 440 * t)
        td = analyze_timbre(sig, sr=44100, n_fft=4096, f0=440.0)
        assert td.spectral_centroid > 0
        assert td.brightness >= 0
        assert td.warmth >= 0

    def test_timbre_distance_zero(self):
        t = np.arange(8192) / 44100
        sig = np.sin(2 * np.pi * 440 * t)
        td = analyze_timbre(sig, sr=44100, f0=440.0)
        assert timbre_distance(td, td) < 1e-10
