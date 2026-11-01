"""Tests for round 16 deepening: transforms, synthesis, theory, semantics, analysis."""
import numpy as np
import pytest


# ── Wavelet analysis ─────────────────────────────────────────────

from models.music_transforms.wavelet_analysis import (
    morlet_wavelet, cwt_transform, scalogram, scale_to_frequency,
    wavelet_ridge_extraction, instantaneous_frequency_from_cwt,
    cross_wavelet_coherence,
)


class TestWaveletAnalysis:
    def test_morlet_shape(self):
        w = morlet_wavelet(128)
        assert w.shape == (128,)
        assert np.iscomplexobj(w)

    def test_cwt_transform_shape(self):
        sig = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200))
        scales = np.array([4.0, 8.0, 16.0])
        c = cwt_transform(sig, scales)
        assert c.shape == (3, 200)

    def test_scalogram_nonneg(self):
        sig = np.random.default_rng(0).normal(size=100)
        scales = np.array([4.0, 8.0])
        c = cwt_transform(sig, scales)
        s = scalogram(c)
        assert np.all(s >= 0)

    def test_scale_to_frequency(self):
        scales = np.array([10.0, 20.0])
        freqs = scale_to_frequency(scales, sr=1000.0)
        assert freqs[0] > freqs[1]

    def test_ridge_extraction(self):
        sig = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 300))
        scales = np.arange(2, 30, dtype=float)
        c = cwt_transform(sig, scales)
        ridge = wavelet_ridge_extraction(c, scales)
        assert ridge.shape == (300,)
        assert np.all(ridge >= 0) and np.all(ridge < len(scales))

    def test_cross_wavelet_coherence_self(self):
        sig = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
        scales = np.array([4.0, 8.0, 12.0])
        c = cwt_transform(sig, scales)
        coh = cross_wavelet_coherence(c, c)
        assert coh.shape == c.shape
        assert np.all(coh >= 0) and np.all(coh <= 1.01)


# ── Reassigned spectrogram ───────────────────────────────────────

from models.music_transforms.reassigned_spectrogram import (
    compute_stft_derivatives, reassign_time_frequency,
    reassigned_spectrogram, synchrosqueezed_transform,
    extract_ridges, ridge_to_frequency,
)


class TestReassignedSpectrogram:
    def test_stft_derivatives_shape(self):
        sig = np.random.default_rng(1).normal(size=2048)
        S, Sd, St = compute_stft_derivatives(sig, n_fft=256, hop=64)
        assert S.shape == Sd.shape == St.shape
        assert S.shape[0] == 129  # 256/2 + 1

    def test_reassigned_spectrogram_output(self):
        sig = np.sin(2 * np.pi * 100 * np.linspace(0, 0.5, 4000))
        power, f_r, t_r = reassigned_spectrogram(sig, sr=8000.0, n_fft=256, hop=64)
        assert power.shape == f_r.shape == t_r.shape
        assert np.all(power >= 0)

    def test_synchrosqueezed_shape(self):
        sig = np.random.default_rng(2).normal(size=2048)
        sst, freqs = synchrosqueezed_transform(sig, sr=1000.0, n_fft=256, hop=64)
        assert sst.shape[0] == len(freqs)
        assert np.all(sst >= 0)

    def test_extract_ridges(self):
        power = np.random.default_rng(3).uniform(size=(64, 20))
        ridges = extract_ridges(power, n_ridges=2, min_length=5)
        assert len(ridges) == 2
        assert all(len(r) == 20 for r in ridges)

    def test_ridge_to_frequency(self):
        ridge = np.array([5, 10, 15])
        freqs = ridge_to_frequency(ridge, sr=1000.0, n_fft=256)
        assert len(freqs) == 3
        assert freqs[0] < freqs[1] < freqs[2]


# ── Granular synthesis ───────────────────────────────────────────

from models.music_synthesis.granular_synthesis import (
    GrainParams, generate_grain, grain_cloud,
    time_stretch_granular, pitch_shift_granular, granular_freeze,
)


class TestGranularSynthesis:
    def test_generate_grain_length(self):
        source = np.random.default_rng(0).normal(size=4000)
        params = GrainParams(duration_ms=50.0)
        grain = generate_grain(source, sr=8000.0, position=0.25, params=params)
        expected = int(50 * 8000 / 1000)
        assert len(grain) == expected

    def test_grain_cloud_normalized(self):
        source = np.random.default_rng(1).normal(size=8000)
        params = GrainParams(duration_ms=30.0, overlap=0.5)
        out = grain_cloud(source, sr=8000.0, duration=0.5, params=params)
        assert len(out) == 4000
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_time_stretch_longer(self):
        source = np.random.default_rng(2).normal(size=4000)
        stretched = time_stretch_granular(source, sr=8000.0, stretch_factor=2.0)
        assert len(stretched) == pytest.approx(8000, abs=500)

    def test_pitch_shift_same_length(self):
        source = np.random.default_rng(3).normal(size=4000)
        shifted = pitch_shift_granular(source, sr=8000.0, semitones=3.0)
        assert len(shifted) == len(source)

    def test_granular_freeze_duration(self):
        source = np.random.default_rng(4).normal(size=8000)
        frozen = granular_freeze(source, sr=8000.0, position=0.3, duration=0.5)
        assert len(frozen) == 4000


# ── Physical modeling ────────────────────────────────────────────

from models.music_synthesis.physical_modeling import (
    karplus_strong, damped_string, bowed_string,
    flute_waveguide, drum_membrane,
)


class TestPhysicalModeling:
    def test_karplus_strong_length(self):
        out = karplus_strong(440.0, sr=8000.0, duration=0.5)
        assert len(out) == 4000

    def test_karplus_strong_decays(self):
        out = karplus_strong(220.0, sr=8000.0, duration=1.0)
        first_quarter = np.max(np.abs(out[:2000]))
        last_quarter = np.max(np.abs(out[-2000:]))
        assert last_quarter < first_quarter

    def test_damped_string_brightness(self):
        bright = damped_string(330.0, sr=8000.0, duration=0.5, brightness=0.9)
        dull = damped_string(330.0, sr=8000.0, duration=0.5, brightness=0.1)
        assert len(bright) == len(dull) == 4000

    def test_bowed_string_normalized(self):
        out = bowed_string(220.0, sr=8000.0, duration=0.5)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_flute_waveguide_normalized(self):
        out = flute_waveguide(440.0, sr=8000.0, duration=0.5)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_drum_membrane_normalized(self):
        out = drum_membrane(100.0, sr=8000.0, duration=0.5)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


# ── Counterpoint rules ───────────────────────────────────────────

from models.music_theory.counterpoint_rules import (
    classify_motion, MotionKind, interval_class,
    check_parallel_perfects, check_hidden_fifths_octaves,
    first_species_check, counterpoint_score,
    contrary_motion_ratio, melodic_variety,
    CounterpointViolation, Species,
)


class TestCounterpointRules:
    def test_classify_motion_contrary(self):
        assert classify_motion(2, -3) == MotionKind.CONTRARY

    def test_classify_motion_parallel(self):
        assert classify_motion(2, 2) == MotionKind.PARALLEL

    def test_classify_motion_oblique(self):
        assert classify_motion(0, 3) == MotionKind.OBLIQUE

    def test_interval_class(self):
        assert interval_class(0, 7) == 7
        assert interval_class(60, 67) == 7

    def test_parallel_fifths_detected(self):
        cf = [60, 62, 64]
        cp = [67, 69, 71]  # parallel fifths
        violations = check_parallel_perfects(cf, cp)
        assert len(violations) >= 1
        assert violations[0].rule == "parallel_perfect"

    def test_perfect_counterpoint_scores_high(self):
        cf = [60, 62, 64, 65, 67, 65, 64, 60]
        cp = [72, 69, 71, 72, 74, 72, 71, 72]
        violations = first_species_check(cf, cp)
        score = counterpoint_score(violations)
        assert 0.0 <= score <= 1.0

    def test_contrary_motion_ratio(self):
        cf = [60, 62, 64, 65]
        cp = [72, 71, 69, 72]
        ratio = contrary_motion_ratio(cf, cp)
        assert 0.0 <= ratio <= 1.0

    def test_melodic_variety(self):
        line = [60, 62, 64, 62, 60, 62, 64]
        v = melodic_variety(line)
        assert 0.0 <= v <= 1.0


# ── Neo-Riemannian ───────────────────────────────────────────────

from models.music_theory.neo_riemannian import (
    Triad, P, L, R, apply_chain, tonnetz_distance,
    shortest_path, hexatonic_system_for, hexatonic_pole,
    nebenverwandt, slide,
)


class TestNeoRiemannian:
    def test_P_toggles_quality(self):
        c_maj = Triad(0, "major")
        c_min = P(c_maj)
        assert c_min.quality == "minor"
        assert P(c_min).quality == "major"

    def test_L_transform(self):
        c_maj = Triad(0, "major")
        result = L(c_maj)
        assert result.quality == "minor"
        assert result.root % 12 == 4  # E minor

    def test_R_transform(self):
        c_maj = Triad(0, "major")
        result = R(c_maj)
        assert result.quality == "minor"
        assert result.root % 12 == 9  # A minor

    def test_PLR_involutory(self):
        c = Triad(0, "major")
        assert P(P(c)) == c
        assert L(L(c)) == c
        assert R(R(c)) == c

    def test_apply_chain(self):
        c = Triad(0, "major")
        path = apply_chain(c, "PLR")
        assert len(path) == 4
        assert path[0] == c

    def test_tonnetz_distance_self(self):
        c = Triad(0, "major")
        assert tonnetz_distance(c, c) == 0.0

    def test_shortest_path_identity(self):
        c = Triad(0, "major")
        assert shortest_path(c, c) == ""

    def test_shortest_path_P(self):
        c = Triad(0, "major")
        path = shortest_path(c, Triad(0, "minor"))
        assert path == "P"

    def test_hexatonic_system(self):
        c = Triad(0, "major")
        sys = hexatonic_system_for(c)
        assert sys is not None

    def test_hexatonic_pole(self):
        c = Triad(0, "major")
        pole = hexatonic_pole(c)
        assert pole.quality != c.quality or pole.root % 12 != c.root % 12


# ── Narrative arc ────────────────────────────────────────────────

from models.music_semantics.narrative_arc import (
    tension_curve, smooth_tension, classify_arc, ArcType,
    emotional_trajectory, vad_distance, emotional_range,
    emotional_velocity, VADPoint,
    segment_entropy, surprise_curve, structural_surprise_score,
)


class TestNarrativeArc:
    def test_tension_curve_weighted(self):
        h = np.array([0.2, 0.5, 0.8, 0.3])
        d = np.array([0.1, 0.4, 0.7, 0.2])
        tc = tension_curve(h, d)
        assert len(tc) == 4
        assert tc[2] == pytest.approx(0.6 * 0.8 + 0.4 * 0.7)

    def test_smooth_tension(self):
        c = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        s = smooth_tension(c, window=3)
        assert len(s) == len(c)
        assert np.std(s) < np.std(c)

    def test_classify_arc_flat(self):
        c = np.ones(10) * 0.5
        result = classify_arc(c)
        assert result.arc_type == ArcType.FLAT

    def test_classify_arc_ramp(self):
        c = np.linspace(0.0, 1.0, 20)
        result = classify_arc(c)
        assert result.arc_type == ArcType.RAMP

    def test_vad_distance(self):
        a = VADPoint(0.0, 0.0, 0.0)
        b = VADPoint(1.0, 0.0, 0.0)
        assert vad_distance(a, b) == pytest.approx(1.0)

    def test_emotional_range(self):
        traj = [VADPoint(0, 0, 0), VADPoint(1, 1, 1)]
        r = emotional_range(traj)
        assert r > 0

    def test_surprise_curve_length(self):
        features = [np.random.default_rng(i).normal(size=50) for i in range(5)]
        sc = surprise_curve(features)
        assert len(sc) == 4


# ── Cross-modal ──────────────────────────────────────────────────

from models.music_semantics.cross_modal import (
    pitch_to_colour, chord_colour, colour_distance,
    spectral_to_texture, texture_distance,
    articulation_to_shape, dynamics_to_shape, shape_distance,
    cross_modal_coherence, synesthesia_profile,
)


class TestCrossModal:
    def test_pitch_to_colour_hue(self):
        c = pitch_to_colour(0)  # C
        assert c.hue == 0.0
        c2 = pitch_to_colour(6)  # F#
        assert c2.hue == 180.0

    def test_chord_colour_average(self):
        c = chord_colour([0, 4, 7])  # C major
        assert 0 <= c.hue <= 360
        assert 0 <= c.saturation <= 1
        assert 0 <= c.lightness <= 1

    def test_colour_distance_self(self):
        c = pitch_to_colour(0)
        assert colour_distance(c, c) == pytest.approx(0.0)

    def test_texture_from_spectral(self):
        t = spectral_to_texture(2000.0, 0.3, 0.6)
        assert 0 <= t.roughness <= 1
        assert 0 <= t.warmth <= 1

    def test_articulation_shapes(self):
        s = articulation_to_shape("staccato")
        assert s.angularity > 0.5
        l = articulation_to_shape("legato")
        assert l.angularity < 0.5

    def test_cross_modal_coherence_range(self):
        from models.music_semantics.cross_modal import ColourAssociation, TextureAssociation, ShapeAssociation
        col = ColourAssociation(180.0, 0.7, 0.5)
        tex = TextureAssociation(0.3, 0.7, 0.5)
        shp = ShapeAssociation(0.5, 0.5, 0.5)
        coh = cross_modal_coherence(col, tex, shp)
        assert 0.0 <= coh <= 1.0

    def test_synesthesia_profile_keys(self):
        profile = synesthesia_profile(
            [0, 4, 7], velocity=0.8, register=0.6,
            spectral_centroid=2000.0, spectral_flatness=0.3,
            low_energy_ratio=0.5, articulation="marcato"
        )
        assert "colour" in profile
        assert "texture" in profile
        assert "shape" in profile
        assert "coherence" in profile


# ── Tension resolution ───────────────────────────────────────────

from models.music_analysis.tension_resolution import (
    interval_dissonance, chord_dissonance, tonal_distance,
    harmonic_tension, tension_curve as tension_curve_chords,
    resolution_strength, detect_resolutions,
    tension_profile, cadence_strength, prolongational_reduction,
)


class TestTensionResolution:
    def test_unison_no_dissonance(self):
        assert interval_dissonance(0) == 0.0

    def test_tritone_high_dissonance(self):
        assert interval_dissonance(6) > 0.8

    def test_chord_dissonance_triad(self):
        major = chord_dissonance([0, 4, 7])
        cluster = chord_dissonance([0, 1, 2])
        assert cluster > major

    def test_tonal_distance_tonic(self):
        d_tonic = tonal_distance([0], key_root=0)
        d_remote = tonal_distance([6], key_root=0)
        assert d_tonic < d_remote

    def test_tension_curve_length(self):
        chords = [[0, 4, 7], [5, 9, 0], [7, 11, 2], [0, 4, 7]]
        tc = tension_curve_chords(chords)
        assert len(tc) == 4

    def test_resolution_strength(self):
        assert resolution_strength(0.8, 0.2) > 0.5
        assert resolution_strength(0.0, 0.5) == 0.0

    def test_detect_resolutions(self):
        curve = np.array([0.2, 0.8, 0.3, 0.7, 0.1])
        res = detect_resolutions(curve, threshold=0.3)
        assert 2 in res  # 0.8 -> 0.3

    def test_cadence_strength_authentic(self):
        dominant = [7, 11, 2]
        tonic = [0, 4, 7]
        strength = cadence_strength(dominant, tonic, key_root=0)
        assert strength > 0

    def test_prolongational_reduction(self):
        curve = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.4])
        reductions = prolongational_reduction(curve, levels=3)
        assert len(reductions) == 3
        assert len(reductions[0]) == 8


# ── Self-similarity ──────────────────────────────────────────────

from models.music_analysis.self_similarity import (
    cosine_similarity_matrix, euclidean_distance_matrix,
    distance_to_similarity, recurrence_matrix,
    novelty_from_ssm, pick_boundaries,
    label_segments as label_ssm_segments, form_string,
    repetition_ratio, find_repeated_sections,
)


class TestSelfSimilarity:
    def test_cosine_ssm_diagonal(self):
        features = np.random.default_rng(0).normal(size=(10, 5))
        ssm = cosine_similarity_matrix(features)
        assert ssm.shape == (10, 10)
        np.testing.assert_array_almost_equal(np.diag(ssm), 1.0, decimal=5)

    def test_euclidean_symmetric(self):
        features = np.random.default_rng(1).normal(size=(8, 4))
        D = euclidean_distance_matrix(features)
        np.testing.assert_array_almost_equal(D, D.T)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)

    def test_distance_to_similarity(self):
        D = np.array([[0, 1], [1, 0]], dtype=float)
        S = distance_to_similarity(D, sigma=1.0)
        assert S[0, 0] == pytest.approx(1.0)
        assert S[0, 1] < 1.0

    def test_recurrence_matrix(self):
        ssm = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.4], [0.3, 0.4, 1.0]])
        R = recurrence_matrix(ssm, threshold=0.5)
        assert R[0, 1] == 1.0
        assert R[0, 2] == 0.0

    def test_novelty_from_ssm(self):
        n = 32
        features = np.zeros((n, 4))
        features[:16] = np.random.default_rng(0).normal(size=(16, 4))
        features[16:] = np.random.default_rng(1).normal(size=(16, 4)) + 5
        ssm = cosine_similarity_matrix(features)
        novelty = novelty_from_ssm(ssm, kernel_size=8)
        assert len(novelty) == n
        mid = novelty[14:18]
        assert np.max(mid) > np.mean(novelty)

    def test_pick_boundaries(self):
        novelty = np.array([0, 0.1, 0.5, 0.2, 0.1, 0.8, 0.3, 0.1])
        bounds = pick_boundaries(novelty, threshold=0.3, min_gap=2)
        assert 2 in bounds or 5 in bounds

    def test_label_and_form_string(self):
        ssm = np.ones((12, 12))
        ssm[:6, 6:] = 0.1
        ssm[6:, :6] = 0.1
        segments = label_ssm_segments(ssm, [6])
        fs = form_string(segments)
        assert len(fs) == 2
        assert fs[0] != fs[1]

    def test_repetition_ratio(self):
        ssm = np.eye(5)
        r = repetition_ratio(ssm, threshold=0.5)
        assert r == 0.0
        ssm_full = np.ones((5, 5))
        r2 = repetition_ratio(ssm_full, threshold=0.5)
        assert r2 == 1.0
