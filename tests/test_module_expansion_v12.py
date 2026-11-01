"""Tests for expansion round 12: score simplification, string resonance, reverberation, data augmentation, SSM analysis."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_accessibility.score_simplification import (
    AbilityLevel, SimplificationProfile, LEVEL_PROFILES,
    NoteEvent, difficulty_score, simplify_melody,
    reduce_polyphony, suggest_level, tempo_adjustment,
)
from models.music_instruments.string_resonance import (
    StringState, harmonic_series, harmonic_amplitude,
    sympathetic_coupling, resonance_spectrum, total_sympathetic_energy,
    decay_time, inharmonicity_coefficient, stretched_partial,
    beating_frequency, analyze_resonance, wolf_tone_risk,
)
from models.music_acoustics.reverberation import (
    ReverbParameters, EarlyReflection,
    rt60_sabine, rt60_eyring, mean_free_path, mixing_time_ms,
    generate_early_reflections, late_field_density, energy_decay_curve,
    clarity_c80, definition_d50, center_time_ts,
    analyze_reverb, reverb_tail_level_db,
)
from models.music_pipeline.data_augmentation import (
    AugmentationType, AugmentationConfig, AugmentationPipeline,
    pitch_shift_semitones, time_stretch, add_noise,
    gain_change_db, polarity_invert, low_pass_simple, high_pass_simple,
    random_eq, mixup, spec_augment_mask, apply_pipeline,
)
from models.music_similarity.ssm_analysis import (
    StructuralSegment, self_similarity_matrix, recurrence_matrix,
    diagonal_score, find_repetitions, novelty_curve,
    segment_from_novelty, label_segments, form_string,
    form_similarity, symmetry_score, repetitiveness,
)


class TestScoreSimplification:
    def test_level_profiles_complete(self):
        for level in AbilityLevel:
            assert level in LEVEL_PROFILES

    def test_beginner_profile_restrictive(self):
        p = LEVEL_PROFILES[AbilityLevel.BEGINNER]
        assert p.max_simultaneous_notes == 1
        assert not p.allow_accidentals
        assert not p.allow_syncopation

    def test_professional_profile_permissive(self):
        p = LEVEL_PROFILES[AbilityLevel.PROFESSIONAL]
        assert p.max_simultaneous_notes == 10
        assert p.max_rhythmic_subdivision == 32

    def test_difficulty_score_empty(self):
        assert difficulty_score([], LEVEL_PROFILES[AbilityLevel.BEGINNER]) == 0.0

    def test_difficulty_score_range(self):
        events = [NoteEvent(pitch=60, onset_beat=i, duration_beats=1.0) for i in range(4)]
        score = difficulty_score(events, LEVEL_PROFILES[AbilityLevel.BEGINNER])
        assert 0.0 <= score <= 1.0

    def test_simplify_melody_clamps_interval(self):
        events = [
            NoteEvent(pitch=60, onset_beat=0, duration_beats=1.0),
            NoteEvent(pitch=90, onset_beat=1, duration_beats=1.0),  # 30 semitones apart
        ]
        result = simplify_melody(events, AbilityLevel.BEGINNER)
        interval = abs(result[1].pitch - result[0].pitch)
        assert interval <= LEVEL_PROFILES[AbilityLevel.BEGINNER].max_interval_semitones

    def test_simplify_melody_removes_accidentals(self):
        events = [NoteEvent(pitch=61, onset_beat=0, duration_beats=1.0)]  # C#
        result = simplify_melody(events, AbilityLevel.BEGINNER)
        assert result[0].pitch % 12 not in (1, 3, 6, 8, 10)

    def test_reduce_polyphony(self):
        events = [
            NoteEvent(pitch=60, onset_beat=0, duration_beats=1.0),
            NoteEvent(pitch=64, onset_beat=0, duration_beats=1.0),
            NoteEvent(pitch=67, onset_beat=0, duration_beats=1.0),
        ]
        result = reduce_polyphony(events, max_voices=2)
        onsets = {}
        for e in result:
            onsets.setdefault(e.onset_beat, []).append(e)
        for group in onsets.values():
            assert len(group) <= 2

    def test_suggest_level(self):
        easy = [NoteEvent(pitch=60 + i, onset_beat=i, duration_beats=1.0) for i in range(4)]
        level = suggest_level(easy)
        assert isinstance(level, AbilityLevel)

    def test_tempo_adjustment(self):
        assert tempo_adjustment(200.0, AbilityLevel.BEGINNER) == 100.0
        assert tempo_adjustment(80.0, AbilityLevel.PROFESSIONAL) == 80.0


class TestStringResonance:
    def test_harmonic_series_length(self):
        h = harmonic_series(440.0, 8)
        assert len(h) == 8
        assert h[0] == 440.0
        assert h[1] == 880.0

    def test_harmonic_amplitude_decay(self):
        a1 = harmonic_amplitude(1)
        a2 = harmonic_amplitude(2)
        assert a1 > a2 > 0

    def test_sympathetic_coupling_unison(self):
        s1 = StringState(fundamental_hz=440.0)
        s2 = StringState(fundamental_hz=440.0)
        c = sympathetic_coupling(s1, s2)
        assert c > 0.5

    def test_sympathetic_coupling_distant(self):
        s1 = StringState(fundamental_hz=100.0)
        s2 = StringState(fundamental_hz=777.0)
        c = sympathetic_coupling(s1, s2)
        assert c < 0.5

    def test_resonance_spectrum_shape(self):
        strings = [StringState(fundamental_hz=f) for f in [196.0, 246.94, 329.63, 440.0]]
        spec = resonance_spectrum(strings, excited_index=0)
        assert 0 not in spec
        assert len(spec) == 3

    def test_total_sympathetic_energy(self):
        strings = [StringState(fundamental_hz=f) for f in [196.0, 392.0]]
        energy = total_sympathetic_energy(strings, 0)
        assert energy >= 0

    def test_decay_time_positive(self):
        s = StringState(fundamental_hz=440.0, amplitude=1.0, damping=0.01)
        t = decay_time(s)
        assert t > 0

    def test_inharmonicity_coefficient(self):
        s = StringState(fundamental_hz=440.0)
        b = inharmonicity_coefficient(s)
        assert b >= 0

    def test_stretched_partial(self):
        f1 = 440.0
        f2_ideal = 2 * f1
        f2_stretched = stretched_partial(f1, 2, 0.001)
        assert f2_stretched > f2_ideal

    def test_beating_frequency(self):
        assert beating_frequency(440.0, 442.0) == pytest.approx(2.0)

    def test_analyze_resonance(self):
        strings = [StringState(fundamental_hz=f) for f in [196.0, 246.94, 329.63, 440.0]]
        analysis = analyze_resonance(strings, 0)
        assert analysis.string_count == 4
        assert analysis.strongest_responder is not None

    def test_wolf_tone_risk(self):
        s = StringState(fundamental_hz=200.0)
        risk = wolf_tone_risk(s, body_resonance_hz=400.0)
        assert risk > 0.5


class TestReverberation:
    def test_rt60_sabine_basic(self):
        t = rt60_sabine(200.0, 200.0, 0.3)
        assert 0.3 < t < 2.0

    def test_rt60_eyring_less_than_sabine(self):
        v, s, a = 200.0, 200.0, 0.3
        t_s = rt60_sabine(v, s, a)
        t_e = rt60_eyring(v, s, a)
        assert t_e < t_s

    def test_mean_free_path(self):
        mfp = mean_free_path(200.0, 200.0)
        assert mfp == pytest.approx(4.0)

    def test_mixing_time(self):
        mt = mixing_time_ms(200.0)
        assert mt > 0

    def test_early_reflections_count(self):
        params = ReverbParameters(early_reflection_count=6)
        ers = generate_early_reflections(params)
        assert len(ers) == 6
        assert all(r.delay_ms > 0 for r in ers)

    def test_late_field_density_decay(self):
        rt = 1.0
        d1 = late_field_density(rt, 0.0)
        d2 = late_field_density(rt, 500.0)
        assert d1 > d2

    def test_energy_decay_curve_shape(self):
        edc = energy_decay_curve(1.5, 2000.0)
        assert edc[0] > edc[-1]
        assert len(edc) > 100

    def test_clarity_c80(self):
        c = clarity_c80(0.8, 0.2)
        assert c > 0

    def test_definition_d50(self):
        d = definition_d50(0.6, 1.0)
        assert d == pytest.approx(0.6)

    def test_analyze_reverb(self):
        params = ReverbParameters()
        profile = analyze_reverb(params)
        assert profile.rt60_sabine > 0
        assert len(profile.early_reflections) > 0

    def test_reverb_tail_level(self):
        level = reverb_tail_level_db(2.0, 1000.0)
        assert level == pytest.approx(-30.0)


class TestDataAugmentation:
    def test_pitch_shift_preserves_length(self):
        sig = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        shifted = pitch_shift_semitones(sig, 2.0, 44100)
        assert len(shifted) == len(sig)

    def test_time_stretch_doubles(self):
        sig = np.ones(1000)
        stretched = time_stretch(sig, 2.0)
        assert len(stretched) == 2000

    def test_add_noise_increases_energy(self):
        sig = np.zeros(1000)
        noisy = add_noise(sig, snr_db=20.0, rng=np.random.default_rng(0))
        assert np.sum(noisy ** 2) > 0

    def test_gain_change_db(self):
        sig = np.ones(100)
        louder = gain_change_db(sig, 6.0)
        assert np.mean(np.abs(louder)) > np.mean(np.abs(sig))

    def test_polarity_invert(self):
        sig = np.array([1.0, -0.5, 0.3])
        inv = polarity_invert(sig)
        np.testing.assert_array_almost_equal(inv, [-1.0, 0.5, -0.3])

    def test_low_pass_attenuates_high(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(4096)
        lp = low_pass_simple(sig, 0.1)
        spec_orig = np.abs(np.fft.rfft(sig))
        spec_lp = np.abs(np.fft.rfft(lp))
        high_band = slice(len(spec_orig) // 2, None)
        assert np.mean(spec_lp[high_band]) < np.mean(spec_orig[high_band])

    def test_high_pass_attenuates_low(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(4096)
        hp = high_pass_simple(sig, 0.3)
        spec_orig = np.abs(np.fft.rfft(sig))
        spec_hp = np.abs(np.fft.rfft(hp))
        low_band = slice(0, len(spec_orig) // 4)
        assert np.mean(spec_hp[low_band]) < np.mean(spec_orig[low_band])

    def test_mixup_interpolation(self):
        a = np.ones(100)
        b = np.zeros(100)
        m = mixup(a, b, alpha=0.7)
        assert np.mean(m) == pytest.approx(0.7, abs=0.01)

    def test_spec_augment_mask_shape(self):
        mask = spec_augment_mask(100, 80, rng=np.random.default_rng(0))
        assert mask.shape == (80, 100)
        assert np.sum(mask == 0) > 0

    def test_apply_pipeline(self):
        sig = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 22050))
        pipeline = AugmentationPipeline(steps=[
            AugmentationConfig(AugmentationType.GAIN_CHANGE, probability=1.0, min_param=3.0, max_param=3.0),
        ])
        result = apply_pipeline(sig, pipeline, rng=np.random.default_rng(0))
        assert np.max(np.abs(result)) > np.max(np.abs(sig))


class TestSSMAnalysis:
    def test_ssm_shape(self):
        features = np.random.default_rng(0).standard_normal((20, 5))
        ssm = self_similarity_matrix(features)
        assert ssm.shape == (20, 20)

    def test_ssm_diagonal_ones(self):
        features = np.random.default_rng(0).standard_normal((10, 3))
        ssm = self_similarity_matrix(features)
        np.testing.assert_array_almost_equal(np.diag(ssm), 1.0)

    def test_ssm_symmetric(self):
        features = np.random.default_rng(0).standard_normal((15, 4))
        ssm = self_similarity_matrix(features)
        np.testing.assert_array_almost_equal(ssm, ssm.T)

    def test_recurrence_matrix_binary(self):
        features = np.random.default_rng(0).standard_normal((20, 5))
        ssm = self_similarity_matrix(features)
        R = recurrence_matrix(ssm, k_nearest=5)
        assert np.all((R == 0) | (R == 1))

    def test_repetition_detection(self):
        rng = np.random.default_rng(42)
        pattern = rng.standard_normal((10, 3))
        features = np.vstack([pattern, pattern, pattern])
        ssm = self_similarity_matrix(features)
        reps = find_repetitions(ssm, threshold=0.5, min_lag=5)
        assert len(reps) > 0
        lags = [r[0] for r in reps]
        assert 10 in lags

    def test_novelty_curve_length(self):
        features = np.random.default_rng(0).standard_normal((30, 5))
        ssm = self_similarity_matrix(features)
        nc = novelty_curve(ssm, kernel_size=6)
        assert len(nc) == 30

    def test_segment_from_novelty(self):
        nc = np.zeros(50)
        nc[10] = 1.0
        nc[25] = 0.8
        boundaries = segment_from_novelty(nc, threshold=0.5)
        assert 0 in boundaries
        assert 10 in boundaries

    def test_label_segments(self):
        boundaries = [0, 10, 20]
        segs = label_segments(boundaries, 30)
        assert len(segs) == 3
        assert segs[0].label == "A"

    def test_form_string(self):
        segs = [StructuralSegment("A", 0, 10), StructuralSegment("B", 10, 20),
                StructuralSegment("A", 20, 30)]
        assert form_string(segs) == "A B A"

    def test_form_similarity_identical(self):
        assert form_similarity("A B A", "A B A") == 1.0

    def test_form_similarity_different(self):
        s = form_similarity("A B A", "C D C")
        assert s < 1.0

    def test_symmetry_score(self):
        features = np.random.default_rng(0).standard_normal((15, 4))
        ssm = self_similarity_matrix(features)
        assert symmetry_score(ssm) > 0.99

    def test_repetitiveness_range(self):
        features = np.random.default_rng(0).standard_normal((20, 5))
        ssm = self_similarity_matrix(features)
        r = repetitiveness(ssm)
        assert -1.0 <= r <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
