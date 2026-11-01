"""Tests for expansion round 5: sequencing, noise, onsets, codecs, cover detection."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_playlist.sequencing import (
    MixStrategy, SequenceTrack, PlaylistSequencer,
    camelot_compatible, camelot_distance, tempo_compatibility,
    transition_score, greedy_sequence, tempo_ramp_sequence,
    energy_arc_sequence,
)
from models.music_acoustics.noise_analysis import (
    NoiseProfile, NoiseGate,
    estimate_noise_floor, compute_snr, compute_snr_from_floor,
    noise_profile, spectral_subtraction, a_weighting_snr,
)
from models.music_analysis.onset_detection import (
    OnsetMethod, Onset, OnsetDetector,
    spectral_flux, high_frequency_content, complex_domain_odf,
    energy_odf, phase_deviation_odf,
    adaptive_threshold, pick_onsets, compute_odf,
)
from models.music_fingerprinting.audio_codec import (
    CodecType, CodecArtifact, CodecRobustnessTest,
    simulate_codec, codec_distortion, analyze_codec_artifacts,
    quantize, pre_echo_simulation,
)
from models.music_similarity.cover_detection import (
    CoverDetector, CoverMatch,
    chroma_from_signal, optimal_transposition_index, transpose_chroma,
    cross_recurrence_matrix, qmax_score, dtw_chroma, hpcp,
)


class TestPlaylistSequencing:
    def test_camelot_compatible_same_key(self):
        assert camelot_compatible("5A", "5A")

    def test_camelot_compatible_adjacent(self):
        assert camelot_compatible("5A", "5B")
        assert camelot_compatible("5A", "6A")
        assert camelot_compatible("5A", "4A")

    def test_camelot_incompatible(self):
        assert not camelot_compatible("1A", "7A")

    def test_camelot_distance_same(self):
        assert camelot_distance("3B", "3B") == 0

    def test_camelot_distance_far(self):
        assert camelot_distance("1A", "7A") >= 5

    def test_tempo_compatibility_match(self):
        assert tempo_compatibility(120.0, 120.0) > 0.9

    def test_tempo_compatibility_double(self):
        assert tempo_compatibility(120.0, 60.0) > 0.5

    def test_tempo_compatibility_mismatch(self):
        assert tempo_compatibility(120.0, 95.0) < 0.1

    def test_transition_score_range(self):
        a = SequenceTrack("a", 120.0, "5A", 0.6)
        b = SequenceTrack("b", 122.0, "5B", 0.65)
        s = transition_score(a, b)
        assert 0.0 <= s <= 1.0

    def test_greedy_sequence_length(self):
        tracks = [SequenceTrack(f"t{i}", 120 + i * 2, f"{(i % 12) + 1}A", i / 10)
                  for i in range(8)]
        order = greedy_sequence(tracks)
        assert len(order) == 8
        assert set(order) == set(range(8))

    def test_tempo_ramp_ascending(self):
        tracks = [SequenceTrack("a", 140.0), SequenceTrack("b", 100.0), SequenceTrack("c", 120.0)]
        order = tempo_ramp_sequence(tracks, ascending=True)
        tempos = [tracks[i].tempo_bpm for i in order]
        assert tempos == sorted(tempos)

    def test_energy_arc_sequence_length(self):
        tracks = [SequenceTrack(f"t{i}", 120.0, energy=i / 5) for i in range(6)]
        order = energy_arc_sequence(tracks, peak_position=0.5)
        assert len(order) == 6

    def test_sequencer_harmonic(self):
        seq = PlaylistSequencer(MixStrategy.HARMONIC)
        tracks = [SequenceTrack(f"t{i}", 120.0, f"{(i % 12) + 1}A") for i in range(5)]
        order = seq.sequence(tracks)
        assert len(order) == 5

    def test_sequencer_flow_score(self):
        seq = PlaylistSequencer()
        tracks = [SequenceTrack(f"t{i}", 120.0, "5A", 0.5) for i in range(4)]
        order = seq.sequence(tracks)
        score = seq.total_flow_score(tracks, order)
        assert 0.0 <= score <= 1.0


class TestNoiseAnalysis:
    def test_estimate_noise_floor(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.001, 44100)
        floor = estimate_noise_floor(noise)
        assert floor < -40.0

    def test_compute_snr(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100.0)
        noise = np.random.default_rng(0).normal(0, 0.01, 4410)
        snr = compute_snr(sig, noise)
        assert snr > 20.0

    def test_noise_profile_shape(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.01, 8192)
        prof = noise_profile(noise, n_fft=2048)
        assert isinstance(prof, NoiseProfile)
        assert len(prof.mean_spectrum) == 1025

    def test_noise_profile_flatness(self):
        rng = np.random.default_rng(2)
        white = rng.normal(0, 0.01, 8192)
        prof = noise_profile(white, n_fft=2048)
        assert prof.spectral_flatness > 0.5

    def test_spectral_subtraction_length(self):
        rng = np.random.default_rng(3)
        sig = np.sin(2 * np.pi * 440 * np.arange(8192) / 44100.0)
        noise_seg = rng.normal(0, 0.01, 4096)
        prof = noise_profile(noise_seg, n_fft=2048)
        cleaned = spectral_subtraction(sig + rng.normal(0, 0.01, 8192), prof)
        assert len(cleaned) == len(sig)

    def test_noise_gate_process(self):
        sig = np.zeros(4410)
        sig[1000:2000] = 0.5
        gate = NoiseGate(threshold_db=-20.0)
        out = gate.process(sig)
        assert np.max(np.abs(out[:500])) < np.max(np.abs(out[1000:2000]))

    def test_a_weighting_snr(self):
        sig = np.sin(2 * np.pi * 1000 * np.arange(4096) / 44100.0)
        noise = np.random.default_rng(4).normal(0, 0.01, 4096)
        snr = a_weighting_snr(sig, noise)
        assert snr > 10.0


class TestOnsetDetection:
    def _click_train(self, n_clicks=5, sr=44100.0):
        sig = np.zeros(int(sr * 2))
        for i in range(n_clicks):
            pos = int(i * sr * 0.4)
            if pos + 100 < len(sig):
                sig[pos:pos + 100] = 0.8 * np.random.default_rng(i).normal(0, 1, 100)
        return sig

    def test_spectral_flux_nonempty(self):
        sig = self._click_train()
        odf = spectral_flux(sig)
        assert len(odf) > 0

    def test_hfc_nonempty(self):
        sig = self._click_train()
        odf = high_frequency_content(sig)
        assert len(odf) > 0

    def test_complex_domain_nonempty(self):
        sig = self._click_train()
        odf = complex_domain_odf(sig)
        assert len(odf) > 0

    def test_energy_odf_nonempty(self):
        sig = self._click_train()
        odf = energy_odf(sig)
        assert len(odf) > 0

    def test_adaptive_threshold_length(self):
        odf = np.random.default_rng(0).random(100)
        thresh = adaptive_threshold(odf, window=5)
        assert len(thresh) == 100

    def test_pick_onsets_finds_clicks(self):
        sig = self._click_train(n_clicks=4)
        odf = spectral_flux(sig)
        onsets = pick_onsets(odf, sr=44100.0, hop=512)
        assert len(onsets) >= 2

    def test_onset_detector_class(self):
        det = OnsetDetector(method=OnsetMethod.SPECTRAL_FLUX)
        sig = self._click_train()
        onsets = det.detect(sig)
        assert all(isinstance(o, Onset) for o in onsets)
        assert all(o.method == OnsetMethod.SPECTRAL_FLUX for o in onsets)

    def test_compute_odf_dispatch(self):
        sig = self._click_train()
        for method in OnsetMethod:
            odf = compute_odf(sig, method)
            assert isinstance(odf, np.ndarray)


class TestAudioCodec:
    def test_quantize_range(self):
        sig = np.linspace(-1, 1, 1000)
        q = quantize(sig, bits=8)
        assert np.all(q >= -1.0) and np.all(q <= 1.0)

    def test_quantize_reduces_levels(self):
        sig = np.linspace(-1, 1, 10000)
        q = quantize(sig, bits=4)
        unique = len(np.unique(q))
        assert unique <= 2 ** 4 + 2

    def test_simulate_codec_pcm_identity(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100.0)
        coded = simulate_codec(sig, CodecType.PCM)
        np.testing.assert_array_equal(sig, coded)

    def test_simulate_codec_mp3_distortion(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100.0)
        coded = simulate_codec(sig, CodecType.MP3_128)
        assert not np.array_equal(sig, coded)

    def test_codec_distortion_value(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100.0)
        coded = simulate_codec(sig, CodecType.MP3_128)
        dist = codec_distortion(sig, coded)
        assert dist < 0.0

    def test_analyze_codec_artifacts(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100.0)
        art = analyze_codec_artifacts(sig, CodecType.AAC_128)
        assert isinstance(art, CodecArtifact)
        assert art.codec == CodecType.AAC_128

    def test_pre_echo_simulation(self):
        sig = np.zeros(4608)
        sig[2304:] = 0.8
        result = pre_echo_simulation(sig, block_size=1152)
        assert len(result) == len(sig)


class TestCoverDetection:
    def _make_chroma(self, key_shift=0, n_frames=20):
        rng = np.random.default_rng(42)
        base = rng.random((n_frames, 12))
        base = np.roll(base, key_shift, axis=1)
        norms = np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
        return base / norms

    def test_oti_identity(self):
        c = self._make_chroma()
        assert optimal_transposition_index(c, c) == 0

    def test_oti_shifted(self):
        c = self._make_chroma()
        c_shifted = transpose_chroma(c, 3)
        oti = optimal_transposition_index(c, c_shifted)
        assert oti == 3 or oti == 9

    def test_transpose_chroma_shape(self):
        c = self._make_chroma(n_frames=10)
        t = transpose_chroma(c, 5)
        assert t.shape == c.shape

    def test_cross_recurrence_matrix_shape(self):
        a = self._make_chroma(n_frames=10)
        b = self._make_chroma(n_frames=15)
        crp = cross_recurrence_matrix(a, b, threshold=0.5)
        assert crp.shape == (10, 15)

    def test_qmax_self(self):
        c = self._make_chroma(n_frames=20)
        crp = cross_recurrence_matrix(c, c, threshold=0.99)
        q = qmax_score(crp)
        assert q > 0.5

    def test_dtw_self_zero(self):
        c = self._make_chroma(n_frames=10)
        d = dtw_chroma(c, c)
        assert d < 0.1

    def test_cover_detector_compare(self):
        c = self._make_chroma(n_frames=15)
        det = CoverDetector()
        match = det.compare(c, c, "q", "r")
        assert isinstance(match, CoverMatch)
        assert match.score > 0.3

    def test_cover_detector_is_cover_self(self):
        c = self._make_chroma(n_frames=15)
        det = CoverDetector(qmax_threshold=0.1)
        assert det.is_cover(c, c)

    def test_hpcp_shape(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100.0)
        h = hpcp(sig, sr=44100.0, n_fft=4096, hop=2048)
        assert h.shape[1] == 12
        assert h.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
