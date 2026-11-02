"""Tests for music_acoustics: psychoacoustics, room acoustics, spatial audio, spectral analysis."""
import numpy as np
import pytest

from models.music_acoustics.psychoacoustics import (
    hz_to_mel, mel_to_hz, hz_to_bark, bark_to_hz, hz_to_erb, erb_to_hz,
    critical_bandwidth, equal_loudness_contour, a_weighting, c_weighting,
    simultaneous_masking, temporal_masking, spl_to_sone, sone_to_phon, phon_to_sone,
)
from models.music_acoustics.room_acoustics import (
    RoomParameters, rt60_sabine, rt60_eyring, mean_free_path, compute_rt60,
    clarity, early_decay_time, schroeder_curve, mixing_time_estimate,
    AbsorptionCoefficients, CONCRETE, CARPET, ACOUSTIC_TILE,
)
from models.music_acoustics.spatial_audio import (
    SPEED_OF_SOUND, HEAD_RADIUS, itd, ild, foa_encode, foa_decode_stereo,
    binaural_pan, distance_attenuation, air_absorption_db,
    constant_power_pan, hrtf_delay_model,
)
from models.music_acoustics.spectral_analysis import (
    stft, istft, mel_filterbank, mel_spectrogram, cqt_frequencies,
    constant_q_transform, chromagram, spectral_centroid, spectral_spread,
    spectral_flatness, spectral_rolloff, spectral_flux,
    compute_spectral_descriptors,
)


# ── Psychoacoustics ──────────────────────────────────────────────────

class TestFrequencyScales:
    def test_mel_roundtrip(self):
        for f in [100.0, 440.0, 1000.0, 8000.0]:
            assert abs(mel_to_hz(hz_to_mel(f)) - f) < 0.01

    def test_bark_roundtrip(self):
        for f in [200.0, 1000.0, 4000.0]:
            recovered = bark_to_hz(hz_to_bark(f))
            assert abs(recovered - f) / f < 0.15  # Bark approx has larger error

    def test_erb_roundtrip(self):
        for f in [100.0, 1000.0, 8000.0]:
            assert abs(erb_to_hz(hz_to_erb(f)) - f) < 0.5

    def test_mel_monotonic(self):
        freqs = np.array([100, 500, 1000, 5000, 10000], dtype=float)
        mels = hz_to_mel(freqs)
        assert np.all(np.diff(mels) > 0)

    def test_critical_bandwidth_positive(self):
        assert critical_bandwidth(1000.0) > 0
        assert critical_bandwidth(100.0) < critical_bandwidth(8000.0)


class TestLoudness:
    def test_equal_loudness_contour_shape(self):
        result = equal_loudness_contour(60.0)
        assert result.shape[1] == 2
        assert result.shape[0] > 10

    def test_equal_loudness_at_1khz(self):
        result = equal_loudness_contour(60.0)
        freqs = result[:, 0]
        spls = result[:, 1]
        idx_1k = np.argmin(np.abs(freqs - 1000.0))
        assert abs(spls[idx_1k] - 60.0) < 1.0

    def test_a_weighting_at_1khz(self):
        w = float(a_weighting(1000.0))
        assert abs(w) < 3.0  # ~0 dB at 1 kHz

    def test_c_weighting_at_1khz(self):
        w = float(c_weighting(1000.0))
        assert abs(w) < 3.0

    def test_a_weighting_low_freq_attenuation(self):
        w_low = float(a_weighting(50.0))
        w_mid = float(a_weighting(1000.0))
        assert w_low < w_mid  # Heavy attenuation at low freq

    def test_sone_phon_roundtrip(self):
        for phon in [50.0, 60.0, 80.0]:
            sone = phon_to_sone(phon)
            recovered = sone_to_phon(sone)
            assert abs(recovered - phon) < 0.01

    def test_spl_to_sone_positive(self):
        assert spl_to_sone(60.0) > 0
        assert spl_to_sone(80.0) > spl_to_sone(60.0)


class TestMasking:
    def test_simultaneous_masking_shape(self):
        probes = np.array([200, 500, 1000, 2000, 4000], dtype=float)
        result = simultaneous_masking(1000.0, 70.0, probes)
        assert result.masked_threshold.shape == probes.shape
        assert result.masker_freq == 1000.0

    def test_masking_peak_near_masker(self):
        probes = np.linspace(200, 8000, 50)
        result = simultaneous_masking(1000.0, 70.0, probes)
        peak_idx = np.argmax(result.masked_threshold)
        peak_freq = probes[peak_idx]
        assert 500 < peak_freq < 2000

    def test_temporal_masking_decay(self):
        t = np.array([0, 10, 50, 100, 200], dtype=float)
        post = temporal_masking(t, 80.0, pre=False)
        assert np.all(np.diff(post) <= 0)  # Monotonically decreasing

    def test_pre_masking_faster_decay(self):
        t = 20.0
        pre = temporal_masking(t, 80.0, pre=True)
        post = temporal_masking(t, 80.0, pre=False)
        assert float(pre) < float(post)  # Pre-masking decays faster


# ── Room Acoustics ───────────────────────────────────────────────────

class TestRT60:
    def test_sabine_formula(self):
        rt = rt60_sabine(200.0, 150.0, 0.3)
        expected = 0.161 * 200.0 / (150.0 * 0.3)
        assert abs(rt - expected) < 0.001

    def test_eyring_less_than_sabine(self):
        s = rt60_sabine(200.0, 150.0, 0.3)
        e = rt60_eyring(200.0, 150.0, 0.3)
        assert e <= s

    def test_zero_absorption_infinite(self):
        assert rt60_sabine(200.0, 150.0, 0.0) == float("inf")
        assert rt60_eyring(200.0, 150.0, 0.0) == float("inf")

    def test_full_absorption_zero_eyring(self):
        assert rt60_eyring(200.0, 150.0, 1.0) == 0.0

    def test_mean_free_path(self):
        mfp = mean_free_path(200.0, 150.0)
        assert abs(mfp - 4.0 * 200.0 / 150.0) < 0.001

    def test_compute_rt60_struct(self):
        room = RoomParameters(200.0, 150.0, 0.3)
        result = compute_rt60(room)
        assert result.rt60_sabine > 0
        assert result.rt60_eyring > 0
        assert result.mean_free_path > 0


class TestClarity:
    def test_clarity_impulse(self):
        sr = 44100
        ir = np.zeros(sr)
        ir[0] = 1.0  # Perfect impulse — all energy is early
        ir[100:200] = 0.001 * np.random.randn(100)
        result = clarity(ir, sr)
        assert result.c50 > 20  # Nearly all energy before 50ms
        assert result.d50 > 0.9

    def test_schroeder_monotonic(self):
        ir = np.random.randn(1000) * np.exp(-np.arange(1000) / 200.0)
        sc = schroeder_curve(ir)
        assert np.all(np.diff(sc) <= 0)

    def test_mixing_time(self):
        mt = mixing_time_estimate(200.0)
        assert mt > 0
        assert mixing_time_estimate(800.0) > mt


class TestAbsorption:
    def test_nrc(self):
        nrc = CARPET.nrc()
        assert 0 < nrc < 1

    def test_concrete_low_absorption(self):
        assert CONCRETE.nrc() < 0.1

    def test_interpolation(self):
        val = ACOUSTIC_TILE.at_freq(750.0)
        assert ACOUSTIC_TILE.freq_500 <= val <= ACOUSTIC_TILE.freq_1000 or \
               ACOUSTIC_TILE.freq_1000 <= val <= ACOUSTIC_TILE.freq_500


# ── Spatial Audio ────────────────────────────────────────────────────

class TestITDILD:
    def test_itd_zero_at_front(self):
        assert abs(itd(0.0)) < 1e-10

    def test_itd_symmetry(self):
        left = itd(45.0)
        right = itd(-45.0)
        assert abs(left + right) < 1e-10

    def test_ild_zero_at_front(self):
        assert abs(ild(0.0, 1000.0)) < 1e-6

    def test_ild_frequency_dependence(self):
        low = ild(90.0, 500.0)
        high = ild(90.0, 4000.0)
        assert high > low  # Higher freq → more shadow


class TestAmbisonics:
    def test_foa_encode_center(self):
        sig = np.ones(100)
        amb = foa_encode(sig, 0.0, 0.0)
        assert amb.W.shape == (100,)
        assert np.allclose(amb.Y, 0.0, atol=1e-10)  # No lateral for 0 azimuth

    def test_foa_decode_stereo_center(self):
        sig = np.ones(100)
        amb = foa_encode(sig, 0.0, 0.0)
        left, right = foa_decode_stereo(amb)
        np.testing.assert_allclose(left, right, atol=1e-10)


class TestBinauralAndPanning:
    def test_binaural_pan_center(self):
        sig = np.ones(1000)
        left, right = binaural_pan(sig, 0.0)
        assert np.sum(left**2) > 0
        assert np.sum(right**2) > 0

    def test_distance_attenuation_inverse(self):
        assert distance_attenuation(1.0) == 1.0
        assert abs(distance_attenuation(2.0) - 0.5) < 0.001
        assert abs(distance_attenuation(10.0) - 0.1) < 0.001

    def test_air_absorption_increases_with_distance(self):
        near = air_absorption_db(1.0, 4000.0)
        far = air_absorption_db(10.0, 4000.0)
        assert far > near

    def test_constant_power_pan_center(self):
        p = constant_power_pan(0.0)
        assert abs(p.left_gain**2 + p.right_gain**2 - 1.0) < 0.01

    def test_constant_power_pan_extremes(self):
        left = constant_power_pan(-1.0)
        right = constant_power_pan(1.0)
        assert left.left_gain > left.right_gain
        assert right.right_gain > right.left_gain

    def test_hrtf_delay_symmetry(self):
        ld, rd = hrtf_delay_model(0.0, 0.0)
        assert abs(ld - rd) < 1e-10
        ld2, rd2 = hrtf_delay_model(45.0, 0.0)
        assert ld2 != rd2


# ── Spectral Analysis ────────────────────────────────────────────────

class TestSTFT:
    def test_stft_shape(self):
        sig = np.random.randn(8000)
        S = stft(sig, n_fft=1024, hop_length=256)
        assert S.shape[0] == 513  # n_fft//2 + 1

    def test_stft_istft_roundtrip(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4000) / 16000)
        S = stft(sig, n_fft=1024, hop_length=256)
        reconstructed = istft(S, hop_length=256)
        min_len = min(len(sig), len(reconstructed))
        corr = np.corrcoef(sig[:min_len], reconstructed[:min_len])[0, 1]
        assert corr > 0.5  # OLA reconstruction with simple windowing


class TestMelFilterbank:
    def test_mel_filterbank_shape(self):
        fb = mel_filterbank(40, 2048, 22050)
        assert fb.shape == (40, 1025)

    def test_mel_spectrogram_shape(self):
        sig = np.random.randn(16000)
        ms = mel_spectrogram(sig, 16000, n_fft=1024, hop_length=256, n_mels=40)
        assert ms.shape[0] == 40


class TestCQT:
    def test_cqt_frequencies_geometric(self):
        freqs = cqt_frequencies(24, f_min=32.7, bins_per_octave=12)
        ratios = freqs[1:] / freqs[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)

    def test_cqt_shape(self):
        sig = np.random.randn(8000)
        C = constant_q_transform(sig, 16000, n_bins=24)
        assert C.shape[0] == 24


class TestChromagram:
    def test_chromagram_shape(self):
        sig = np.random.randn(16000)
        ch = chromagram(sig, 16000)
        assert ch.shape[0] == 12

    def test_chromagram_440hz_peak(self):
        sr = 16000
        t = np.arange(sr) / sr
        sig = np.sin(2 * np.pi * 440 * t)
        ch = chromagram(sig, sr, n_fft=2048, hop_length=512)
        mean_chroma = ch.mean(axis=1)
        peak_class = int(np.argmax(mean_chroma))
        # Implementation: pitch_class = round(12*log2(f/440)) % 12, so A=0
        assert peak_class == 0


class TestSpectralDescriptors:
    def test_centroid_at_peak(self):
        freqs = np.arange(100)
        mag = np.zeros(100)
        mag[50] = 1.0
        c = spectral_centroid(mag, freqs)
        assert abs(c - 50.0) < 0.01

    def test_flatness_uniform(self):
        mag = np.ones(100)
        f = spectral_flatness(mag)
        assert abs(f - 1.0) < 0.01

    def test_flatness_peaked(self):
        mag = np.zeros(100)
        mag[50] = 100.0
        mag += 0.001  # Avoid log(0)
        f = spectral_flatness(mag)
        assert f < 0.1

    def test_rolloff(self):
        freqs = np.arange(100, dtype=float)
        mag = np.ones(100)
        r = spectral_rolloff(mag, freqs, threshold=0.85)
        assert 80 <= r <= 90

    def test_flux_zero_identical(self):
        a = np.ones(100)
        assert spectral_flux(a, a) == 0.0

    def test_flux_positive_increase(self):
        a = np.ones(100)
        b = np.ones(100) * 2
        assert spectral_flux(b, a) > 0

    def test_compute_descriptors(self):
        freqs = np.linspace(0, 8000, 513)
        mag = np.random.rand(513)
        prev = np.random.rand(513)
        desc = compute_spectral_descriptors(mag, freqs, prev)
        assert desc.centroid > 0
        assert desc.spread > 0
        assert 0 <= desc.flatness <= 1
        assert desc.rolloff > 0
        assert desc.flux >= 0
