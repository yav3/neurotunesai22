"""Tests for deep automated feature extraction: spectral, rhythmic, harmonic, timbral."""
import math
import numpy as np
import pytest

from models.music_features.deep_extractor import (
    SpectralFeatureExtractor,
    RhythmFeatureExtractor,
    HarmonicFeatureExtractor,
    TimbreFeatureExtractor,
    UnifiedDeepExtractor,
    _hann_window,
    _stft_frame,
)


SR = 16000
FRAME = 1024


def _sine(freq: float, dur: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


def _noise(dur: float = 0.5, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * dur))


# --- Helpers ---

class TestHelpers:
    def test_hann_window_length(self):
        w = _hann_window(256)
        assert len(w) == 256

    def test_hann_window_endpoints(self):
        w = _hann_window(128)
        assert w[0] < 0.01
        assert w[64] > 0.99

    def test_stft_frame_shape(self):
        sig = _sine(440.0)
        w = _hann_window(FRAME)
        frames = _stft_frame(sig, FRAME, FRAME // 2, w)
        assert frames.shape[1] == FRAME // 2 + 1
        assert frames.shape[0] > 0

    def test_stft_short_signal(self):
        sig = np.array([0.1, 0.2, 0.3])
        w = _hann_window(FRAME)
        frames = _stft_frame(sig, FRAME, FRAME // 2, w)
        assert frames.shape[0] >= 1


# --- SpectralFeatureExtractor ---

class TestSpectralFeatures:
    @pytest.fixture
    def ext(self):
        return SpectralFeatureExtractor()

    @pytest.fixture
    def sine_spec(self):
        sig = _sine(440.0)
        chunk = sig[:FRAME]
        w = _hann_window(FRAME)
        spec = np.abs(np.fft.rfft(chunk * w))
        freqs = np.linspace(0, SR / 2, len(spec))
        return spec, freqs

    def test_centroid_positive(self, ext, sine_spec):
        spec, freqs = sine_spec
        c = ext.spectral_centroid(spec, freqs)
        assert c > 0

    def test_centroid_near_440(self, ext, sine_spec):
        spec, freqs = sine_spec
        c = ext.spectral_centroid(spec, freqs)
        assert 300 < c < 600

    def test_bandwidth_positive(self, ext, sine_spec):
        spec, freqs = sine_spec
        c = ext.spectral_centroid(spec, freqs)
        bw = ext.spectral_bandwidth(spec, freqs, c)
        assert bw >= 0

    def test_rolloff_in_range(self, ext, sine_spec):
        spec, _ = sine_spec
        r = ext.spectral_rolloff(spec)
        assert 0 <= r <= 1.0

    def test_flatness_range(self, ext, sine_spec):
        spec, _ = sine_spec
        f = ext.spectral_flatness(spec)
        assert 0 <= f <= 1.0

    def test_noise_flatness_higher(self, ext):
        sig = _noise()
        chunk = sig[:FRAME]
        w = _hann_window(FRAME)
        spec = np.abs(np.fft.rfft(chunk * w))
        f_noise = ext.spectral_flatness(spec)

        sig2 = _sine(440.0)
        chunk2 = sig2[:FRAME]
        spec2 = np.abs(np.fft.rfft(chunk2 * w))
        f_sine = ext.spectral_flatness(spec2)
        assert f_noise > f_sine

    def test_crest_positive(self, ext, sine_spec):
        spec, _ = sine_spec
        assert ext.spectral_crest(spec) > 0

    def test_slope_finite(self, ext, sine_spec):
        spec, freqs = sine_spec
        s = ext.spectral_slope(spec, freqs)
        assert math.isfinite(s)

    def test_decrease_finite(self, ext, sine_spec):
        spec, _ = sine_spec
        d = ext.spectral_decrease(spec)
        assert math.isfinite(d)

    def test_mfcc_shape(self, ext):
        sig = _sine(440.0)
        mfccs = ext.mfcc(sig, SR, n_mfcc=13, frame_size=FRAME)
        assert mfccs.shape[1] == 13
        assert mfccs.shape[0] > 0

    def test_mfcc_short_signal(self, ext):
        sig = np.array([0.1, 0.2])
        mfccs = ext.mfcc(sig, SR, n_mfcc=13, frame_size=FRAME)
        assert mfccs.shape[1] == 13

    def test_empty_spectrum(self, ext):
        spec = np.zeros(100)
        freqs = np.linspace(0, 8000, 100)
        assert ext.spectral_centroid(spec, freqs) == 0.0
        assert ext.spectral_flatness(spec) == 0.0


# --- RhythmFeatureExtractor ---

class TestRhythmFeatures:
    @pytest.fixture
    def ext(self):
        return RhythmFeatureExtractor()

    def test_onset_strength_shape(self, ext):
        sig = _sine(440.0)
        env = ext.onset_strength(sig, SR, FRAME)
        assert len(env) > 0

    def test_onset_strength_nonneg(self, ext):
        sig = _noise()
        env = ext.onset_strength(sig, SR, FRAME)
        assert all(v >= 0 for v in env)

    def test_tempo_positive(self, ext):
        sig = _sine(440.0, dur=2.0)
        env = ext.onset_strength(sig, SR, FRAME)
        tempo = ext.tempo_estimate(env, SR, FRAME // 2)
        assert tempo > 0

    def test_pulse_clarity_range(self, ext):
        sig = _sine(440.0)
        env = ext.onset_strength(sig, SR, FRAME)
        pc = ext.pulse_clarity(env)
        assert 0 <= pc <= 1.0

    def test_beat_histogram_nonneg(self, ext):
        sig = _sine(440.0, dur=2.0)
        env = ext.onset_strength(sig, SR, FRAME)
        hist = ext.beat_histogram(env, SR, FRAME // 2)
        assert all(h >= 0 for h in hist)

    def test_rhythm_complexity_uniform(self, ext):
        beats = [0.0, 0.5, 1.0, 1.5, 2.0]
        rc = ext.rhythm_complexity(beats)
        assert abs(rc) < 1e-9

    def test_rhythm_complexity_varied(self, ext):
        beats = [0.0, 0.3, 1.0, 1.2, 2.5]
        rc = ext.rhythm_complexity(beats)
        assert rc > 0

    def test_syncopation_index(self, ext):
        onsets = np.array([0.0, 0.5, 1.0])
        grid = np.array([0.0, 0.5, 1.0])
        si = ext.syncopation_index(onsets, grid)
        assert 0 <= si <= 1.0

    def test_groove_similarity(self, ext):
        a = np.array([1.0, 0, 1.0, 0])
        b = np.array([1.0, 0, 1.0, 0])
        assert ext.groove_similarity(a, b) > 0.9

    def test_empty_signal(self, ext):
        env = ext.onset_strength(np.array([]), SR, FRAME)
        assert len(env) >= 0


# --- HarmonicFeatureExtractor ---

class TestHarmonicFeatures:
    @pytest.fixture
    def ext(self):
        return HarmonicFeatureExtractor()

    @pytest.fixture
    def sine_spec(self):
        sig = _sine(440.0)
        chunk = sig[:FRAME]
        w = _hann_window(FRAME)
        return np.abs(np.fft.rfft(chunk * w))

    def test_chroma_length(self, ext, sine_spec):
        chroma = ext.chroma_vector(sine_spec, SR)
        assert len(chroma) == 12

    def test_chroma_nonneg(self, ext, sine_spec):
        chroma = ext.chroma_vector(sine_spec, SR)
        assert all(c >= 0 for c in chroma)

    def test_chroma_sums_to_one(self, ext, sine_spec):
        chroma = ext.chroma_vector(sine_spec, SR)
        total = sum(chroma)
        if total > 0:
            assert abs(total - 1.0) < 0.01

    def test_tonnetz_length(self, ext):
        chroma = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
        tn = ext.tonnetz_features(chroma)
        assert len(tn) == 6

    def test_tonnetz_finite(self, ext):
        chroma = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
        tn = ext.tonnetz_features(chroma)
        assert all(math.isfinite(v) for v in tn)

    def test_key_estimation_returns_pair(self, ext):
        chroma = np.zeros(12)
        chroma[0] = 1.0
        chroma[4] = 0.8
        chroma[7] = 0.6
        name, conf = ext.key_estimation(chroma)
        assert isinstance(name, str)
        assert 0 <= conf <= 1.0

    def test_key_estimation_c_major(self, ext):
        chroma = np.array([1.0, 0, 0.5, 0, 0.8, 0.3, 0, 0.7, 0, 0.3, 0, 0.2])
        name, conf = ext.key_estimation(chroma)
        assert "C" in name or "A" in name

    def test_tpr_in_range(self, ext, sine_spec):
        tpr = ext.tonal_power_ratio(sine_spec, 440.0, SR)
        assert 0 <= tpr <= 1.0

    def test_tpr_sine_high(self, ext, sine_spec):
        tpr = ext.tonal_power_ratio(sine_spec, 440.0, SR)
        assert tpr > 0.3

    def test_empty_chroma(self, ext):
        spec = np.zeros(513)
        chroma = ext.chroma_vector(spec, SR)
        assert len(chroma) == 12


# --- TimbreFeatureExtractor ---

class TestTimbreFeatures:
    @pytest.fixture
    def ext(self):
        return TimbreFeatureExtractor()

    def test_zcr_sine(self, ext):
        sig = _sine(440.0)
        zcr = ext.zero_crossing_rate(sig)
        assert zcr > 0

    def test_zcr_range(self, ext):
        sig = _noise()
        zcr = ext.zero_crossing_rate(sig)
        assert 0 <= zcr <= 1.0

    def test_rms_positive(self, ext):
        sig = _sine(440.0)
        assert ext.rms_energy(sig) > 0

    def test_rms_empty(self, ext):
        assert ext.rms_energy(np.array([])) == 0.0

    def test_attack_time_nonneg(self, ext):
        env = np.abs(_sine(440.0, dur=0.1))
        at = ext.attack_time(env)
        assert at >= 0

    def test_temporal_centroid_range(self, ext):
        env = np.abs(_sine(440.0, dur=0.1))
        tc = ext.temporal_centroid(env)
        assert 0 <= tc <= len(env)

    def test_log_attack_time_nonneg(self, ext):
        env = np.abs(_sine(440.0, dur=0.1))
        lat = ext.log_attack_time(env)
        assert lat >= 0

    def test_spectral_irregularity(self, ext):
        harmonics = np.array([1.0, 0.5, 0.25, 0.125])
        irr = ext.spectral_irregularity(harmonics)
        assert irr >= 0

    def test_tristimulus_sums_to_one(self, ext):
        harmonics = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05])
        t1, t2, t3 = ext.tristimulus(harmonics)
        assert abs(t1 + t2 + t3 - 1.0) < 0.01

    def test_tristimulus_empty(self, ext):
        assert ext.tristimulus(np.array([])) == (0.0, 0.0, 0.0)

    def test_brightness_range(self, ext):
        spec = np.ones(513)
        freqs = np.linspace(0, 8000, 513)
        b = ext.brightness(spec, freqs)
        assert 0 <= b <= 1.0

    def test_brightness_zero_spectrum(self, ext):
        spec = np.zeros(513)
        freqs = np.linspace(0, 8000, 513)
        assert ext.brightness(spec, freqs) == 0.0


# --- UnifiedDeepExtractor ---

class TestUnifiedDeepExtractor:
    @pytest.fixture
    def ude(self):
        return UnifiedDeepExtractor()

    def test_feature_names_count(self, ude):
        names = ude.feature_names()
        assert len(names) == 44

    def test_feature_names_include_mfcc(self, ude):
        names = ude.feature_names()
        assert "mfcc_0" in names
        assert "mfcc_12" in names

    def test_feature_names_include_chroma(self, ude):
        names = ude.feature_names()
        assert "chroma_0" in names
        assert "chroma_11" in names

    def test_extract_all_returns_dict(self, ude):
        sig = _sine(440.0)
        feats = ude.extract_all(sig, SR)
        assert isinstance(feats, dict)

    def test_extract_all_keys(self, ude):
        sig = _sine(440.0)
        feats = ude.extract_all(sig, SR)
        for name in ude.feature_names():
            assert name in feats, f"Missing feature: {name}"

    def test_extract_all_finite(self, ude):
        sig = _sine(440.0)
        feats = ude.extract_all(sig, SR)
        for name, val in feats.items():
            assert math.isfinite(val), f"{name} = {val}"

    def test_extract_all_noise(self, ude):
        sig = _noise()
        feats = ude.extract_all(sig, SR)
        assert len(feats) == 44

    def test_extract_all_short_signal(self, ude):
        sig = np.array([0.1, -0.1, 0.2])
        feats = ude.extract_all(sig, SR)
        assert len(feats) == 44

    def test_feature_matrix_shape(self, ude):
        signals = [_sine(440.0), _sine(880.0), _noise()]
        mat = ude.feature_matrix(signals, SR)
        assert mat.shape == (3, 44)

    def test_feature_matrix_finite(self, ude):
        signals = [_sine(440.0), _noise()]
        mat = ude.feature_matrix(signals, SR)
        assert np.all(np.isfinite(mat))

    def test_normalize_passthrough(self, ude):
        feats = {"a": 1.0, "b": 2.0}
        result = ude.normalize_features(feats)
        assert result == feats

    def test_normalize_zscore(self, ude):
        feats = {"a": 10.0, "b": 20.0}
        stats = {"a": (5.0, 2.5), "b": (15.0, 5.0)}
        result = ude.normalize_features(feats, method="z-score", stats=stats)
        assert abs(result["a"] - 2.0) < 1e-9
        assert abs(result["b"] - 1.0) < 1e-9

    def test_sine_vs_noise_centroid(self, ude):
        sine_feats = ude.extract_all(_sine(440.0), SR)
        noise_feats = ude.extract_all(_noise(), SR)
        assert noise_feats["spectral_centroid"] > sine_feats["spectral_centroid"]


# --- Import from __init__ ---

class TestDeepExtractorExports:
    def test_import_from_package(self):
        from models.music_features import (
            DeepSpectralExtractor,
            DeepRhythmExtractor,
            DeepHarmonicExtractor,
            DeepTimbreExtractor,
            UnifiedDeepExtractor,
        )
        assert callable(DeepSpectralExtractor)
        assert callable(UnifiedDeepExtractor)
