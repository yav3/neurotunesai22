"""Tests for music data validation and quality."""
import numpy as np
import pytest

from models.music_data.validation import (
    validate_audio_features,
    validate_pitch_sequence,
    validate_onset_times,
    completeness_score,
    feature_matrix_quality,
    detect_duplicates,
    detect_outliers_iqr,
)


class TestAudioFeatures:
    def test_valid_features(self):
        f = {"tempo_bpm": 120.0, "rms_energy": 0.3}
        r = validate_audio_features(f)
        assert r.valid is True

    def test_out_of_range(self):
        f = {"tempo_bpm": 500.0}
        r = validate_audio_features(f)
        assert r.valid is False
        assert any("tempo_bpm" in e for e in r.errors)

    def test_nan_feature(self):
        f = {"rms_energy": float("nan")}
        r = validate_audio_features(f)
        assert r.valid is False

    def test_unknown_keys_ignored(self):
        f = {"custom_feature": 999.0}
        r = validate_audio_features(f)
        assert r.valid is True


class TestPitchSequence:
    def test_valid_sequence(self):
        notes = np.array([60, 62, 64, 65, 67])
        r = validate_pitch_sequence(notes)
        assert r.valid is True

    def test_out_of_range_warning(self):
        notes = np.array([60, 200])
        r = validate_pitch_sequence(notes)
        assert len(r.warnings) > 0

    def test_large_interval_warning(self):
        notes = np.array([30, 80])
        r = validate_pitch_sequence(notes)
        assert any("2 octaves" in w for w in r.warnings)

    def test_empty(self):
        r = validate_pitch_sequence(np.array([]))
        assert r.valid is True
        assert len(r.warnings) == 1


class TestOnsetTimes:
    def test_monotonic(self):
        t = np.array([0.0, 0.5, 1.0, 1.5])
        r = validate_onset_times(t)
        assert r.valid is True

    def test_non_monotonic(self):
        t = np.array([0.0, 1.0, 0.5])
        r = validate_onset_times(t)
        assert r.valid is False

    def test_very_short_ioi(self):
        t = np.array([0.0, 0.01, 1.0])
        r = validate_onset_times(t)
        assert any("20ms" in w for w in r.warnings)


class TestCompleteness:
    def test_all_present(self):
        data = {"a": 1, "b": 2, "c": 3}
        assert completeness_score(data, ["a", "b", "c"]) == 1.0

    def test_partial(self):
        data = {"a": 1}
        assert completeness_score(data, ["a", "b"]) == 0.5

    def test_none_value(self):
        data = {"a": None, "b": 2}
        assert completeness_score(data, ["a", "b"]) == 0.5

    def test_empty_required(self):
        assert completeness_score({}, []) == 1.0


class TestFeatureMatrixQuality:
    def test_clean_matrix(self):
        X = np.random.randn(50, 3)
        q = feature_matrix_quality(X)
        assert q["nan_fraction"] == 0.0
        assert q["inf_fraction"] == 0.0

    def test_nan_matrix(self):
        X = np.array([[1.0, float("nan")], [2.0, 3.0]])
        q = feature_matrix_quality(X)
        assert q["nan_fraction"] > 0

    def test_constant_column(self):
        X = np.ones((10, 2))
        X[:, 1] = np.arange(10)
        q = feature_matrix_quality(X)
        assert q["variance_ratio"] == pytest.approx(0.0, abs=1e-6)


class TestDuplicates:
    def test_no_dupes(self):
        assert detect_duplicates(["a", "b", "c"]) == []

    def test_has_dupes(self):
        dupes = detect_duplicates(["a", "b", "a", "c", "b"])
        assert "a" in dupes
        assert "b" in dupes


class TestOutliers:
    def test_no_outliers(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = detect_outliers_iqr(v)
        assert not np.any(mask)

    def test_clear_outlier(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        mask = detect_outliers_iqr(v)
        assert mask[-1] is np.True_

    def test_short_array(self):
        v = np.array([1.0, 2.0])
        mask = detect_outliers_iqr(v)
        assert not np.any(mask)
