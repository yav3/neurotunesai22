"""Deep tests for music_metadata: fingerprinting, tag inference, normalization."""
import numpy as np
import pytest

from models.music_metadata.fingerprint import (
    Landmark, MatchResult,
    find_spectral_peaks, generate_landmarks, hash_landmark,
    build_fingerprint_db, query_fingerprint,
)
from models.music_metadata.tag_inference import (
    TagPrediction,
    infer_genre_tags, infer_mood_tags, infer_instrument_tags,
    infer_era_tags, combine_tag_predictions,
)
from models.music_metadata.normalization import (
    normalize_key_signature, normalize_tempo, normalize_duration,
    validate_metadata_schema, merge_metadata,
)


# ===== Fingerprinting =====

class TestFindSpectralPeaks:
    def test_single_peak(self):
        spec = np.zeros((5, 20))
        spec[2, 10] = 5.0
        peaks = find_spectral_peaks(spec, threshold=1.0)
        assert (2, 10) in peaks

    def test_no_peaks_below_threshold(self):
        spec = np.ones((5, 20)) * 0.1
        peaks = find_spectral_peaks(spec, threshold=1.0)
        assert len(peaks) == 0

    def test_max_peaks_per_frame(self):
        spec = np.zeros((1, 50))
        for i in range(2, 48, 4):
            spec[0, i] = float(i)
        peaks = find_spectral_peaks(spec, threshold=0.0, max_peaks_per_frame=3)
        assert len(peaks) <= 3

    def test_multiple_frames(self):
        spec = np.zeros((10, 30))
        spec[3, 15] = 2.0
        spec[7, 20] = 3.0
        peaks = find_spectral_peaks(spec, threshold=1.0)
        assert len(peaks) == 2


class TestGenerateLandmarks:
    def test_basic_landmarks(self):
        peaks = [(0, 10), (2, 15), (5, 20)]
        lms = generate_landmarks(peaks, fan_value=5, max_dt=10)
        assert len(lms) > 0
        assert all(isinstance(lm, Landmark) for lm in lms)

    def test_dt_positive(self):
        peaks = [(0, 10), (3, 20), (7, 30)]
        lms = generate_landmarks(peaks, fan_value=10, max_dt=50)
        for lm in lms:
            assert lm.dt > 0

    def test_max_dt_respected(self):
        peaks = [(0, 10), (100, 20)]
        lms = generate_landmarks(peaks, max_dt=50)
        assert len(lms) == 0

    def test_fan_value_limit(self):
        peaks = [(0, 5)] + [(i, 10 + i) for i in range(1, 30)]
        lms = generate_landmarks(peaks, fan_value=3, max_dt=100)
        from_anchor_0 = [lm for lm in lms if lm.time1 == 0]
        assert len(from_anchor_0) <= 3


class TestHashLandmark:
    def test_deterministic(self):
        lm = Landmark(time1=0, freq1=100, time2=5, freq2=200)
        assert hash_landmark(lm) == hash_landmark(lm)

    def test_different_landmarks_differ(self):
        lm1 = Landmark(time1=0, freq1=100, time2=5, freq2=200)
        lm2 = Landmark(time1=0, freq1=100, time2=5, freq2=201)
        assert hash_landmark(lm1) != hash_landmark(lm2)

    def test_hash_is_int(self):
        lm = Landmark(time1=1, freq1=50, time2=3, freq2=75)
        assert isinstance(hash_landmark(lm), int)


class TestFingerprintDB:
    def _make_landmarks(self, offset=0):
        return [
            Landmark(time1=offset, freq1=100, time2=offset + 3, freq2=200),
            Landmark(time1=offset + 1, freq1=150, time2=offset + 4, freq2=250),
            Landmark(time1=offset + 2, freq1=120, time2=offset + 5, freq2=180),
        ]

    def test_build_and_query(self):
        db = build_fingerprint_db({"track_a": self._make_landmarks(0)})
        results = query_fingerprint(self._make_landmarks(0), db, min_matches=2)
        assert len(results) >= 1
        assert results[0].track_id == "track_a"

    def test_no_match(self):
        db = build_fingerprint_db({"track_a": self._make_landmarks(0)})
        query = [Landmark(time1=0, freq1=999, time2=5, freq2=998)]
        results = query_fingerprint(query, db, min_matches=1)
        assert len(results) == 0

    def test_confidence_range(self):
        db = build_fingerprint_db({"t": self._make_landmarks(0)})
        results = query_fingerprint(self._make_landmarks(0), db, min_matches=1)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_multiple_tracks(self):
        db = build_fingerprint_db({
            "a": self._make_landmarks(0),
            "b": self._make_landmarks(100),
        })
        results = query_fingerprint(self._make_landmarks(0), db, min_matches=2)
        track_ids = {r.track_id for r in results}
        assert "a" in track_ids


# ===== Tag Inference =====

class TestInferGenreTags:
    def test_returns_predictions(self):
        tags = infer_genre_tags(
            {"centroid": 1200, "spread": 800},
            {"tempo": 90},
        )
        assert len(tags) > 0
        assert all(isinstance(t, TagPrediction) for t in tags)

    def test_classical_profile(self):
        tags = infer_genre_tags(
            {"centroid": 1200, "spread": 800},
            {"tempo": 90},
        )
        names = [t.tag for t in tags]
        assert "classical" in names

    def test_electronic_profile(self):
        tags = infer_genre_tags(
            {"centroid": 3500, "spread": 2500},
            {"tempo": 128},
        )
        names = [t.tag for t in tags]
        assert "electronic" in names

    def test_max_five_results(self):
        tags = infer_genre_tags({"centroid": 2000, "spread": 1500}, {"tempo": 110})
        assert len(tags) <= 5


class TestInferMoodTags:
    def test_happy(self):
        tags = infer_mood_tags(valence=0.8, arousal=0.7, energy=0.8)
        names = [t.tag for t in tags]
        assert "happy" in names or "energetic" in names

    def test_sad(self):
        tags = infer_mood_tags(valence=0.1, arousal=0.1, energy=0.1)
        names = [t.tag for t in tags]
        assert "sad" in names

    def test_confidence_bounded(self):
        tags = infer_mood_tags(valence=0.5, arousal=0.5, energy=0.5)
        for t in tags:
            assert 0.0 <= t.confidence <= 1.0


class TestInferInstrumentTags:
    def test_returns_predictions(self):
        mfcc = np.array([10.0, -5.0, 3.0, 1.0, 0.5])
        tags = infer_instrument_tags(spectral_centroid=2000, mfcc_means=mfcc, zcr=0.05)
        assert len(tags) > 0

    def test_low_centroid_suggests_bass(self):
        mfcc = np.array([15.0, -3.0, 2.0, 1.0])
        tags = infer_instrument_tags(spectral_centroid=400, mfcc_means=mfcc, zcr=0.02)
        names = [t.tag for t in tags]
        assert "bass" in names


class TestInferEraTags:
    def test_modern(self):
        tags = infer_era_tags({"loudness": -6, "bandwidth": 5000, "stereo_width": 0.8})
        names = [t.tag for t in tags]
        assert any("2010" in n or "2000" in n for n in names)

    def test_vintage(self):
        tags = infer_era_tags({"loudness": -30, "bandwidth": 1500, "stereo_width": 0.1})
        names = [t.tag for t in tags]
        assert any("1970" in n or "pre-1970" in n for n in names)


class TestCombineTagPredictions:
    def test_merges_duplicates(self):
        a = [TagPrediction("rock", 0.8), TagPrediction("pop", 0.5)]
        b = [TagPrediction("rock", 0.6)]
        merged = combine_tag_predictions([a, b], min_confidence=0.3)
        rock = [t for t in merged if t.tag == "rock"]
        assert len(rock) == 1
        assert 0.6 < rock[0].confidence < 0.8

    def test_filters_low_confidence(self):
        a = [TagPrediction("rare", 0.1)]
        merged = combine_tag_predictions([a], min_confidence=0.3)
        assert all(t.tag != "rare" for t in merged)


# ===== Normalization =====

class TestNormalizeKeySignature:
    def test_basic(self):
        assert normalize_key_signature("c major") == "C major"

    def test_minor(self):
        assert normalize_key_signature("eb minor") == "Eb minor"

    def test_sharp_unicode(self):
        assert normalize_key_signature("F♯ minor") == "F# minor"

    def test_flat_unicode(self):
        assert normalize_key_signature("B♭ major") == "Bb major"

    def test_short_mode(self):
        assert normalize_key_signature("Am") == "A minor"

    def test_unparsable_returns_input(self):
        assert normalize_key_signature("XYZ") == "XYZ"


class TestNormalizeTempo:
    def test_normal_range(self):
        assert normalize_tempo(120.0) == 120.0

    def test_halves_fast_low_confidence(self):
        assert normalize_tempo(240.0, confidence=0.3) == pytest.approx(120.0)

    def test_doubles_slow_low_confidence(self):
        assert normalize_tempo(40.0, confidence=0.3) == pytest.approx(80.0)

    def test_clamps_upper(self):
        assert normalize_tempo(500.0, confidence=0.9) == 300.0

    def test_zero_returns_zero(self):
        assert normalize_tempo(0.0) == 0.0


class TestNormalizeDuration:
    def test_from_seconds(self):
        assert normalize_duration(seconds=3.5) == pytest.approx(3.5)

    def test_from_samples(self):
        assert normalize_duration(sr=44100, n_samples=44100) == pytest.approx(1.0)

    def test_negative_clamped(self):
        assert normalize_duration(seconds=-1.0) == 0.0

    def test_no_args(self):
        assert normalize_duration() == 0.0


class TestValidateMetadataSchema:
    def test_all_present(self):
        meta = {"title": "Song", "artist": "Band"}
        errors = validate_metadata_schema(meta, ["title", "artist"])
        assert len(errors) == 0

    def test_missing_field(self):
        meta = {"title": "Song"}
        errors = validate_metadata_schema(meta, ["title", "artist"])
        assert "missing:artist" in errors

    def test_null_field(self):
        meta = {"title": None}
        errors = validate_metadata_schema(meta, ["title"])
        assert "null:title" in errors

    def test_empty_string(self):
        meta = {"title": "  "}
        errors = validate_metadata_schema(meta, ["title"])
        assert "empty:title" in errors


class TestMergeMetadata:
    def test_primary_wins(self):
        result = merge_metadata({"a": 1}, {"a": 2})
        assert result["a"] == 1

    def test_secondary_wins(self):
        result = merge_metadata({"a": 1}, {"a": 2}, conflict_strategy="secondary_wins")
        assert result["a"] == 2

    def test_longest_string(self):
        result = merge_metadata(
            {"title": "Hi"}, {"title": "Hello World"},
            conflict_strategy="longest",
        )
        assert result["title"] == "Hello World"

    def test_fills_gaps(self):
        result = merge_metadata({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}


# ===== Package exports =====

class TestMetadataExports:
    def test_fingerprint_exports(self):
        import models.music_metadata as mm
        assert hasattr(mm, "Landmark")
        assert hasattr(mm, "find_spectral_peaks")
        assert hasattr(mm, "query_fingerprint")

    def test_tag_inference_exports(self):
        import models.music_metadata as mm
        assert hasattr(mm, "TagPrediction")
        assert hasattr(mm, "infer_genre_tags")
        assert hasattr(mm, "combine_tag_predictions")

    def test_normalization_exports(self):
        import models.music_metadata as mm
        assert hasattr(mm, "normalize_key_signature")
        assert hasattr(mm, "validate_metadata_schema")
        assert hasattr(mm, "merge_metadata_fields")
