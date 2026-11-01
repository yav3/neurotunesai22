"""Tests for expansion round 11: linked data, binaural render, instrument combo, modal interchange, dynamic range."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_metadata.linked_data import (
    Namespace, Triple, LinkedRecord,
    build_track_record, build_artist_record, resolve_same_as,
    triple_count, filter_by_namespace, merge_records,
)
from models.music_spatial.binaural_render import (
    SourcePosition, BinauralOutput,
    itd_samples, ild_factor, render_binaural,
    mix_binaural, crossfeed, perceived_width,
)
from models.music_arrangement.instrument_combination import (
    InstrumentFamily, InstrumentProfile, COMMON_INSTRUMENTS,
    timbral_distance, timbral_compatibility, range_overlap,
    register_complement, combination_score, family_diversity,
    suggest_complement, blend_brightness, total_range,
)
from models.music_improvisation.modal_interchange_improv import (
    Mode, MODE_INTERVALS, mode_pitch_classes, common_tones,
    mode_distance, InterchangeOption, find_interchange_options,
    pivot_note, interchange_tension,
    InterchangeSequence, build_interchange_sequence,
)
from models.music_production.dynamic_range import (
    DynamicRangeStats, CompressorSettings,
    amplitude_to_db, db_to_amplitude, rms_level, peak_level,
    analyze_dynamic_range, compression_curve, gain_reduction,
    apply_compression, limiter_ceiling, loudness_normalize,
    multiband_split_points,
)


class TestLinkedData:
    def test_build_track_record(self):
        rec = build_track_record("t1", "Song", "Artist", 180.0, isrc="USRC17607839")
        assert rec.entity_type == "Track"
        assert "isrc" in rec.external_ids
        assert triple_count(rec) == 5

    def test_build_artist_record(self):
        rec = build_artist_record("a1", "Bach", genres=["classical", "baroque"])
        assert len([t for t in rec.triples if t.predicate == "genre"]) == 2

    def test_resolve_same_as(self):
        r1 = build_track_record("t1", "A", "X", 120.0, isrc="US123")
        r2 = build_track_record("t2", "B", "Y", 130.0, isrc="US123")
        links = resolve_same_as([r1, r2])
        assert any(len(v) == 2 for v in links.values())

    def test_filter_by_namespace(self):
        rec = build_track_record("t1", "S", "A", 100.0)
        dc = filter_by_namespace(rec, Namespace.DUBLIN_CORE)
        assert all(t.namespace == Namespace.DUBLIN_CORE for t in dc)
        assert len(dc) == 2  # title + creator

    def test_merge_records(self):
        r1 = build_track_record("t1", "S", "A", 100.0, isrc="X")
        r2 = build_track_record("t1", "S", "A", 100.0, mbid="MB1")
        merged = merge_records(r1, r2)
        assert "isrc" in merged.external_ids
        assert "musicbrainz" in merged.external_ids

    def test_merge_different_ids_raises(self):
        r1 = build_track_record("t1", "S", "A", 100.0)
        r2 = build_track_record("t2", "S", "A", 100.0)
        with pytest.raises(ValueError):
            merge_records(r1, r2)

    def test_triple_uri(self):
        t = Triple("s", "title", "Song", Namespace.DUBLIN_CORE)
        assert t.uri == "dc:title"


class TestBinauralRender:
    def test_itd_center(self):
        assert itd_samples(0.0, 44100) == 0

    def test_itd_side(self):
        d = itd_samples(90.0, 44100)
        assert d > 0

    def test_ild_center(self):
        gl, gr = ild_factor(0.0)
        assert abs(gl - gr) < 0.1

    def test_render_binaural_shape(self):
        mono = np.random.default_rng(0).standard_normal(1000)
        pos = SourcePosition(azimuth=45.0, distance=2.0)
        out = render_binaural(mono, pos, 44100)
        assert len(out.left) == len(out.right)
        assert len(out.left) >= 1000

    def test_render_binaural_ild(self):
        mono = np.ones(1000) * 0.5
        pos_left = SourcePosition(azimuth=-60.0)
        out = render_binaural(mono, pos_left, 44100)
        assert out.ild_db > 0  # left louder

    def test_mix_binaural(self):
        mono = np.ones(500)
        o1 = render_binaural(mono, SourcePosition(azimuth=-30.0), 44100)
        o2 = render_binaural(mono, SourcePosition(azimuth=30.0), 44100)
        mx = mix_binaural([o1, o2])
        assert mx.duration_s > 0

    def test_crossfeed(self):
        mono = np.ones(500)
        out = render_binaural(mono, SourcePosition(azimuth=-90.0), 44100)
        cf = crossfeed(out, amount=0.3)
        assert np.sum(np.abs(cf.right)) > np.sum(np.abs(out.right)) * 0.5

    def test_perceived_width(self):
        mono = np.random.default_rng(1).standard_normal(2000)
        out = render_binaural(mono, SourcePosition(azimuth=90.0), 44100)
        w = perceived_width(out)
        assert 0.0 <= w <= 1.0


class TestInstrumentCombination:
    def test_timbral_distance_same(self):
        v = COMMON_INSTRUMENTS["violin"]
        assert timbral_distance(v, v) == 0.0

    def test_timbral_compatibility_range(self):
        c = timbral_compatibility(COMMON_INSTRUMENTS["violin"], COMMON_INSTRUMENTS["cello"])
        assert 0 < c <= 1.0

    def test_range_overlap_strings(self):
        overlap = range_overlap(COMMON_INSTRUMENTS["violin"], COMMON_INSTRUMENTS["cello"])
        assert overlap > 0

    def test_register_complement(self):
        rc = register_complement(COMMON_INSTRUMENTS["flute"], COMMON_INSTRUMENTS["cello"])
        assert rc > 0.5  # very different ranges

    def test_combination_score_ensemble(self):
        ensemble = [COMMON_INSTRUMENTS["violin"], COMMON_INSTRUMENTS["cello"],
                     COMMON_INSTRUMENTS["flute"]]
        s = combination_score(ensemble)
        assert 0 < s <= 1.0

    def test_family_diversity(self):
        ensemble = [COMMON_INSTRUMENTS["violin"], COMMON_INSTRUMENTS["flute"],
                     COMMON_INSTRUMENTS["trumpet"]]
        fd = family_diversity(ensemble)
        assert fd == 3 / len(InstrumentFamily)

    def test_suggest_complement(self):
        existing = [COMMON_INSTRUMENTS["violin"]]
        suggestion = suggest_complement(existing)
        assert suggestion is not None
        assert suggestion.name != "violin"

    def test_blend_brightness(self):
        ensemble = [COMMON_INSTRUMENTS["flute"], COMMON_INSTRUMENTS["cello"]]
        b = blend_brightness(ensemble)
        assert abs(b - (0.9 + 0.4) / 2) < 1e-9

    def test_total_range(self):
        ensemble = [COMMON_INSTRUMENTS["cello"], COMMON_INSTRUMENTS["flute"]]
        lo, hi = total_range(ensemble)
        assert lo == 36  # cello low
        assert hi == 96  # flute high


class TestModalInterchange:
    def test_mode_pitch_classes_ionian(self):
        pcs = mode_pitch_classes(0, Mode.IONIAN)
        assert pcs == {0, 2, 4, 5, 7, 9, 11}  # C major

    def test_common_tones_parallel(self):
        ct = common_tones(0, Mode.IONIAN, 0, Mode.MIXOLYDIAN)
        assert len(ct) == 6  # differ only on 7th degree

    def test_mode_distance_same(self):
        assert mode_distance(0, Mode.IONIAN, 0, Mode.IONIAN) == 0

    def test_mode_distance_different(self):
        d = mode_distance(0, Mode.IONIAN, 0, Mode.PHRYGIAN)
        assert d > 0

    def test_find_interchange_options(self):
        opts = find_interchange_options(0, Mode.IONIAN)
        assert len(opts) > 0
        assert opts[0].smoothness >= opts[-1].smoothness  # sorted

    def test_pivot_note_exists(self):
        pn = pivot_note(0, Mode.IONIAN, Mode.DORIAN)
        assert pn is not None
        assert pn in common_tones(0, Mode.IONIAN, 0, Mode.DORIAN)

    def test_interchange_tension_range(self):
        t = interchange_tension(0, Mode.IONIAN, Mode.ALTERED)
        assert 0 < t <= 1.0

    def test_build_interchange_sequence(self):
        seq = build_interchange_sequence(0, Mode.IONIAN, 4, rng=np.random.default_rng(42))
        assert isinstance(seq, InterchangeSequence)
        assert len(seq.modes) == 4
        assert seq.total_beats == 32.0
        assert seq.n_changes == 3

    def test_tension_profile(self):
        seq = build_interchange_sequence(0, Mode.DORIAN, 3, rng=np.random.default_rng(0))
        tp = seq.tension_profile()
        assert len(tp) == 2
        assert all(0 <= t <= 1 for t in tp)


class TestDynamicRange:
    def test_db_roundtrip(self):
        assert abs(db_to_amplitude(amplitude_to_db(0.5)) - 0.5) < 1e-9

    def test_rms_level_sine(self):
        t = np.linspace(0, 1, 44100)
        sine = np.sin(2 * np.pi * 440 * t)
        rms = rms_level(sine)
        assert abs(rms - 1 / np.sqrt(2)) < 0.01

    def test_peak_level(self):
        sig = np.array([0.0, 0.5, -0.8, 0.3])
        assert peak_level(sig) == 0.8

    def test_analyze_dynamic_range(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(44100) * 0.3
        stats = analyze_dynamic_range(sig)
        assert stats.peak_db > stats.rms_db
        assert stats.crest_factor_db > 0

    def test_compression_curve_below_threshold(self):
        s = CompressorSettings(threshold_db=-20.0, ratio=4.0)
        assert compression_curve(-30.0, s) == -30.0  # no compression

    def test_compression_curve_above_threshold(self):
        s = CompressorSettings(threshold_db=-20.0, ratio=4.0)
        out = compression_curve(-10.0, s)
        assert out < -10.0  # compressed
        expected = -20.0 + (-10.0 - (-20.0)) / 4.0
        assert abs(out - expected) < 1e-9

    def test_gain_reduction_negative(self):
        s = CompressorSettings(threshold_db=-20.0, ratio=4.0)
        gr = gain_reduction(-10.0, s)
        assert gr < 0

    def test_apply_compression_reduces_peak(self):
        sig = np.sin(2 * np.pi * 100 * np.linspace(0, 0.1, 4410))
        s = CompressorSettings(threshold_db=-6.0, ratio=8.0)
        compressed = apply_compression(sig, s, 44100)
        assert peak_level(compressed) <= peak_level(sig)

    def test_limiter_ceiling(self):
        sig = np.array([0.0, 0.5, 1.0, -1.0])
        limited = limiter_ceiling(sig, ceiling_db=-6.0)
        assert peak_level(limited) <= db_to_amplitude(-6.0) + 1e-9

    def test_loudness_normalize(self):
        sig = np.random.default_rng(0).standard_normal(4410) * 0.1
        normed = loudness_normalize(sig, target_rms_db=-14.0)
        assert abs(amplitude_to_db(rms_level(normed)) - (-14.0)) < 0.5

    def test_multiband_split_points(self):
        assert multiband_split_points(3) == [200.0, 2000.0]
        assert len(multiband_split_points(6)) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
