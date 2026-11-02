"""Tests for Chinese traditional music: pentatonic modes, jianpu, instruments."""
import pytest
import numpy as np

from models.music_ethnomusicology.chinese_music import (
    ChineseMode, MODE_CATALOG, PENTATONIC_MODES, HEPTATONIC_SCALES,
    INSTRUMENT_FAMILIES, JIANPU_MAP,
    classify_mode, pitch_to_jianpu, jianpu_sequence,
    pentatonic_interval_histogram, mode_affinity_vector,
    silk_bamboo_balance, QinTuning,
)


class TestPentatonicModes:
    def test_five_modes(self):
        assert len(PENTATONIC_MODES) == 5

    def test_each_has_five_notes(self):
        for name, intervals in PENTATONIC_MODES.items():
            assert len(intervals) == 5, f"{name} != 5 notes"

    def test_all_start_on_zero(self):
        for name, intervals in PENTATONIC_MODES.items():
            assert intervals[0] == 0, f"{name} doesn't start on 0"


class TestHeptatonicScales:
    def test_three_scales(self):
        assert len(HEPTATONIC_SCALES) == 3

    def test_seven_notes_each(self):
        for name, intervals in HEPTATONIC_SCALES.items():
            assert len(intervals) == 7, f"{name} != 7 notes"


class TestModeCatalog:
    def test_five_modes(self):
        assert len(MODE_CATALOG) == 5

    def test_frozen(self):
        with pytest.raises(AttributeError):
            MODE_CATALOG["gong"].name = "x"

    def test_seasons_present(self):
        seasons = {m.season for m in MODE_CATALOG.values()}
        assert "spring" in seasons
        assert "winter" in seasons


class TestClassifyMode:
    def test_gong_mode(self):
        results = classify_mode([0, 2, 4, 7, 9])
        assert results[0][0] == "gong"

    def test_yu_mode(self):
        results = classify_mode([0, 3, 5, 7, 10])
        assert results[0][0] == "yu"

    def test_partial_match(self):
        results = classify_mode([0, 2, 4])
        assert len(results) > 0
        assert all(0 < s <= 1.0 for _, s in results)

    def test_empty(self):
        assert classify_mode([]) == []


class TestJianpu:
    def test_do(self):
        assert pitch_to_jianpu(60, 60) == "1"

    def test_re(self):
        assert pitch_to_jianpu(62, 60) == "2"

    def test_octave_up(self):
        result = pitch_to_jianpu(72, 60)
        assert "·" in result

    def test_octave_down(self):
        result = pitch_to_jianpu(48, 60)
        assert "," in result

    def test_sequence(self):
        notes = [60, 62, 64, 65, 67]
        seq = jianpu_sequence(notes)
        assert len(seq) == 5
        assert seq[0] == "1"


class TestIntervalHistogram:
    def test_basic(self):
        hist = pentatonic_interval_histogram([0, 2, 4, 7, 9])
        assert hist.shape == (12,)
        assert abs(hist.sum() - 1.0) < 1e-9

    def test_single_note(self):
        hist = pentatonic_interval_histogram([60])
        assert hist.sum() == 0.0


class TestModeAffinityVector:
    def test_shape(self):
        v = mode_affinity_vector([0, 2, 4, 7, 9])
        assert v.shape == (5,)

    def test_gong_highest_for_gong_pitches(self):
        v = mode_affinity_vector([0, 2, 4, 7, 9])
        assert v[0] == max(v)


class TestInstrumentFamilies:
    def test_eight_families(self):
        assert len(INSTRUMENT_FAMILIES) == 8

    def test_silk_has_erhu(self):
        assert "erhu" in INSTRUMENT_FAMILIES["silk"]

    def test_bamboo_has_dizi(self):
        assert "dizi" in INSTRUMENT_FAMILIES["bamboo"]


class TestSilkBambooBalance:
    def test_all_silk(self):
        assert silk_bamboo_balance(["erhu", "pipa"]) == 1.0

    def test_all_bamboo(self):
        assert silk_bamboo_balance(["dizi", "xiao"]) == -1.0

    def test_balanced(self):
        assert silk_bamboo_balance(["erhu", "dizi"]) == 0.0

    def test_empty(self):
        assert silk_bamboo_balance([]) == 0.0

    def test_non_matching(self):
        assert silk_bamboo_balance(["piano", "violin"]) == 0.0


class TestQinTuning:
    def test_standard_seven_strings(self):
        qin = QinTuning()
        assert len(qin.tuning) == 7

    def test_harmonic_octave(self):
        qin = QinTuning()
        h2 = qin.harmonic(1, 2)
        assert abs(h2 - qin.tuning[0] * 2) < 1e-6

    def test_all_harmonics_count(self):
        qin = QinTuning()
        harms = qin.all_harmonics(1, max_node=5)
        assert len(harms) == 5

    def test_invalid_string(self):
        qin = QinTuning()
        assert qin.harmonic(0, 1) == 0.0
        assert qin.harmonic(8, 1) == 0.0

    def test_open_string_intervals(self):
        qin = QinTuning()
        intervals = qin.open_string_intervals()
        assert len(intervals) == 6
        assert all(i > 0 for i in intervals)


class TestPackageExports:
    def test_import_from_package(self):
        from models.music_ethnomusicology import (
            ChineseMode, MODE_CATALOG, classify_chinese_mode,
            QinTuning, CHINESE_INSTRUMENT_FAMILIES,
        )
        assert callable(classify_chinese_mode)
        assert len(MODE_CATALOG) == 5
