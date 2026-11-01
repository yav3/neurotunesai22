"""Tests for expanded ethnomusicology modules: celtic, klezmer, gamelan."""
import numpy as np
import pytest
from models.music_ethnomusicology.gamelan import (
    ScaleType, Pathet, IramaLevel, ColotomyStructure, KotekanPattern,
    SLENDRO_CENTS, PELOG_CENTS, COLOTOMY,
    gamelan_interval_cents, pathet_classification, colotomy_structure,
    kotekan_pattern, balungan_density, irama_level, garap_elaboration,
)
from models.music_ethnomusicology.celtic_music import (
    CelticMode, TuneType, OrnamentType,
    classify_tune_type, classify_celtic_mode, generate_ornament,
    session_set_order, tune_key_histogram,
)
from models.music_ethnomusicology.klezmer import (
    KlezmerMode, DanceType,
    classify_klezmer_mode, has_augmented_second, generate_krekhts,
    classify_dance, freygish_scale, mode_tension,
)


class TestGamelanExpanded:
    def test_slendro_cents(self):
        assert len(SLENDRO_CENTS) == 5
        assert SLENDRO_CENTS[0] == 0

    def test_pelog_cents(self):
        assert len(PELOG_CENTS) == 7
        assert PELOG_CENTS[0] == 0
        assert PELOG_CENTS[-1] == 800

    def test_scale_type_enum(self):
        assert ScaleType.SLENDRO.value == "slendro"
        assert ScaleType.PELOG.value == "pelog"

    def test_irama_level_enum(self):
        assert IramaLevel.LANCAR.value < IramaLevel.WILED.value

    def test_colotomy_dict(self):
        assert "lancaran" in COLOTOMY
        assert "ketawang" in COLOTOMY
        assert isinstance(COLOTOMY["lancaran"], ColotomyStructure)

    def test_gamelan_interval_cents(self):
        c = gamelan_interval_cents(0, 1, ScaleType.SLENDRO)
        assert c == pytest.approx(240.0)

    def test_pathet_classification(self):
        melody = [0, 1, 2, 3, 4, 0, 1, 2]
        p = pathet_classification(melody, ScaleType.SLENDRO)
        assert isinstance(p, Pathet)

    def test_colotomy_structure_lancaran(self):
        cs = colotomy_structure(16, "lancaran")
        assert isinstance(cs, dict)
        assert "gong" in cs

    def test_kotekan_pattern(self):
        melody = [1, 2, 3, 4, 5, 1, 2, 3]
        kp = kotekan_pattern(melody, 5)
        assert isinstance(kp, KotekanPattern)
        assert len(kp.polos) == len(melody)
        assert len(kp.sangsih) == len(melody)

    def test_balungan_density(self):
        melody = [1, 0, 2, 0, 3, 0, 4, 0]
        d = balungan_density(melody)
        assert 0.0 <= d <= 1.0

    def test_irama_level(self):
        assert irama_level(1.0) == IramaLevel.LANCAR
        assert irama_level(2.0) == IramaLevel.TANGGUNG

    def test_garap_elaboration(self):
        balungan = [1, 2, 3, 4]
        elab = garap_elaboration(balungan, IramaLevel.TANGGUNG, 5)
        assert len(elab) >= len(balungan)


class TestCelticMusic:
    def test_classify_tune_type_reel(self):
        tt = classify_tune_type(120, 4, 4)
        assert tt == TuneType.REEL

    def test_classify_tune_type_jig(self):
        tt = classify_tune_type(120, 6, 8)
        assert tt == TuneType.JIG

    def test_classify_celtic_mode_dorian(self):
        intervals = [2, 1, 2, 2, 2, 1, 2]
        mode = classify_celtic_mode(intervals)
        assert mode == CelticMode.DORIAN

    def test_classify_celtic_mode_mixolydian(self):
        intervals = [2, 2, 1, 2, 2, 1, 2]
        mode = classify_celtic_mode(intervals)
        assert mode == CelticMode.MIXOLYDIAN

    def test_generate_ornament(self):
        orn = generate_ornament(60, OrnamentType.CUT)
        assert isinstance(orn, list)
        assert len(orn) > 0

    def test_session_set_order(self):
        tunes = [("reel1", TuneType.REEL), ("jig1", TuneType.JIG), ("reel2", TuneType.REEL)]
        order = session_set_order(tunes)
        assert len(order) == 3

    def test_tune_key_histogram(self):
        keys = ["D", "G", "D", "A", "D"]
        hist = tune_key_histogram(keys)
        assert hist["D"] == 3
        assert hist["G"] == 1


class TestKlezmer:
    def test_freygish_scale(self):
        scale = freygish_scale(60)
        assert scale[0] == 60
        assert len(scale) == 8

    def test_has_augmented_second(self):
        intervals = [1, 3, 1, 2, 1, 2, 2]
        assert has_augmented_second(intervals) is True
        assert has_augmented_second([2, 2, 1, 2, 2, 2, 1]) is False

    def test_classify_klezmer_mode_freygish(self):
        intervals = [1, 3, 1, 2, 1, 2, 2]
        mode = classify_klezmer_mode(intervals)
        assert mode == KlezmerMode.FREYGISH

    def test_generate_krekhts(self):
        kr = generate_krekhts(67)
        assert isinstance(kr, list)
        assert len(kr) > 0

    def test_classify_dance_bulgar(self):
        d = classify_dance(130, 4, 4)
        assert d == DanceType.BULGAR

    def test_classify_dance_hora(self):
        d = classify_dance(80, 3, 8)
        assert d == DanceType.HORA

    def test_mode_tension(self):
        intervals = [1, 3, 1, 2, 1, 2, 2]
        t = mode_tension(intervals)
        assert isinstance(t, float)
        assert t > 0.0
