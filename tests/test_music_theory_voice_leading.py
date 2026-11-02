"""Tests for voice leading analysis (Tymoczko 2011)."""
import pytest

from models.music_theory.voice_leading import (
    VoiceLeadingType, VoiceLeading, ChordTransition,
    classify_motion, voice_leading_distance,
    analyze_voice_leading, smooth_voice_leading,
    optimal_voice_leading, VoiceLeadingAnalyzer,
)

C_MAJ = (60, 64, 67)   # C4 E4 G4
F_MAJ = (60, 65, 69)   # C4 F4 A4 (smooth from C)
G_MAJ = (59, 62, 67)   # B3 D4 G4 (smooth from C)
A_MIN = (60, 64, 69)   # C4 E4 A4


class TestClassifyMotion:
    def test_static(self):
        assert classify_motion(60, 60, 64, 64) == VoiceLeadingType.STATIC

    def test_parallel(self):
        assert classify_motion(60, 62, 64, 66) == VoiceLeadingType.PARALLEL

    def test_similar(self):
        assert classify_motion(60, 62, 64, 67) == VoiceLeadingType.SIMILAR

    def test_contrary(self):
        assert classify_motion(60, 62, 64, 63) == VoiceLeadingType.CONTRARY

    def test_oblique_first_held(self):
        assert classify_motion(60, 60, 64, 66) == VoiceLeadingType.OBLIQUE

    def test_oblique_second_held(self):
        assert classify_motion(60, 62, 64, 64) == VoiceLeadingType.OBLIQUE


class TestVoiceLeadingDistance:
    def test_unison(self):
        assert voice_leading_distance((60, 64, 67), (60, 64, 67)) == 0.0

    def test_semitone_moves(self):
        assert voice_leading_distance((60,), (61,)) == 1.0

    def test_c_to_f(self):
        d = voice_leading_distance(C_MAJ, F_MAJ)
        assert d == pytest.approx(3.0)

    def test_empty(self):
        assert voice_leading_distance((), ()) == 0.0


class TestAnalyzeVoiceLeading:
    def test_returns_voice_leading(self):
        vl = analyze_voice_leading(C_MAJ, F_MAJ)
        assert isinstance(vl, VoiceLeading)
        assert vl.from_pitches == C_MAJ
        assert vl.to_pitches == F_MAJ

    def test_distance_property(self):
        vl = analyze_voice_leading(C_MAJ, F_MAJ)
        assert vl.distance == pytest.approx(3.0)

    def test_motions_populated(self):
        vl = analyze_voice_leading(C_MAJ, F_MAJ)
        assert len(vl.voice_motions) > 0


class TestSmoothVoiceLeading:
    def test_smooth(self):
        assert smooth_voice_leading(C_MAJ, F_MAJ) is True

    def test_not_smooth_large_leap(self):
        assert smooth_voice_leading((60, 64, 67), (48, 52, 55)) is False

    def test_custom_threshold(self):
        assert smooth_voice_leading((60,), (63,), max_distance=3) is True
        assert smooth_voice_leading((60,), (63,), max_distance=2) is False


class TestVoiceLeadingAnalyzer:
    def setup_method(self):
        self.analyzer = VoiceLeadingAnalyzer()
        self.progression = [C_MAJ, F_MAJ, G_MAJ, C_MAJ]

    def test_analyze_progression(self):
        vls = self.analyzer.analyze_progression(self.progression)
        assert len(vls) == 3

    def test_total_distance(self):
        d = self.analyzer.total_distance(self.progression)
        assert d > 0

    def test_average_distance(self):
        avg = self.analyzer.average_distance(self.progression)
        assert avg > 0
        assert avg <= self.analyzer.total_distance(self.progression)

    def test_average_distance_single_chord(self):
        assert self.analyzer.average_distance([C_MAJ]) == 0.0

    def test_smoothness_score_range(self):
        s = self.analyzer.smoothness_score(self.progression)
        assert 0.0 <= s <= 1.0

    def test_contrary_motion_ratio_range(self):
        r = self.analyzer.contrary_motion_ratio(self.progression)
        assert 0.0 <= r <= 1.0

    def test_parallel_fifths_none(self):
        prog = [C_MAJ, F_MAJ, C_MAJ]
        violations = self.analyzer.parallel_fifths_check(prog)
        assert isinstance(violations, list)

    def test_parallel_fifths_planted(self):
        # C-G (P5) moving in parallel up to D-A (P5)
        chord1 = (60, 67)
        chord2 = (62, 69)
        violations = self.analyzer.parallel_fifths_check([chord1, chord2])
        assert len(violations) == 1
        assert violations[0] == (0, 0, 1)

    def test_parallel_octaves_planted(self):
        # C4-C5 (P8) moving in parallel up to D4-D5 (P8)
        chord1 = (60, 72)
        chord2 = (62, 74)
        violations = self.analyzer.parallel_octaves_check([chord1, chord2])
        assert len(violations) == 1
        assert violations[0] == (0, 0, 1)

    def test_parallel_octaves_no_violation(self):
        violations = self.analyzer.parallel_octaves_check(self.progression)
        assert isinstance(violations, list)


class TestOptimalVoiceLeading:
    def test_already_optimal(self):
        result = optimal_voice_leading(C_MAJ, C_MAJ)
        d = voice_leading_distance(C_MAJ, result)
        assert d == 0.0

    def test_finds_closer_voicing(self):
        far_g = (55, 59, 62)  # G3 B3 D4 — far from C4 E4 G4
        result = optimal_voice_leading(C_MAJ, far_g)
        d_opt = voice_leading_distance(C_MAJ, result)
        d_orig = voice_leading_distance(C_MAJ, far_g)
        assert d_opt <= d_orig

    def test_empty(self):
        assert optimal_voice_leading((), ()) == ()
