"""Tests for music_composition: counterpoint, form, orchestration, voice leading, Markov."""
import pytest
from models.music_composition.counterpoint import (
    FirstSpecies, SecondSpecies, is_consonant, contrary_motion_score,
    CONSONANT_INTERVALS, PERFECT_CONSONANCES,
)
from models.music_composition.form import SongForm, FormGenerator, VariationEngine
from models.music_composition.orchestration import (
    InstrumentRange, INSTRUMENTS, Orchestrator,
)
from models.music_composition.voice_leading import (
    VoiceLeader, NeoRiemannian, minimal_voice_leading, voice_leading_smoothness,
)
from models.music_composition.markov_composer import PitchMarkov, RhythmMarkov, MarkovComposer


class TestCounterpoint:
    def test_parallel_fifths(self):
        fs = FirstSpecies()
        v1 = [60, 62, 64]
        v2 = [67, 69, 71]
        violations = fs.check_parallel_fifths(v1, v2)
        assert len(violations) == 2

    def test_no_parallel_fifths(self):
        fs = FirstSpecies()
        v1 = [60, 62, 64]
        v2 = [67, 68, 71]
        violations = fs.check_parallel_fifths(v1, v2)
        assert len(violations) == 0

    def test_parallel_octaves(self):
        fs = FirstSpecies()
        v1 = [60, 62]
        v2 = [72, 74]
        violations = fs.check_parallel_octaves(v1, v2)
        assert len(violations) == 1

    def test_voice_crossing(self):
        fs = FirstSpecies()
        v1 = [60, 70, 60]
        v2 = [72, 65, 72]
        crossings = fs.check_voice_crossing(v1, v2)
        assert 1 in crossings

    def test_score_range(self):
        fs = FirstSpecies()
        cantus = [60, 62, 64, 65, 67]
        cp = [72, 71, 69, 69, 67]
        s = fs.score(cantus, cp)
        assert 0.0 <= s <= 1.0

    def test_score_empty(self):
        fs = FirstSpecies()
        assert fs.score([], []) == 0.0

    def test_consonant_intervals(self):
        assert is_consonant(0)
        assert is_consonant(7)
        assert is_consonant(12)
        assert not is_consonant(1)
        assert not is_consonant(6)

    def test_contrary_motion(self):
        v1 = [60, 62, 60, 62]
        v2 = [72, 70, 72, 70]
        score = contrary_motion_score(v1, v2)
        assert score == pytest.approx(1.0)

    def test_contrary_motion_none(self):
        v1 = [60, 62, 64]
        v2 = [72, 74, 76]
        score = contrary_motion_score(v1, v2)
        assert score == 0.0

    def test_weak_beat_dissonance(self):
        ss = SecondSpecies()
        cantus = [60, 64, 67]
        cp = [72, 73, 74, 75, 76, 77]
        violations = ss.check_weak_beat_dissonance(cantus, cp)
        assert isinstance(violations, list)


class TestForm:
    def test_verse_chorus(self):
        f = SongForm.verse_chorus()
        assert f.name == "verse-chorus"
        assert "verse" in f.sections
        assert "chorus" in f.sections

    def test_aaba(self):
        f = SongForm.aaba()
        assert f.sections == ["A", "A", "B", "A"]

    def test_rondo(self):
        f = SongForm.rondo()
        assert f.sections[0] == "A"
        assert f.sections[2] == "A"
        assert f.sections[4] == "A"

    def test_blues_12bar(self):
        f = SongForm.blues_12bar()
        assert len(f.sections) == 12

    def test_sonata(self):
        f = SongForm.sonata()
        assert "exposition" in f.sections
        assert "recapitulation" in f.sections

    def test_section_count(self):
        f = SongForm.verse_chorus()
        counts = f.section_count()
        assert counts["chorus"] == 3

    def test_form_generator_pop(self):
        fg = FormGenerator()
        sections = fg.generate("pop")
        assert "intro" in sections
        assert "outro" in sections

    def test_form_generator_repeats(self):
        fg = FormGenerator()
        base = fg.generate("binary")
        doubled = fg.generate("binary", repeats=2)
        assert len(doubled) == 2 * len(base)

    def test_form_generator_unknown(self):
        fg = FormGenerator()
        sections = fg.generate("nonexistent")
        assert sections == fg.generate("pop")


class TestVariationEngine:
    def test_transpose(self):
        ve = VariationEngine()
        assert ve.transpose([60, 64, 67], 2) == [62, 66, 69]

    def test_invert(self):
        ve = VariationEngine()
        result = ve.invert([60, 64, 67])
        assert result == [60, 56, 53]

    def test_invert_with_axis(self):
        ve = VariationEngine()
        result = ve.invert([60, 64, 67], axis=64)
        assert result == [68, 64, 61]

    def test_retrograde(self):
        ve = VariationEngine()
        assert ve.retrograde([1, 2, 3]) == [3, 2, 1]

    def test_retrograde_inversion(self):
        ve = VariationEngine()
        pitches = [60, 64, 67]
        ri = ve.retrograde_inversion(pitches)
        inv = ve.invert(pitches)
        assert ri == list(reversed(inv))

    def test_augmentation(self):
        ve = VariationEngine()
        assert ve.augmentation([0.5, 1.0], 2.0) == [1.0, 2.0]

    def test_diminution(self):
        ve = VariationEngine()
        assert ve.diminution([1.0, 2.0], 2.0) == [0.5, 1.0]

    def test_sequence(self):
        ve = VariationEngine()
        result = ve.sequence([60, 64], 2, 3)
        assert result == [60, 64, 62, 66, 64, 68]

    def test_empty_operations(self):
        ve = VariationEngine()
        assert ve.transpose([], 5) == []
        assert ve.invert([]) == []
        assert ve.retrograde([]) == []


class TestOrchestration:
    def test_instrument_ranges(self):
        assert INSTRUMENTS["piano"].min_pitch == 21
        assert INSTRUMENTS["piano"].max_pitch == 108

    def test_fits_range(self):
        orch = Orchestrator()
        assert orch.fits_range(60, INSTRUMENTS["piano"])
        assert not orch.fits_range(10, INSTRUMENTS["piano"])

    def test_clarinet_transposition(self):
        cl = INSTRUMENTS["clarinet"]
        assert cl.transposition == -2

    def test_find_instruments(self):
        orch = Orchestrator()
        insts = orch.find_instruments(60)
        names = [i.name for i in insts]
        assert "piano" in names
        assert "violin" in names

    def test_range_violations(self):
        orch = Orchestrator()
        pitches = [20, 60, 110]
        violations = orch.range_violations(pitches, INSTRUMENTS["piano"])
        assert 0 in violations
        assert 2 in violations
        assert 1 not in violations

    def test_assign_voices(self):
        orch = Orchestrator()
        pitches = [60, 72, 48, 84]
        assignment = orch.assign_voices(pitches)
        total = sum(len(v) for v in assignment.values())
        assert total == 4

    def test_family_balance(self):
        orch = Orchestrator()
        pitches = [60, 72, 48]
        assignment = orch.assign_voices(pitches)
        balance = orch.family_balance(assignment)
        assert isinstance(balance, dict)


class TestVoiceLeading:
    def test_minimal_cost_unison(self):
        vl = minimal_voice_leading([60, 64, 67], [60, 64, 67])
        assert vl.cost == 0
        assert vl.motion == [0, 0, 0]

    def test_minimal_cost_step(self):
        vl = minimal_voice_leading([60, 64, 67], [60, 65, 69])
        assert vl.cost == 3

    def test_smoothness_unison(self):
        s = voice_leading_smoothness([60, 64, 67], [60, 64, 67])
        assert s == 1.0

    def test_smoothness_distant(self):
        s = voice_leading_smoothness([60, 64, 67], [72, 76, 79])
        assert s < 0.8

    def test_common_tones(self):
        vl = VoiceLeader()
        common = vl.common_tones([60, 64, 67], [60, 65, 69])
        assert 0 in common

    def test_parallel_motion(self):
        vl = VoiceLeader()
        assert vl.parallel_motion([2, 2, 2])
        assert not vl.parallel_motion([2, -1, 2])

    def test_contrary_motion(self):
        vl = VoiceLeader()
        assert vl.contrary_motion([2, -1, 3])
        assert not vl.contrary_motion([2, 2, 2])

    def test_unequal_voices(self):
        vl = minimal_voice_leading([60, 64], [60, 64, 67])
        assert isinstance(vl.cost, int)


class TestNeoRiemannian:
    def test_leading_c_major(self):
        nr = NeoRiemannian()
        c_major = [0, 4, 7]
        result = nr.leading(c_major)
        assert len(result) == 3
        assert result != c_major

    def test_parallel_c_major(self):
        nr = NeoRiemannian()
        c_major = [0, 4, 7]
        result = nr.parallel(c_major)
        assert 3 in result

    def test_relative_c_major(self):
        nr = NeoRiemannian()
        c_major = [0, 4, 7]
        result = nr.relative(c_major)
        assert len(result) == 3

    def test_leading_minor(self):
        nr = NeoRiemannian()
        a_minor = [9, 0, 4]
        result = nr.leading(sorted([p % 12 for p in a_minor]))
        assert len(result) == 3


class TestMarkovComposer:
    def test_pitch_markov_train(self):
        pm = PitchMarkov(order=1)
        pm.train([[60, 62, 64, 65, 67, 65, 64, 62, 60]])
        prob = pm.transition_probability([60], 62)
        assert prob > 0.0

    def test_pitch_generate(self):
        pm = PitchMarkov(order=1)
        pm.train([[60, 62, 64, 65, 67, 65, 64, 62]] * 10)
        result = pm.generate([60], 8, rng=__import__("random").Random(42))
        assert len(result) <= 8
        assert result[0] == 60

    def test_rhythm_markov(self):
        rm = RhythmMarkov(order=1)
        rm.train([[1.0, 0.5, 0.5, 1.0, 0.25, 0.25, 0.5]] * 10)
        result = rm.generate([1.0], 6, rng=__import__("random").Random(42))
        assert all(isinstance(d, float) for d in result)

    def test_composer_compose(self):
        mc = MarkovComposer(pitch_order=1, rhythm_order=1)
        mc.train(
            [[60, 62, 64, 65, 67, 65, 64, 62]] * 10,
            [[1.0, 0.5, 0.5, 1.0, 0.25, 0.25, 0.5, 1.0]] * 10,
        )
        result = mc.compose([60], [1.0], length=8, seed=42)
        assert len(result) <= 8
        for pitch, dur in result:
            assert isinstance(pitch, int)
            assert isinstance(dur, float)

    def test_composer_entropy(self):
        mc = MarkovComposer()
        mc.train(
            [[60, 62, 64, 62, 60]] * 5,
            [[1.0, 0.5, 0.5, 1.0, 1.0]] * 5,
        )
        ent = mc.entropy()
        assert "pitch" in ent
        assert "rhythm" in ent
        assert ent["pitch"] >= 0.0

    def test_untrained_generate(self):
        pm = PitchMarkov()
        result = pm.generate([60, 62], 5)
        assert result == [60, 62]

    def test_transition_probability_unknown(self):
        pm = PitchMarkov()
        pm.train([[60, 62, 64]])
        assert pm.transition_probability([99], 60) == 0.0

    def test_deterministic_seed(self):
        mc = MarkovComposer()
        mc.train([[60, 62, 64, 65, 67]] * 20, [[1.0, 0.5, 0.5, 1.0, 1.0]] * 20)
        r1 = mc.compose([60], [1.0], length=10, seed=123)
        r2 = mc.compose([60], [1.0], length=10, seed=123)
        assert r1 == r2
