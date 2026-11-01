"""Tests for the music_generation module."""
from __future__ import annotations

import pytest
import numpy as np

from models.music_generation.waveform import (
    Oscillator, WaveShape, ADSREnvelope, WaveformSynthesizer, NoiseGenerator,
)
from models.music_generation.melody import (
    MelodyConstraints, MarkovMelodyGenerator, ContourMelodyGenerator, MelodyMutator,
)
from models.music_generation.harmony import (
    VoiceLeadingRules, ChordProgressionGenerator, HarmonySuggester, COMMON_PROGRESSIONS,
)
from models.music_generation.arrangement import (
    InstrumentRange, ArrangementTemplate, ArrangementEngine, DynamicsEngine, STANDARD_INSTRUMENTS,
)
from models.music_generation.structure import (
    SongSection, SongStructureTemplate, SongStructureGenerator, SectionTransition,
    SECTION_ENERGY,
)


# --- Waveform ---

class TestOscillator:
    def test_sine_shape(self):
        osc = Oscillator(440, 1.0, 0.0, WaveShape.SINE)
        w = osc.generate(0.01, 22050)
        assert w.dtype == np.float32
        assert len(w) == 220

    def test_amplitude(self):
        osc = Oscillator(440, 0.5, 0.0, WaveShape.SINE)
        w = osc.generate(0.1, 22050)
        assert np.max(np.abs(w)) <= 0.5 + 1e-6

    def test_saw_range(self):
        osc = Oscillator(100, 1.0, 0.0, WaveShape.SAW)
        w = osc.generate(0.05, 22050)
        assert np.max(w) <= 1.0 + 1e-6
        assert np.min(w) >= -1.0 - 1e-6

    def test_square_values(self):
        osc = Oscillator(100, 1.0, 0.0, WaveShape.SQUARE)
        w = osc.generate(0.05, 22050)
        unique = set(np.unique(w).tolist())
        assert unique <= {-1.0, 0.0, 1.0}

    def test_triangle_range(self):
        osc = Oscillator(100, 1.0, 0.0, WaveShape.TRIANGLE)
        w = osc.generate(0.05, 22050)
        assert np.max(w) <= 1.0 + 1e-6
        assert np.min(w) >= -1.0 - 1e-6


class TestADSREnvelope:
    def test_starts_at_zero(self):
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1)
        sig = np.ones(22050, dtype=np.float32)
        out = env.apply(sig, 22050)
        assert abs(out[0]) < 0.01

    def test_sustain_level(self):
        env = ADSREnvelope(0.01, 0.01, 0.5, 0.01)
        sig = np.ones(22050, dtype=np.float32)
        out = env.apply(sig, 22050)
        mid = len(out) // 2
        assert abs(out[mid] - 0.5) < 0.1

    def test_ends_near_zero(self):
        env = ADSREnvelope(0.01, 0.01, 0.5, 0.1)
        sig = np.ones(4410, dtype=np.float32)
        out = env.apply(sig, 22050)
        assert abs(out[-1]) < 0.15


class TestWaveformSynthesizer:
    def test_empty(self):
        s = WaveformSynthesizer()
        w = s.synthesize(0.1)
        assert np.allclose(w, 0)

    def test_additive(self):
        s = WaveformSynthesizer()
        s.add_oscillator(Oscillator(440))
        s.add_oscillator(Oscillator(880, 0.5))
        w = s.synthesize(0.1, normalize=True)
        assert np.max(np.abs(w)) <= 1.0 + 1e-6

    def test_clear(self):
        s = WaveformSynthesizer()
        s.add_oscillator(Oscillator(440))
        s.clear()
        w = s.synthesize(0.1)
        assert np.allclose(w, 0)


class TestNoiseGenerator:
    def test_white_length(self):
        n = NoiseGenerator.white(1000, rng=np.random.default_rng(0))
        assert len(n) == 1000

    def test_pink_normalized(self):
        n = NoiseGenerator.pink(2000, rng=np.random.default_rng(0))
        assert np.max(np.abs(n)) <= 1.0 + 1e-6

    def test_brown_normalized(self):
        n = NoiseGenerator.brown(2000, rng=np.random.default_rng(0))
        assert np.max(np.abs(n)) <= 1.0 + 1e-6

    def test_white_is_float32(self):
        assert NoiseGenerator.white(100).dtype == np.float32


# --- Melody ---

class TestMarkovMelodyGenerator:
    def test_untrained_fallback(self):
        gen = MarkovMelodyGenerator()
        m = gen.generate(8, seed=42)
        assert len(m) == 8

    def test_trained_output(self):
        gen = MarkovMelodyGenerator(order=1)
        gen.train([[60, 62, 64, 65, 67, 65, 64, 62, 60]] * 10)
        m = gen.generate(10, seed=1)
        assert len(m) == 10

    def test_transition_count(self):
        gen = MarkovMelodyGenerator(order=1)
        gen.train([[60, 62, 64]])
        assert gen.transition_count == 2

    def test_deterministic_with_seed(self):
        gen = MarkovMelodyGenerator()
        m1 = gen.generate(10, seed=42)
        m2 = gen.generate(10, seed=42)
        assert m1 == m2


class TestContourMelodyGenerator:
    def test_ascending(self):
        gen = ContourMelodyGenerator()
        m = gen.generate("ascending", 8, seed=0)
        assert m[-1] >= m[0]

    def test_descending(self):
        gen = ContourMelodyGenerator()
        m = gen.generate("descending", 8, seed=0)
        assert m[-1] <= m[0]

    def test_arch(self):
        gen = ContourMelodyGenerator()
        m = gen.generate("arch", 10, seed=0)
        mid = len(m) // 2
        assert m[mid] >= m[0] or m[mid] >= m[-1]

    def test_flat(self):
        gen = ContourMelodyGenerator()
        m = gen.generate("flat", 8, seed=0)
        spread = max(m) - min(m)
        assert spread < 24


class TestMelodyMutator:
    def test_transpose(self):
        assert MelodyMutator.transpose([60, 64, 67], 2) == [62, 66, 69]

    def test_invert(self):
        inv = MelodyMutator.invert([60, 64, 67])
        assert inv == [60, 56, 53]

    def test_retrograde(self):
        assert MelodyMutator.retrograde([60, 64, 67]) == [67, 64, 60]

    def test_augment(self):
        aug = MelodyMutator.augment([60, 64, 67], 2.0)
        assert aug == [60, 68, 74]

    def test_diminish(self):
        dim = MelodyMutator.diminish([60, 64, 67], 0.5)
        assert dim == [60, 62, 63]

    def test_empty_melody(self):
        assert MelodyMutator.transpose([], 5) == []
        assert MelodyMutator.invert([]) == []
        assert MelodyMutator.retrograde([]) == []
        assert MelodyMutator.augment([]) == []


# --- Harmony ---

class TestChordProgressionGenerator:
    def test_from_template(self):
        gen = ChordProgressionGenerator(root=0)
        prog = gen.generate_from_template("I-IV-V-I")
        assert len(prog) == 4

    def test_random(self):
        gen = ChordProgressionGenerator(root=0)
        prog = gen.generate_random(8, seed=0)
        assert len(prog) == 8
        assert all(0 <= p < 12 for p in prog)

    def test_available_templates(self):
        gen = ChordProgressionGenerator()
        templates = gen.available_templates()
        assert "I-IV-V-I" in templates
        assert "12-bar-blues" in templates

    def test_twelve_bar_blues(self):
        gen = ChordProgressionGenerator(root=0)
        prog = gen.generate_from_template("12-bar-blues")
        assert len(prog) == 12


class TestVoiceLeadingRules:
    def test_parallel_fifths(self):
        a = [60, 62]
        b = [67, 69]
        assert VoiceLeadingRules.has_parallel_fifths(a, b)

    def test_no_parallel_fifths(self):
        a = [60, 62]
        b = [64, 69]
        assert not VoiceLeadingRules.has_parallel_fifths(a, b)

    def test_voice_crossing(self):
        voices = [[72, 60], [60, 72]]
        assert VoiceLeadingRules.has_voice_crossing(voices)

    def test_smooth_cost(self):
        cost = VoiceLeadingRules.smooth_voice_leading_cost([60, 64, 67], [62, 65, 69])
        assert cost == 5.0


class TestHarmonySuggester:
    def test_basic_suggestion(self):
        hs = HarmonySuggester()
        chords = hs.suggest_chords([60, 64, 67, 72], key_root=0, beats_per_chord=4)
        assert len(chords) == 1
        assert "root_degree" in chords[0]
        assert "quality" in chords[0]

    def test_multiple_segments(self):
        hs = HarmonySuggester()
        melody = list(range(60, 76))
        chords = hs.suggest_chords(melody, beats_per_chord=4)
        assert len(chords) == 4

    def test_empty_melody(self):
        assert HarmonySuggester().suggest_chords([]) == []


# --- Arrangement ---

class TestInstrumentRange:
    def test_contains(self):
        ir = InstrumentRange("Piano", 0, 21, 108)
        assert ir.contains(60)
        assert not ir.contains(10)

    def test_clamp(self):
        ir = InstrumentRange("Bass", 33, 28, 60)
        assert ir.clamp(20) == 32
        assert ir.clamp(70) == 58
        assert ir.clamp(40) == 40


class TestArrangementEngine:
    def test_assign_parts(self):
        engine = ArrangementEngine()
        tmpl = ArrangementTemplate.pop_band()
        melody = [60, 64, 67, 72, 67, 64, 60, 60]
        chords = [0, 5, 7, 0, 0, 5, 7, 0]
        parts = engine.assign_parts(melody, chords, tmpl)
        assert "synth_lead" in parts
        assert "bass" in parts
        assert len(parts["synth_lead"]) == 8

    def test_orchestra_template(self):
        tmpl = ArrangementTemplate.orchestra()
        assert "flute" in tmpl.roles

    def test_electronic_template(self):
        tmpl = ArrangementTemplate.electronic()
        assert "drums" in tmpl.roles


class TestDynamicsEngine:
    def test_crescendo(self):
        c = DynamicsEngine.crescendo(10)
        assert c[0] < c[-1]
        assert len(c) == 10

    def test_diminuendo(self):
        d = DynamicsEngine.diminuendo(10)
        assert d[0] > d[-1]

    def test_sforzando(self):
        sf = DynamicsEngine.sforzando(10, accent_pos=3, accent_level=1.0, base=0.5)
        assert sf[3] == 1.0
        assert sf[0] == 0.5

    def test_apply_dynamics(self):
        vels = [100] * 5
        curve = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        result = DynamicsEngine.apply_dynamics(vels, curve)
        assert result[0] == 50
        assert result[-1] == 90


# --- Structure ---

class TestSongStructureTemplate:
    def test_verse_chorus(self):
        t = SongStructureTemplate.verse_chorus()
        assert t.sections[0] == SongSection.INTRO
        assert t.sections[-1] == SongSection.OUTRO
        assert SongSection.CHORUS in t.sections

    def test_aaba(self):
        t = SongStructureTemplate.aaba()
        assert len(t.sections) == 4

    def test_total_bars(self):
        t = SongStructureTemplate.verse_chorus()
        assert t.total_bars() > 0

    def test_energy_profile(self):
        t = SongStructureTemplate.verse_chorus()
        ep = t.energy_profile()
        assert len(ep) == len(t.sections)
        assert all(0 <= e <= 1 for e in ep)


class TestSongStructureGenerator:
    def test_generate(self):
        gen = SongStructureGenerator()
        t = gen.generate(target_duration_s=210, bpm=120, seed=42)
        assert t.sections[0] == SongSection.INTRO
        assert t.sections[-1] == SongSection.OUTRO

    def test_short_song(self):
        gen = SongStructureGenerator()
        t = gen.generate(target_duration_s=30, bpm=120, seed=0)
        assert len(t.sections) >= 2

    def test_deterministic(self):
        gen = SongStructureGenerator()
        t1 = gen.generate(seed=99)
        t2 = gen.generate(seed=99)
        assert [s.value for s in t1.sections] == [s.value for s in t2.sections]

    def test_plan_transitions(self):
        gen = SongStructureGenerator()
        t = gen.generate(seed=1)
        trans = gen.plan_transitions(t)
        assert len(trans) == len(t.sections) - 1


class TestSectionTransition:
    def test_direct(self):
        tr = SectionTransition.direct(SongSection.VERSE, SongSection.CHORUS)
        assert tr.transition_type == "direct"

    def test_modulation(self):
        tr = SectionTransition.modulation(SongSection.CHORUS, SongSection.BRIDGE, 2)
        assert tr.key_change == 2

    def test_crossfade(self):
        tr = SectionTransition.crossfade(SongSection.VERSE, SongSection.CHORUS, 2)
        assert tr.crossfade_bars == 2
