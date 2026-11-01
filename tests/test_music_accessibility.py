"""Tests for the music_accessibility module."""
import pytest
import numpy as np

from models.music_accessibility.hearing_profile import (
    HearingLossType, HearingProfile, STANDARD_AUDIOGRAM_FREQUENCIES,
    frequency_response_compensation, tinnitus_notch_filter,
    speech_intelligibility_index, comfortable_listening_level,
)
from models.music_accessibility.adaptive_interface import (
    AccessibilityMode, InputModality, AdaptiveConfig,
    adapt_tempo, simplify_interface, map_input_to_midi,
    latency_tolerance, recommended_font_scale, recommended_color_scheme,
)
from models.music_accessibility.haptic_music import (
    HapticChannel, HAPTIC_LAYOUTS, rhythm_haptic, melody_contour_haptic,
)
from models.music_accessibility.visual_notation import (
    BrailleMusic, pitch_to_braille, duration_to_braille, sequence_to_braille,
    pitch_to_color, ShapeNotation, SimplifiedStaff, simplify_notation,
)
from models.music_accessibility.audio_description import (
    describe_pitch, describe_interval, describe_rhythm,
    describe_dynamics, describe_chord, describe_progression,
    generate_audio_guide,
)


class TestHearingProfile:
    def test_normal_hearing(self):
        p = HearingProfile(loss_type=HearingLossType.NORMAL)
        assert p.average_loss() == 0.0
        assert p.severity() == "normal"

    def test_mild_loss(self):
        p = HearingProfile(
            loss_type=HearingLossType.SENSORINEURAL,
            audiogram_db={250: 20, 500: 25, 1000: 30, 2000: 35},
        )
        assert p.severity() == "mild"

    def test_severe_loss(self):
        p = HearingProfile(
            loss_type=HearingLossType.SENSORINEURAL,
            audiogram_db={f: 80 for f in STANDARD_AUDIOGRAM_FREQUENCIES},
        )
        assert p.severity() == "severe"

    def test_compensation_gains(self):
        p = HearingProfile(
            loss_type=HearingLossType.SENSORINEURAL,
            audiogram_db={250: 10, 500: 20, 1000: 30, 2000: 40, 4000: 50},
        )
        gains = frequency_response_compensation(p)
        assert len(gains) == 10
        assert gains[-1] > gains[0]

    def test_comfortable_level(self):
        normal = HearingProfile(loss_type=HearingLossType.NORMAL)
        impaired = HearingProfile(
            loss_type=HearingLossType.SENSORINEURAL,
            audiogram_db={1000: 40},
        )
        assert comfortable_listening_level(impaired) > comfortable_listening_level(normal)

    def test_tinnitus_filter(self):
        h, freqs = tinnitus_notch_filter(4000.0)
        assert len(h) == 256
        assert len(freqs) > 0

    def test_sii_equal_signal_noise(self):
        sig = np.ones(10) * 60
        noise = np.ones(10) * 60
        p = HearingProfile(loss_type=HearingLossType.NORMAL)
        sii = speech_intelligibility_index(p, sig, noise)
        assert 0.0 <= sii <= 1.0


class TestAdaptiveInterface:
    def test_adapt_tempo_full_capability(self):
        assert adapt_tempo(120.0, 1.0) == 120.0

    def test_adapt_tempo_reduced(self):
        assert adapt_tempo(120.0, 0.5) == 60.0

    def test_simplify_interface(self):
        assert simplify_interface(20, 0.5) == 10
        assert simplify_interface(20, 0.1) >= 1

    def test_map_input_breath(self):
        cfg = AdaptiveConfig(mode=AccessibilityMode.MOTOR)
        midi = map_input_to_midi(0.5, InputModality.BREATH, cfg)
        assert 48 <= midi <= 96

    def test_map_input_switch(self):
        cfg = AdaptiveConfig(mode=AccessibilityMode.MOTOR)
        midi = map_input_to_midi(0.5, InputModality.SWITCH, cfg)
        assert 60 <= midi <= 72

    def test_latency_tolerance(self):
        assert latency_tolerance(AccessibilityMode.AUDITORY) < latency_tolerance(AccessibilityMode.COGNITIVE)

    def test_font_scale_visual(self):
        assert recommended_font_scale(AccessibilityMode.VISUAL) > 1.0

    def test_color_scheme_visual(self):
        scheme = recommended_color_scheme(AccessibilityMode.VISUAL)
        assert scheme["background"] == "#000000"


class TestHapticMusic:
    def test_rhythm_haptic(self):
        signal = rhythm_haptic([0.0, 0.5, 1.0], [0.8, 0.6, 0.9], 2.0)
        assert len(signal) == 500
        assert signal[0] > 0

    def test_melody_contour(self):
        signal = melody_contour_haptic([60, 67, 72], [0.5, 0.5, 0.5])
        assert len(signal) > 0

    def test_haptic_layouts(self):
        assert "wrist" in HAPTIC_LAYOUTS
        assert HAPTIC_LAYOUTS["wrist"][0] < HAPTIC_LAYOUTS["wrist"][1]


class TestVisualNotation:
    def test_pitch_to_braille(self):
        b = pitch_to_braille(60)
        assert isinstance(b, str) and len(b) == 1

    def test_duration_to_braille(self):
        b = duration_to_braille(1.0)
        assert isinstance(b, str)

    def test_sequence_to_braille(self):
        result = sequence_to_braille([60, 64, 67], [1.0, 1.0, 2.0])
        assert len(result) > 0

    def test_pitch_to_color(self):
        r, g, b = pitch_to_color(60)
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

    def test_color_octave_equivalent(self):
        assert pitch_to_color(60) == pitch_to_color(72)

    def test_simplify_notation(self):
        result = simplify_notation([60, 64, 67], [1.0, 0.5, 2.0])
        assert len(result) == 3
        assert "note_name" in result[0]
        assert "color" in result[0]
        assert "shape" in result[0]

    def test_shape_notation_values(self):
        assert ShapeNotation.CIRCLE.value == "whole"
        assert ShapeNotation.TRIANGLE.value == "quarter"


class TestAudioDescription:
    def test_describe_middle_c(self):
        assert describe_pitch(60) == "middle C"

    def test_describe_interval_fifth(self):
        assert describe_interval(7) == "perfect fifth"

    def test_describe_interval_compound(self):
        desc = describe_interval(19)
        assert "fifth" in desc and "octave" in desc

    def test_describe_dynamics(self):
        assert describe_dynamics(0) == "silent"
        assert describe_dynamics(96) == "f"
        assert describe_dynamics(127) == "fff"

    def test_describe_chord_major(self):
        assert "major triad" in describe_chord([60, 64, 67])

    def test_describe_chord_minor(self):
        assert "minor triad" in describe_chord([60, 63, 67])

    def test_describe_progression(self):
        prog = describe_progression([[60, 64, 67], [65, 69, 72]])
        assert "→" in prog

    def test_describe_rhythm(self):
        desc = describe_rhythm([1.0, 0.5, 0.5], 120)
        assert "3 notes" in desc

    def test_audio_guide(self):
        guide = generate_audio_guide([60, 64, 67], [1.0, 1.0, 1.0])
        assert len(guide) == 3
        assert "middle C" in guide[0]
        assert "up" in guide[1]


class TestPackageExports:
    def test_all_exports(self):
        import models.music_accessibility as ma
        assert hasattr(ma, "HearingProfile")
        assert hasattr(ma, "AccessibilityMode")
        assert hasattr(ma, "HapticChannel")
        assert hasattr(ma, "BrailleMusic")
        assert hasattr(ma, "describe_pitch")
