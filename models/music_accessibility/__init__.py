"""Music accessibility: hearing profiles, adaptive interfaces, haptic rendering, visual notation, audio description."""

from .hearing_profile import (
    HearingLossType, HearingProfile, STANDARD_AUDIOGRAM_FREQUENCIES,
    frequency_response_compensation, tinnitus_notch_filter,
    speech_intelligibility_index, comfortable_listening_level,
)
from .adaptive_interface import (
    AccessibilityMode, InputModality, AdaptiveConfig,
    adapt_tempo, simplify_interface, map_input_to_midi,
    latency_tolerance, recommended_font_scale, recommended_color_scheme,
)
from .haptic_music import (
    HapticChannel, HAPTIC_LAYOUTS,
    audio_to_haptic, rhythm_haptic, bass_haptic, melody_contour_haptic,
)
from .visual_notation import (
    BrailleMusic, pitch_to_braille, duration_to_braille, sequence_to_braille,
    ColorNotation, pitch_to_color, ShapeNotation,
    SimplifiedStaff, simplify_notation,
)
from .audio_description import (
    describe_pitch, describe_interval, describe_rhythm,
    describe_dynamics, describe_chord, describe_progression,
    generate_audio_guide,
)

from .score_simplification import (
    AbilityLevel, SimplificationProfile, LEVEL_PROFILES,
    NoteEvent, difficulty_score, simplify_melody,
    reduce_polyphony, suggest_level, tempo_adjustment,
)

__all__ = [
    "AbilityLevel", "SimplificationProfile", "LEVEL_PROFILES",
    "NoteEvent", "difficulty_score", "simplify_melody",
    "reduce_polyphony", "suggest_level", "tempo_adjustment",
    "HearingLossType", "HearingProfile", "STANDARD_AUDIOGRAM_FREQUENCIES",
    "frequency_response_compensation", "tinnitus_notch_filter",
    "speech_intelligibility_index", "comfortable_listening_level",
    "AccessibilityMode", "InputModality", "AdaptiveConfig",
    "adapt_tempo", "simplify_interface", "map_input_to_midi",
    "latency_tolerance", "recommended_font_scale", "recommended_color_scheme",
    "HapticChannel", "HAPTIC_LAYOUTS",
    "audio_to_haptic", "rhythm_haptic", "bass_haptic", "melody_contour_haptic",
    "BrailleMusic", "pitch_to_braille", "duration_to_braille", "sequence_to_braille",
    "ColorNotation", "pitch_to_color", "ShapeNotation",
    "SimplifiedStaff", "simplify_notation",
    "describe_pitch", "describe_interval", "describe_rhythm",
    "describe_dynamics", "describe_chord", "describe_progression",
    "generate_audio_guide",
]
