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
    DescriptionLevel, AudioDescription,
    describe_pitch, describe_interval, describe_rhythm,
    describe_dynamics, describe_chord, describe_progression,
    generate_audio_guide,
    describe_tempo, describe_rhythm_features, describe_harmony_features,
    describe_dynamics_features, describe_timbre_features,
    generate_full_description,
)
from .haptic_mapping import (
    HapticPattern, HapticEvent,
    beat_to_haptic, melody_to_haptic, dynamics_to_intensity,
    rhythm_pattern_to_vibration, continuous_haptic_envelope,
)
from .cognitive_load import (
    CognitiveLoadEstimate,
    harmonic_complexity_load, rhythmic_complexity_load,
    textural_density_load, combined_cognitive_load,
    passage_cognitive_profile,
)

from .sonification import (
    ParameterMapping, PitchSonifier, RhythmSonifier,
    MultidimensionalSonifier, SonificationEvaluator,
)
from .speech_to_music import (
    PitchTracker, VoiceToMIDI, VocalGestureDetector, VoiceCommandClassifier,
)
from .score_simplification import (
    AbilityLevel, SimplificationProfile, LEVEL_PROFILES,
    NoteEvent, difficulty_score, simplify_melody,
    reduce_polyphony, suggest_level, tempo_adjustment,
)

from .frequency_remapping import (
    STANDARD_FREQS, AudiogramPoint, RemappingConfig, RemappingResult,
    interpolate_audiogram, hearing_loss_gain, apply_frequency_gain,
    frequency_lowering, spectral_envelope_compression,
    equal_loudness_weight, full_remapping_pipeline,
    compensation_error, bandwidth_utilization,
)

__all__ = [
    "STANDARD_FREQS", "AudiogramPoint", "RemappingConfig", "RemappingResult",
    "interpolate_audiogram", "hearing_loss_gain", "apply_frequency_gain",
    "frequency_lowering", "spectral_envelope_compression",
    "equal_loudness_weight", "full_remapping_pipeline",
    "compensation_error", "bandwidth_utilization",
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
    "DescriptionLevel", "AudioDescription",
    "describe_pitch", "describe_interval", "describe_rhythm",
    "describe_dynamics", "describe_chord", "describe_progression",
    "generate_audio_guide",
    "describe_tempo", "describe_rhythm_features", "describe_harmony_features",
    "describe_dynamics_features", "describe_timbre_features",
    "generate_full_description",
    "HapticPattern", "HapticEvent",
    "beat_to_haptic", "melody_to_haptic", "dynamics_to_intensity",
    "rhythm_pattern_to_vibration", "continuous_haptic_envelope",
    "CognitiveLoadEstimate",
    "harmonic_complexity_load", "rhythmic_complexity_load",
    "textural_density_load", "combined_cognitive_load",
    "passage_cognitive_profile",
    "ParameterMapping", "PitchSonifier", "RhythmSonifier",
    "MultidimensionalSonifier", "SonificationEvaluator",
    "PitchTracker", "VoiceToMIDI", "VocalGestureDetector", "VoiceCommandClassifier",
    "PlaybackRegion", "CuePoint", "PlaybackProfile",
    "time_stretch_factor", "pitch_shift_ratio", "resample_signal",
    "mono_mixdown", "dynamic_compression_simple",
    "extract_loop_region", "apply_crossfade", "find_nearest_cue",
    "generate_practice_regions", "playback_duration", "apply_profile",
    "accessibility_score",
    "MotorProfile", "MotorCapability",
    "simplify_intervals", "quantize_timing", "reduce_polyphony",
    "adaptive_tempo", "target_prediction", "tremor_filter",
    "error_tolerance_score", "fatigue_model", "difficulty_estimate", "adapt_score",
    "SimplificationLevel", "CognitiveMetric", "CognitiveProfile", "SimplificationResult",
    "estimate_information_rate", "cs_rhythmic_complexity", "cs_melodic_complexity",
    "harmonic_complexity_score", "cognitive_load_estimate",
    "cs_simplify_rhythm", "cs_simplify_melody", "reduce_texture",
    "regularise_structure", "adapt_tempo_to_profile", "create_simplified_version",
    "BrailleOctave", "OCTAVE_MARKS",
    "NOTE_SIGNS_EIGHTH", "NOTE_SIGNS_QUARTER", "NOTE_SIGNS_HALF", "NOTE_SIGNS_WHOLE",
    "REST_SIGNS", "ACCIDENTAL_SIGNS",
    "BrailleNote", "BrailleMeasure",
    "note_to_braille", "measure_to_braille", "rest_to_braille",
    "midi_to_braille_note", "melody_to_braille",
    "braille_to_text_description", "interval_sign",
    "score_to_braille", "braille_line_length",
    "Audiogram", "HearingAidPrescription", "CIVocoderConfig",
    "ISO226_FREQUENCIES", "ISO226_THRESHOLDS_0_PHON",
    "presbycusis_threshold_shift", "noise_induced_threshold_shift",
    "simulate_audiogram", "hearing_aid_gain",
    "cochlear_implant_vocoder", "hm_sii",
    "frequency_compression", "dynamic_range_compression",
]

from .assistive_playback import (
    PlaybackRegion, CuePoint, PlaybackProfile,
    time_stretch_factor, pitch_shift_ratio, resample_signal,
    mono_mixdown, dynamic_compression_simple,
    extract_loop_region, apply_crossfade, find_nearest_cue,
    generate_practice_regions, playback_duration, apply_profile,
    accessibility_score,
)
from .motor_adaptation import (
    MotorProfile, MotorCapability,
    simplify_intervals, quantize_timing, reduce_polyphony,
    adaptive_tempo, target_prediction, tremor_filter,
    error_tolerance_score, fatigue_model, difficulty_estimate, adapt_score,
)
from .cognitive_simplification import (
    SimplificationLevel, CognitiveMetric, CognitiveProfile, SimplificationResult,
    estimate_information_rate,
    rhythmic_complexity as cs_rhythmic_complexity,
    melodic_complexity as cs_melodic_complexity,
    harmonic_complexity_score, cognitive_load_estimate,
    simplify_rhythm as cs_simplify_rhythm,
    simplify_melody as cs_simplify_melody,
    reduce_texture, regularise_structure, adapt_tempo_to_profile,
    create_simplified_version,
)
from .hearing_models import (
    Audiogram, HearingAidPrescription, CIVocoderConfig,
    ISO226_FREQUENCIES, ISO226_THRESHOLDS_0_PHON,
    presbycusis_threshold_shift, noise_induced_threshold_shift,
    simulate_audiogram, hearing_aid_gain,
    cochlear_implant_vocoder, speech_intelligibility_index as hm_sii,
    frequency_compression, dynamic_range_compression,
)
from .braille_music import (
    BrailleOctave, OCTAVE_MARKS,
    NOTE_SIGNS_EIGHTH, NOTE_SIGNS_QUARTER, NOTE_SIGNS_HALF, NOTE_SIGNS_WHOLE,
    REST_SIGNS, ACCIDENTAL_SIGNS,
    BrailleNote, BrailleMeasure,
    note_to_braille, measure_to_braille, rest_to_braille,
    midi_to_braille_note, melody_to_braille,
    braille_to_text_description, interval_sign,
    score_to_braille, braille_line_length,
)
