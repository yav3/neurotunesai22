"""Adaptive music interface configuration for accessibility."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AccessibilityMode(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    MOTOR = "motor"
    COGNITIVE = "cognitive"
    MULTI = "multi"


class InputModality(Enum):
    TOUCH = "touch"
    GAZE = "gaze"
    BREATH = "breath"
    SWITCH = "switch"
    GESTURE = "gesture"
    VOICE = "voice"


@dataclass
class AdaptiveConfig:
    mode: AccessibilityMode
    input_modality: InputModality = InputModality.TOUCH
    response_time_multiplier: float = 1.0
    visual_contrast: float = 1.0
    haptic_feedback: bool = True
    font_scale: float = 1.0


def adapt_tempo(base_tempo: float, motor_capability: float) -> float:
    """Slow tempo for reduced motor capability. motor_capability in [0, 1]."""
    capability = max(0.1, min(1.0, motor_capability))
    return base_tempo * capability


def simplify_interface(n_controls: int, cognitive_load_factor: float) -> int:
    """Reduce visible controls for cognitive accessibility."""
    factor = max(0.2, min(1.0, cognitive_load_factor))
    return max(1, int(n_controls * factor))


def map_input_to_midi(
    input_value: float,
    modality: InputModality,
    config: AdaptiveConfig,
) -> int:
    """Normalize diverse input modalities to MIDI note number (0-127)."""
    v = max(0.0, min(1.0, input_value))
    if modality == InputModality.BREATH:
        midi = int(v * 48 + 48)  # C3 to C7
    elif modality == InputModality.GAZE:
        midi = int(v * 36 + 48)  # narrower range
    elif modality == InputModality.SWITCH:
        midi = int(v * 12 + 60)  # one octave
    else:
        midi = int(v * 127)
    return max(0, min(127, midi))


def latency_tolerance(mode: AccessibilityMode) -> float:
    """Maximum acceptable latency in milliseconds."""
    tolerances = {
        AccessibilityMode.VISUAL: 50.0,
        AccessibilityMode.AUDITORY: 20.0,
        AccessibilityMode.MOTOR: 100.0,
        AccessibilityMode.COGNITIVE: 200.0,
        AccessibilityMode.MULTI: 150.0,
    }
    return tolerances.get(mode, 50.0)


def recommended_font_scale(mode: AccessibilityMode) -> float:
    if mode in (AccessibilityMode.VISUAL, AccessibilityMode.COGNITIVE):
        return 1.5
    return 1.0


def recommended_color_scheme(mode: AccessibilityMode) -> dict[str, str]:
    if mode == AccessibilityMode.VISUAL:
        return {"background": "#000000", "foreground": "#FFFF00", "accent": "#00FFFF"}
    return {"background": "#FFFFFF", "foreground": "#000000", "accent": "#0066CC"}
