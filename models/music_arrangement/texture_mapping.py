"""Musical texture classification and density analysis."""

from __future__ import annotations

from enum import Enum


class TextureType(Enum):
    MONOPHONIC = "monophonic"
    HOMOPHONIC = "homophonic"
    POLYPHONIC = "polyphonic"
    HETEROPHONIC = "heterophonic"
    HOMORHYTHMIC = "homorhythmic"


def classify_texture(
    simultaneous_parts: int,
    rhythm_similarity: float,
    melodic_independence: float,
) -> TextureType:
    if simultaneous_parts <= 1:
        return TextureType.MONOPHONIC
    if rhythm_similarity > 0.9 and melodic_independence < 0.3:
        return TextureType.HOMORHYTHMIC
    if rhythm_similarity > 0.7 and melodic_independence < 0.5:
        return TextureType.HOMOPHONIC
    if melodic_independence > 0.7:
        return TextureType.POLYPHONIC
    return TextureType.HETEROPHONIC


def texture_density(
    n_voices: int,
    pitch_range_semitones: int,
) -> float:
    if pitch_range_semitones <= 0 or n_voices <= 0:
        return 0.0
    return n_voices / pitch_range_semitones


def recommended_texture(
    n_available_parts: int,
    tempo_bpm: float,
    dynamic_level: float,
) -> TextureType:
    if n_available_parts <= 1:
        return TextureType.MONOPHONIC
    if tempo_bpm > 160 and n_available_parts > 2:
        return TextureType.HOMORHYTHMIC
    if dynamic_level < 0.3:
        return TextureType.MONOPHONIC if n_available_parts <= 2 else TextureType.HOMOPHONIC
    if n_available_parts >= 4 and tempo_bpm < 100:
        return TextureType.POLYPHONIC
    return TextureType.HOMOPHONIC
