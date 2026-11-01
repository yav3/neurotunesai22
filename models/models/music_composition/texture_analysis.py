"""Texture analysis: monophonic/homophonic/polyphonic classification, density measures.

Implements musical texture classification, voice counting, textural density
estimation, and texture segmentation for compositional analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TextureType(Enum):
    MONOPHONIC = "monophonic"
    HOMOPHONIC = "homophonic"
    POLYPHONIC = "polyphonic"
    HETEROPHONIC = "heterophonic"
    HOMORHYTHMIC = "homorhythmic"


@dataclass(frozen=True)
class TextureProfile:
    texture_type: TextureType
    n_voices: int
    density: float
    rhythmic_independence: float
    pitch_spread: float
    onset_synchrony: float


def estimate_voice_count(pitches: np.ndarray, threshold_semitones: float = 2.0) -> int:
    if len(pitches) == 0:
        return 0
    unique = np.unique(pitches[pitches > 0])
    if len(unique) <= 1:
        return max(1, len(unique))
    sorted_p = np.sort(unique)
    gaps = np.diff(sorted_p)
    voices = 1
    for g in gaps:
        if g > threshold_semitones:
            voices += 1
    return voices


def pitch_spread(pitches: np.ndarray) -> float:
    active = pitches[pitches > 0]
    if len(active) < 2:
        return 0.0
    return float(np.max(active) - np.min(active))


def onset_synchrony(onset_times: list[list[float]], tolerance_s: float = 0.05) -> float:
    if len(onset_times) < 2:
        return 1.0
    all_onsets: list[float] = []
    for voice in onset_times:
        all_onsets.extend(voice)
    if not all_onsets:
        return 1.0
    all_onsets.sort()
    total_pairs = 0
    synced_pairs = 0
    for i in range(len(onset_times)):
        for j in range(i + 1, len(onset_times)):
            for t_i in onset_times[i]:
                for t_j in onset_times[j]:
                    total_pairs += 1
                    if abs(t_i - t_j) <= tolerance_s:
                        synced_pairs += 1
    if total_pairs == 0:
        return 1.0
    return synced_pairs / total_pairs


def rhythmic_independence_score(onset_times: list[list[float]], window_s: float = 1.0) -> float:
    if len(onset_times) < 2:
        return 0.0
    sync = onset_synchrony(onset_times)
    return 1.0 - sync


def textural_density(
    n_voices: int,
    note_density_per_second: float,
    pitch_range_semitones: float,
    max_voices: int = 8,
    max_density: float = 20.0,
    max_range: float = 48.0,
) -> float:
    voice_component = min(n_voices / max_voices, 1.0)
    density_component = min(note_density_per_second / max_density, 1.0)
    range_component = min(pitch_range_semitones / max_range, 1.0)
    return float(0.4 * voice_component + 0.35 * density_component + 0.25 * range_component)


def classify_texture(
    n_voices: int,
    rhythmic_indep: float,
    onset_sync: float,
) -> TextureType:
    if n_voices <= 1:
        return TextureType.MONOPHONIC
    if onset_sync > 0.8:
        return TextureType.HOMORHYTHMIC
    if rhythmic_indep > 0.6:
        return TextureType.POLYPHONIC
    if rhythmic_indep > 0.3:
        return TextureType.HOMOPHONIC
    return TextureType.HOMOPHONIC


def analyze_texture(
    pitches: np.ndarray,
    onset_times: list[list[float]] | None = None,
    note_density: float = 5.0,
) -> TextureProfile:
    n_v = estimate_voice_count(pitches)
    spread = pitch_spread(pitches)

    if onset_times and len(onset_times) >= 2:
        sync = onset_synchrony(onset_times)
        indep = rhythmic_independence_score(onset_times)
    else:
        sync = 0.5
        indep = 0.5 if n_v > 1 else 0.0

    tex_type = classify_texture(n_v, indep, sync)
    density = textural_density(n_v, note_density, spread)

    return TextureProfile(
        texture_type=tex_type,
        n_voices=n_v,
        density=density,
        rhythmic_independence=indep,
        pitch_spread=spread,
        onset_synchrony=sync,
    )


@dataclass(frozen=True)
class TextureSegment:
    start_frame: int
    end_frame: int
    texture_type: TextureType
    density: float


def segment_texture(
    pitch_frames: np.ndarray,
    frame_size: int = 16,
) -> list[TextureSegment]:
    n = len(pitch_frames)
    if n == 0:
        return []
    segments: list[TextureSegment] = []
    i = 0
    while i < n:
        end = min(i + frame_size, n)
        chunk = pitch_frames[i:end]
        n_v = estimate_voice_count(chunk.flatten() if chunk.ndim > 1 else chunk)
        spread = pitch_spread(chunk.flatten() if chunk.ndim > 1 else chunk)
        density = textural_density(n_v, 5.0, spread)
        tex_type = classify_texture(n_v, 0.5, 0.5)
        segments.append(TextureSegment(
            start_frame=i,
            end_frame=end,
            texture_type=tex_type,
            density=density,
        ))
        i = end
    return segments
