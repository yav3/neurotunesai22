"""Texture planning and density control for composition.

Homorhythmic vs polyphonic density curves, registral spread,
voice-count planning, and textural contrast analysis.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class TextureType(Enum):
    MONOPHONIC = "monophonic"
    HOMOPHONIC = "homophonic"
    POLYPHONIC = "polyphonic"
    HETEROPHONIC = "heterophonic"
    UNISON = "unison"
    ANTIPHONAL = "antiphonal"


@dataclass
class TextureSegment:
    start_beat: float
    end_beat: float
    texture_type: TextureType
    n_voices: int
    density: float
    registral_spread: float


@dataclass
class TexturePlan:
    segments: List[TextureSegment]
    total_beats: float
    avg_density: float
    contrast_score: float


def onset_density(onsets_per_voice: List[np.ndarray], window_beats: float,
                  total_beats: float) -> np.ndarray:
    """Compute onset density curve over time.

    Each element of onsets_per_voice is an array of beat positions.
    Returns density per window.
    """
    n_windows = max(1, int(total_beats / window_beats))
    density = np.zeros(n_windows)
    for onsets in onsets_per_voice:
        for t in onsets:
            idx = min(int(t / window_beats), n_windows - 1)
            if 0 <= idx < n_windows:
                density[idx] += 1
    density /= window_beats
    return density


def registral_spread(pitches_per_voice: List[np.ndarray]) -> float:
    """Registral spread: range of all sounding pitches in semitones."""
    all_pitches = np.concatenate([p for p in pitches_per_voice if len(p) > 0])
    if len(all_pitches) == 0:
        return 0.0
    return float(np.ptp(all_pitches))


def registral_centroid(pitches_per_voice: List[np.ndarray]) -> float:
    """Mean pitch across all voices."""
    all_pitches = np.concatenate([p for p in pitches_per_voice if len(p) > 0])
    if len(all_pitches) == 0:
        return 60.0
    return float(np.mean(all_pitches))


def classify_texture(n_voices: int, onset_correlation: float) -> TextureType:
    """Classify texture from voice count and rhythmic correlation.

    onset_correlation: mean pairwise correlation of onset patterns.
    """
    if n_voices <= 1:
        return TextureType.MONOPHONIC
    if onset_correlation > 0.85:
        return TextureType.HOMOPHONIC
    if onset_correlation > 0.6:
        return TextureType.HETEROPHONIC
    return TextureType.POLYPHONIC


def voice_onset_correlation(onsets_a: np.ndarray, onsets_b: np.ndarray,
                            resolution: float = 0.25,
                            total_beats: float = 16.0) -> float:
    """Correlation between two onset-time sequences quantised to a grid."""
    n_bins = max(1, int(total_beats / resolution))
    grid_a = np.zeros(n_bins)
    grid_b = np.zeros(n_bins)
    for t in onsets_a:
        idx = min(int(t / resolution), n_bins - 1)
        if 0 <= idx < n_bins:
            grid_a[idx] = 1.0
    for t in onsets_b:
        idx = min(int(t / resolution), n_bins - 1)
        if 0 <= idx < n_bins:
            grid_b[idx] = 1.0
    if np.std(grid_a) < 1e-10 or np.std(grid_b) < 1e-10:
        return 0.0
    return float(np.corrcoef(grid_a, grid_b)[0, 1])


def plan_texture_arc(total_beats: float, n_sections: int = 4,
                     arc_shape: str = "arch") -> TexturePlan:
    """Plan a textural arc over a piece.

    arc_shape: "arch" (thin→thick→thin), "crescendo" (thin→thick),
               "rondo" (alternating), "plateau" (thin→thick→thick→thin).
    """
    section_len = total_beats / n_sections
    segments: List[TextureSegment] = []

    for i in range(n_sections):
        frac = i / max(n_sections - 1, 1)
        if arc_shape == "arch":
            density = 0.3 + 0.7 * np.sin(np.pi * frac)
            n_v = 1 + int(3 * np.sin(np.pi * frac))
        elif arc_shape == "crescendo":
            density = 0.2 + 0.8 * frac
            n_v = 1 + int(3 * frac)
        elif arc_shape == "rondo":
            density = 0.3 if i % 2 == 0 else 0.8
            n_v = 2 if i % 2 == 0 else 4
        elif arc_shape == "plateau":
            if frac < 0.25:
                density = 0.3
                n_v = 1
            elif frac > 0.75:
                density = 0.3
                n_v = 1
            else:
                density = 0.8
                n_v = 4
        else:
            density = 0.5
            n_v = 2

        tex = TextureType.MONOPHONIC if n_v <= 1 else (
            TextureType.HOMOPHONIC if density > 0.6 else TextureType.POLYPHONIC
        )
        segments.append(TextureSegment(
            start_beat=i * section_len,
            end_beat=(i + 1) * section_len,
            texture_type=tex,
            n_voices=n_v,
            density=density,
            registral_spread=12.0 * n_v,
        ))

    densities = [s.density for s in segments]
    contrast = _texture_contrast(segments)
    return TexturePlan(
        segments=segments,
        total_beats=total_beats,
        avg_density=float(np.mean(densities)),
        contrast_score=contrast,
    )


def _texture_contrast(segments: List[TextureSegment]) -> float:
    """Measure textural contrast between adjacent segments."""
    if len(segments) < 2:
        return 0.0
    changes = 0.0
    for i in range(1, len(segments)):
        if segments[i].texture_type != segments[i - 1].texture_type:
            changes += 1.0
        changes += abs(segments[i].density - segments[i - 1].density)
        changes += abs(segments[i].n_voices - segments[i - 1].n_voices) / 4.0
    return changes / (len(segments) - 1)


def density_interpolation(start_density: float, end_density: float,
                          n_steps: int, curve: str = "linear") -> np.ndarray:
    """Interpolate density values for gradual textural transitions."""
    t = np.linspace(0, 1, n_steps)
    if curve == "exponential":
        t = t ** 2
    elif curve == "logarithmic":
        t = np.sqrt(t)
    elif curve == "sigmoid":
        t = 1.0 / (1.0 + np.exp(-6 * (t - 0.5)))
    return start_density + (end_density - start_density) * t


def voice_count_curve(total_beats: float, max_voices: int = 4,
                      shape: str = "arch", resolution: float = 1.0) -> np.ndarray:
    """Generate a voice-count curve over time."""
    n = max(1, int(total_beats / resolution))
    t = np.linspace(0, 1, n)
    if shape == "arch":
        raw = np.sin(np.pi * t)
    elif shape == "crescendo":
        raw = t
    elif shape == "diminuendo":
        raw = 1.0 - t
    else:
        raw = np.ones(n) * 0.5
    return np.round(1 + (max_voices - 1) * raw).astype(int)
