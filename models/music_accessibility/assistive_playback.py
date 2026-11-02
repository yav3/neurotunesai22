"""Assistive playback: tempo control, pitch shifting, loop regions, cue navigation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PlaybackRegion:
    start_sec: float
    end_sec: float
    label: str = ""
    loop_count: int = 1


@dataclass(frozen=True)
class CuePoint:
    time_sec: float
    label: str
    category: str = "section"


@dataclass(frozen=True)
class PlaybackProfile:
    tempo_factor: float = 1.0
    pitch_shift_semitones: float = 0.0
    mono_mixdown: bool = False
    boost_vocals_db: float = 0.0
    reduce_dynamics: bool = False
    loop_regions: List[PlaybackRegion] = field(default_factory=list)
    cue_points: List[CuePoint] = field(default_factory=list)


def time_stretch_factor(original_bpm: float, target_bpm: float) -> float:
    if original_bpm <= 0 or target_bpm <= 0:
        return 1.0
    return target_bpm / original_bpm


def pitch_shift_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)


def resample_signal(signal: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0 or factor == 1.0:
        return signal.copy()
    n_out = int(len(signal) / factor)
    if n_out < 1:
        return signal[:1].copy()
    indices = np.linspace(0, len(signal) - 1, n_out)
    return np.interp(indices, np.arange(len(signal)), signal)


def mono_mixdown(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left + right) * 0.5


def dynamic_compression_simple(signal: np.ndarray,
                                threshold_db: float = -20.0,
                                ratio: float = 4.0) -> np.ndarray:
    env = np.abs(signal)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-10))
    over = np.maximum(env_db - threshold_db, 0.0)
    gain_reduction_db = over * (1.0 - 1.0 / ratio)
    gain = 10.0 ** (-gain_reduction_db / 20.0)
    return signal * gain


def extract_loop_region(signal: np.ndarray, sr: int,
                        region: PlaybackRegion) -> np.ndarray:
    start = int(region.start_sec * sr)
    end = int(region.end_sec * sr)
    start = max(0, min(start, len(signal)))
    end = max(start, min(end, len(signal)))
    segment = signal[start:end]
    if region.loop_count <= 1:
        return segment
    return np.tile(segment, region.loop_count)


def apply_crossfade(a: np.ndarray, b: np.ndarray,
                    fade_samples: int) -> np.ndarray:
    fade_samples = min(fade_samples, len(a), len(b))
    if fade_samples <= 0:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    result = np.concatenate([a[:-fade_samples],
                              a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in,
                              b[fade_samples:]])
    return result


def find_nearest_cue(cue_points: List[CuePoint],
                     current_time: float,
                     direction: str = "forward") -> Optional[CuePoint]:
    if not cue_points:
        return None
    if direction == "forward":
        candidates = [c for c in cue_points if c.time_sec > current_time + 0.01]
        return min(candidates, key=lambda c: c.time_sec) if candidates else None
    else:
        candidates = [c for c in cue_points if c.time_sec < current_time - 0.01]
        return max(candidates, key=lambda c: c.time_sec) if candidates else None


def generate_practice_regions(cue_points: List[CuePoint],
                               total_duration: float) -> List[PlaybackRegion]:
    if not cue_points:
        return [PlaybackRegion(0, total_duration, "full")]
    sorted_cues = sorted(cue_points, key=lambda c: c.time_sec)
    regions = []
    for i, cue in enumerate(sorted_cues):
        end = sorted_cues[i + 1].time_sec if i + 1 < len(sorted_cues) else total_duration
        regions.append(PlaybackRegion(cue.time_sec, end, cue.label))
    return regions


def playback_duration(signal_length: int, sr: int,
                      profile: PlaybackProfile) -> float:
    base_duration = signal_length / sr
    stretched = base_duration / max(profile.tempo_factor, 0.01)
    loop_extra = sum(
        (r.end_sec - r.start_sec) * max(r.loop_count - 1, 0)
        for r in profile.loop_regions
    )
    return stretched + loop_extra


def apply_profile(signal: np.ndarray, sr: int,
                  profile: PlaybackProfile) -> np.ndarray:
    out = signal.copy()
    if profile.tempo_factor != 1.0:
        out = resample_signal(out, profile.tempo_factor)
    if profile.reduce_dynamics:
        out = dynamic_compression_simple(out)
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.95
    return out


def accessibility_score(profile: PlaybackProfile) -> float:
    score = 0.5
    if 0.5 <= profile.tempo_factor <= 0.8:
        score += 0.15
    if profile.mono_mixdown:
        score += 0.1
    if profile.reduce_dynamics:
        score += 0.1
    if len(profile.cue_points) >= 3:
        score += 0.1
    if len(profile.loop_regions) >= 1:
        score += 0.05
    return min(score, 1.0)
