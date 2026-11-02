"""Harmonic rhythm analysis: chord change rate, metric placement, harmonic acceleration.

Measures how harmony interacts with meter — chord change frequency,
on-beat vs off-beat changes, harmonic rhythm acceleration/deceleration
patterns, cadential rhythm, and phrase-level harmonic density.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class HarmonicEvent:
    """A chord onset in time."""
    onset_beat: float
    chord_label: str
    duration_beats: float = 1.0
    is_structural: bool = True  # vs passing/embellishing


@dataclass
class HarmonicRhythmProfile:
    change_rate: float           # changes per bar
    mean_duration: float         # mean chord duration in beats
    std_duration: float          # std of chord durations
    on_beat_ratio: float         # fraction of changes on strong beats
    acceleration_slope: float    # positive = speeding up
    syncopation_ratio: float     # fraction of changes on weak beats
    cadential_deceleration: float  # slowdown at phrase endings


def chord_change_rate(events: List[HarmonicEvent],
                      beats_per_bar: int = 4) -> float:
    """Average chord changes per bar."""
    if len(events) < 2:
        return 0.0
    total_beats = sum(e.duration_beats for e in events)
    n_bars = total_beats / beats_per_bar
    return (len(events) - 1) / max(n_bars, 1e-10)


def on_beat_ratio(events: List[HarmonicEvent],
                  beats_per_bar: int = 4,
                  strong_beats: Optional[List[int]] = None) -> float:
    """Fraction of chord changes occurring on strong beats."""
    if strong_beats is None:
        strong_beats = [0, beats_per_bar // 2] if beats_per_bar >= 4 else [0]
    if len(events) < 2:
        return 1.0
    on_strong = 0
    for e in events[1:]:  # skip first (always on beat)
        beat_in_bar = e.onset_beat % beats_per_bar
        if any(abs(beat_in_bar - sb) < 0.25 for sb in strong_beats):
            on_strong += 1
    return on_strong / (len(events) - 1)


def harmonic_acceleration(events: List[HarmonicEvent]) -> np.ndarray:
    """Instantaneous harmonic rate at each event (1/duration)."""
    durations = np.array([e.duration_beats for e in events])
    return 1.0 / np.maximum(durations, 0.01)


def acceleration_slope(events: List[HarmonicEvent]) -> float:
    """Linear trend in harmonic rate: positive = accelerating."""
    rates = harmonic_acceleration(events)
    if len(rates) < 2:
        return 0.0
    x = np.arange(len(rates), dtype=float)
    x -= np.mean(x)
    return float(np.sum(x * (rates - np.mean(rates))) / max(np.sum(x ** 2), 1e-10))


def phrase_harmonic_density(events: List[HarmonicEvent],
                            phrase_boundaries: List[float]) -> List[float]:
    """Chord changes per beat within each phrase."""
    densities = []
    for i in range(len(phrase_boundaries) - 1):
        start, end = phrase_boundaries[i], phrase_boundaries[i + 1]
        count = sum(1 for e in events if start <= e.onset_beat < end)
        span = end - start
        densities.append(count / max(span, 1e-10))
    return densities


def cadential_rhythm_score(events: List[HarmonicEvent],
                           cadence_window: int = 4) -> float:
    """Measure chord-duration increase in final beats (cadential broadening)."""
    if len(events) < cadence_window + 2:
        return 0.0
    cadence_durations = np.array([e.duration_beats for e in events[-cadence_window:]])
    pre_durations = np.array([e.duration_beats for e in events[:-cadence_window]])
    mean_cadence = np.mean(cadence_durations)
    mean_pre = np.mean(pre_durations)
    if mean_pre < 1e-10:
        return 0.0
    return float((mean_cadence - mean_pre) / mean_pre)


def harmonic_syncopation(events: List[HarmonicEvent],
                         beats_per_bar: int = 4) -> float:
    """Fraction of chord changes on weak subdivisions."""
    if len(events) < 2:
        return 0.0
    syncopated = 0
    for e in events[1:]:
        beat_in_bar = e.onset_beat % beats_per_bar
        if beat_in_bar % 1.0 > 0.25:
            syncopated += 1
    return syncopated / (len(events) - 1)


def harmonic_regularity(events: List[HarmonicEvent]) -> float:
    """1 − coefficient of variation of chord durations (1 = perfectly regular)."""
    durations = np.array([e.duration_beats for e in events])
    if len(durations) < 2:
        return 1.0
    mean_d = np.mean(durations)
    if mean_d < 1e-10:
        return 0.0
    cv = np.std(durations) / mean_d
    return float(max(0.0, 1.0 - cv))


def compute_harmonic_rhythm_profile(events: List[HarmonicEvent],
                                    beats_per_bar: int = 4) -> HarmonicRhythmProfile:
    durations = np.array([e.duration_beats for e in events])
    return HarmonicRhythmProfile(
        change_rate=chord_change_rate(events, beats_per_bar),
        mean_duration=float(np.mean(durations)) if len(durations) > 0 else 0.0,
        std_duration=float(np.std(durations)) if len(durations) > 1 else 0.0,
        on_beat_ratio=on_beat_ratio(events, beats_per_bar),
        acceleration_slope=acceleration_slope(events),
        syncopation_ratio=harmonic_syncopation(events, beats_per_bar),
        cadential_deceleration=cadential_rhythm_score(events),
    )


def harmonic_rhythm_distance(a: HarmonicRhythmProfile,
                             b: HarmonicRhythmProfile) -> float:
    """Euclidean distance between two harmonic rhythm profiles."""
    va = np.array([a.change_rate, a.mean_duration, a.on_beat_ratio,
                   a.acceleration_slope, a.syncopation_ratio])
    vb = np.array([b.change_rate, b.mean_duration, b.on_beat_ratio,
                   b.acceleration_slope, b.syncopation_ratio])
    return float(np.sqrt(np.sum((va - vb) ** 2)))


def classify_harmonic_rhythm(profile: HarmonicRhythmProfile) -> str:
    """Classify harmonic rhythm style."""
    if profile.change_rate < 0.5:
        return "static"      # pedal point / drone
    elif profile.change_rate < 1.5:
        return "slow"        # one change per bar or less
    elif profile.change_rate < 3.0:
        return "moderate"    # 1-2 changes per bar
    elif profile.change_rate < 6.0:
        return "fast"        # multiple changes per bar
    else:
        return "rapid"       # bebop-style rapid changes
