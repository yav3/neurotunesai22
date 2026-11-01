"""Harmonic tension modeling: roughness, dissonance, tonal distance, resolution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from models.music_harmony.chords import Chord, ChordQuality, CHORD_TYPES
from models.music_harmony.modulation import Key
from models.music_harmony.functional import HarmonicFunction, classify_function


INTERVAL_DISSONANCE = np.array([
    0.0,   # unison
    1.0,   # m2
    0.8,   # M2
    0.25,  # m3
    0.2,   # M3
    0.15,  # P4
    0.65,  # tritone
    0.05,  # P5
    0.25,  # m6
    0.2,   # M6
    0.55,  # m7
    0.5,   # M7
], dtype=np.float64)


def interval_dissonance(interval: int) -> float:
    return float(INTERVAL_DISSONANCE[abs(interval) % 12])


def chord_dissonance(chord: Chord) -> float:
    pcs = sorted(chord.pitch_classes)
    if len(pcs) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            total += interval_dissonance(pcs[j] - pcs[i])
            count += 1
    return total / count if count > 0 else 0.0


ROUGHNESS_CURVE_PARAMS = (3.5, 5.75, 0.24)


def roughness_pair(f1: float, f2: float) -> float:
    if f1 <= 0 or f2 <= 0:
        return 0.0
    a, b, s = ROUGHNESS_CURVE_PARAMS
    f_low, f_high = min(f1, f2), max(f1, f2)
    x = s * (f_high - f_low) / f_low
    return float(np.exp(-a * x) - np.exp(-b * x))


def chord_roughness(frequencies: Sequence[float]) -> float:
    n = len(frequencies)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += roughness_pair(frequencies[i], frequencies[j])
            count += 1
    return total / count if count > 0 else 0.0


def pitch_to_freq(midi_pitch: int) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def chord_roughness_from_pitches(pitches: Sequence[int]) -> float:
    freqs = [pitch_to_freq(p) for p in pitches]
    return chord_roughness(freqs)


@dataclass
class TensionProfile:
    dissonance: list[float]
    roughness: list[float]
    tonal_distance: list[float]
    composite: list[float]


def _tonal_distance(chord: Chord, key: Key) -> float:
    chord_pcs = chord.pitch_classes
    key_pcs = key.pitch_classes
    if not chord_pcs:
        return 0.0
    non_diatonic = len(chord_pcs - key_pcs)
    return non_diatonic / len(chord_pcs)


def tension_profile(
    chords: Sequence[Chord],
    key: Key,
    voicings: Sequence[Sequence[int]] | None = None,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> TensionProfile:
    diss: list[float] = []
    rough: list[float] = []
    tonal: list[float] = []

    for i, chord in enumerate(chords):
        diss.append(chord_dissonance(chord))
        tonal.append(_tonal_distance(chord, key))

        if voicings is not None and i < len(voicings):
            rough.append(chord_roughness_from_pitches(voicings[i]))
        else:
            intervals = CHORD_TYPES.get(chord.quality, (0, 4, 7))
            default_pitches = [60 + chord.root + iv for iv in intervals]
            rough.append(chord_roughness_from_pitches(default_pitches))

    w_d, w_r, w_t = weights
    composite = [
        w_d * d + w_r * r + w_t * t
        for d, r, t in zip(diss, rough, tonal)
    ]

    return TensionProfile(
        dissonance=diss, roughness=rough,
        tonal_distance=tonal, composite=composite,
    )


def resolution_strength(pre_chord: Chord, post_chord: Chord, key: Key) -> float:
    t_pre = chord_dissonance(pre_chord) + _tonal_distance(pre_chord, key)
    t_post = chord_dissonance(post_chord) + _tonal_distance(post_chord, key)
    drop = t_pre - t_post
    func_post = classify_function(post_chord, key.root, key.mode)
    tonic_bonus = 0.3 if func_post == HarmonicFunction.TONIC else 0.0
    return max(0.0, min(1.0, drop + tonic_bonus))


def harmonic_rhythm_density(
    onset_times: Sequence[float],
    total_duration: float,
) -> float:
    if total_duration <= 0 or len(onset_times) == 0:
        return 0.0
    return len(onset_times) / total_duration


def harmonic_rhythm_regularity(onset_times: Sequence[float]) -> float:
    if len(onset_times) < 2:
        return 1.0
    intervals = np.diff(onset_times)
    if np.mean(intervals) < 1e-10:
        return 1.0
    cv = float(np.std(intervals) / np.mean(intervals))
    return max(0.0, 1.0 - cv)


def tension_contour_features(profile: TensionProfile) -> dict[str, float]:
    c = np.array(profile.composite)
    if len(c) == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "range": 0.0, "slope": 0.0}
    slope = 0.0
    if len(c) > 1:
        x = np.arange(len(c), dtype=np.float64)
        slope = float(np.polyfit(x, c, 1)[0])
    return {
        "mean": float(np.mean(c)),
        "std": float(np.std(c)),
        "max": float(np.max(c)),
        "range": float(np.max(c) - np.min(c)),
        "slope": slope,
    }
