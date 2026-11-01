"""Tuning acoustics: string inharmonicity, piano stretch tuning,
beat frequency analysis, temperament acoustic consequences,
combination tones, and tuning stability.

Based on Fletcher & Rossing (1998) physics of musical instruments,
Railsback (1938) stretch tuning curve, and Plomp & Levelt (1965)
dissonance theory. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class StringParams:
    frequency: float
    length: float
    tension: float
    linear_density: float
    stiffness: float


@dataclass
class InharmonicityProfile:
    partials: np.ndarray
    ideal_frequencies: np.ndarray
    actual_frequencies: np.ndarray
    cents_deviation: np.ndarray
    b_coefficient: float


@dataclass
class BeatInfo:
    beat_frequency: float
    partial_1: int
    partial_2: int
    mistuning_cents: float


@dataclass
class StretchTuningProfile:
    notes: np.ndarray
    ideal_12tet: np.ndarray
    stretched: np.ndarray
    deviation_cents: np.ndarray


def inharmonicity_coefficient(diameter: float, length: float,
                              youngs_modulus: float = 2e11,
                              tension: float = 800.0) -> float:
    moment = np.pi * (diameter / 2) ** 4 / 4
    return float(np.pi ** 2 * youngs_modulus * moment / (tension * length ** 2))


def partial_frequency(f0: float, n: int, B: float) -> float:
    return f0 * n * np.sqrt(1 + B * n ** 2)


def inharmonicity_profile(f0: float, B: float,
                          n_partials: int = 16) -> InharmonicityProfile:
    ns = np.arange(1, n_partials + 1)
    ideal = f0 * ns
    actual = np.array([partial_frequency(f0, int(n), B) for n in ns])
    ratio = actual / ideal
    cents = 1200.0 * np.log2(ratio + 1e-20)
    return InharmonicityProfile(ns, ideal, actual, cents, B)


def beat_frequency(f1: float, f2: float) -> float:
    return abs(f1 - f2)


def interval_beats(f_low: float, f_high: float,
                   B_low: float = 0.0, B_high: float = 0.0,
                   max_partial: int = 8) -> List[BeatInfo]:
    beats = []
    for n1 in range(1, max_partial + 1):
        p1 = partial_frequency(f_low, n1, B_low)
        for n2 in range(1, max_partial + 1):
            p2 = partial_frequency(f_high, n2, B_high)
            bf = abs(p1 - p2)
            if 0.5 < bf < 20.0:
                cents = 1200.0 * np.log2(p1 / p2 + 1e-20)
                beats.append(BeatInfo(float(bf), n1, n2, float(abs(cents))))
    beats.sort(key=lambda b: b.beat_frequency)
    return beats


def railsback_stretch(midi_note: int, a4_freq: float = 440.0) -> float:
    semitones = midi_note - 69
    octave = semitones / 12.0
    stretch_cents = 0.0
    if octave < -2:
        stretch_cents = -15.0 * (octave + 2)
    elif octave > 2:
        stretch_cents = 12.0 * (octave - 2)
    ideal = a4_freq * 2.0 ** (semitones / 12.0)
    return ideal * 2.0 ** (stretch_cents / 1200.0)


def stretch_tuning_profile(note_range: Tuple[int, int] = (21, 108),
                           a4_freq: float = 440.0) -> StretchTuningProfile:
    notes = np.arange(note_range[0], note_range[1] + 1)
    ideal = a4_freq * 2.0 ** ((notes - 69) / 12.0)
    stretched = np.array([railsback_stretch(int(n), a4_freq) for n in notes])
    deviation = 1200.0 * np.log2(stretched / ideal + 1e-20)
    return StretchTuningProfile(notes, ideal, stretched, deviation)


def combination_tone(f1: float, f2: float,
                     order: int = 1) -> float:
    flow, fhigh = min(f1, f2), max(f1, f2)
    if order == 1:
        return fhigh - flow
    elif order == 2:
        return 2 * flow - fhigh
    elif order == 3:
        return 3 * flow - 2 * fhigh
    return abs(fhigh - flow)


def combination_tones(f1: float, f2: float,
                      max_order: int = 3) -> List[Tuple[int, float]]:
    result = []
    flow, fhigh = min(f1, f2), max(f1, f2)
    for order in range(1, max_order + 1):
        ct = combination_tone(flow, fhigh, order)
        if ct > 20.0:
            result.append((order, ct))
    result.append((0, flow + fhigh))
    return result


def plomp_levelt_dissonance(f1: float, f2: float,
                            a1: float = 1.0, a2: float = 1.0) -> float:
    s = 0.24 / (0.021 * min(f1, f2) + 19)
    d = abs(f2 - f1)
    exp1 = np.exp(-3.5 * s * d)
    exp2 = np.exp(-5.75 * s * d)
    return float(a1 * a2 * (exp1 - exp2))


def spectral_dissonance(frequencies: np.ndarray,
                        amplitudes: np.ndarray) -> float:
    n = len(frequencies)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += plomp_levelt_dissonance(
                float(frequencies[i]), float(frequencies[j]),
                float(amplitudes[i]), float(amplitudes[j])
            )
    return total


def temperament_beat_rates(temperament_cents: np.ndarray,
                           reference_freq: float = 261.63,
                           intervals: Optional[List[int]] = None
                           ) -> np.ndarray:
    if intervals is None:
        intervals = [3, 4, 5, 7, 8, 9]
    n = len(intervals)
    beats = np.zeros(n)
    for i, semitones in enumerate(intervals):
        cents = float(temperament_cents[semitones % 12])
        freq_actual = reference_freq * 2.0 ** (cents / 1200.0)
        just_ratios = {3: 6/5, 4: 5/4, 5: 4/3, 7: 3/2, 8: 8/5, 9: 5/3}
        if semitones in just_ratios:
            freq_just = reference_freq * just_ratios[semitones]
            beats[i] = abs(freq_actual - freq_just)
    return beats


def tuning_stability(frequencies: np.ndarray,
                     target_frequencies: np.ndarray) -> float:
    if len(frequencies) == 0:
        return 1.0
    cents_error = 1200.0 * np.log2(frequencies / (target_frequencies + 1e-20) + 1e-20)
    rms_error = np.sqrt(np.mean(cents_error ** 2))
    return float(1.0 / (1.0 + rms_error / 10.0))


def octave_purity(f_low: float, f_high: float,
                  B: float = 0.0) -> float:
    ideal_ratio = 2.0
    actual_ratio = f_high / f_low
    if B > 0:
        p2_low = partial_frequency(f_low, 2, B)
        ideal_ratio = p2_low / f_low
    cents_error = 1200.0 * np.log2(actual_ratio / ideal_ratio + 1e-20)
    return float(abs(cents_error))


def equal_temperament_cents() -> np.ndarray:
    return np.arange(12) * 100.0


def just_intonation_cents() -> np.ndarray:
    ratios = [1/1, 16/15, 9/8, 6/5, 5/4, 4/3,
              45/32, 3/2, 8/5, 5/3, 9/5, 15/8]
    return 1200.0 * np.log2(np.array(ratios))


def pythagorean_cents() -> np.ndarray:
    ratios = [1.0]
    r = 1.0
    for _ in range(11):
        r *= 3.0 / 2.0
        while r >= 2.0:
            r /= 2.0
        ratios.append(r)
    ratios.sort()
    return 1200.0 * np.log2(np.array(ratios))


def temperament_comparison(temp1_cents: np.ndarray,
                           temp2_cents: np.ndarray) -> float:
    diff = temp1_cents[:12] - temp2_cents[:12]
    return float(np.sqrt(np.mean(diff ** 2)))
