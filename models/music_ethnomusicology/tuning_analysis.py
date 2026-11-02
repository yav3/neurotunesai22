"""Tuning analysis: system identification, intonation measurement, adaptive correction."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class IntonationSystem(str, Enum):
    EQUAL_12 = "equal_12"
    JUST = "just"
    PYTHAGOREAN = "pythagorean"
    MEANTONE = "meantone"
    MAQAM = "maqam"
    GAMELAN_SLENDRO = "gamelan_slendro"
    GAMELAN_PELOG = "gamelan_pelog"
    RAGA = "raga"
    CUSTOM = "custom"


@dataclass
class PitchMeasurement:
    time: float
    frequency_hz: float
    cents_from_ref: float = 0.0
    confidence: float = 1.0

    @property
    def midi_cents(self) -> float:
        if self.frequency_hz <= 0:
            return 0.0
        return 1200.0 * np.log2(self.frequency_hz / 440.0) + 6900.0


@dataclass
class IntonationProfile:
    system: IntonationSystem
    deviations_cents: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_deviation: float = 0.0
    std_deviation: float = 0.0
    max_deviation: float = 0.0
    fit_score: float = 0.0


@dataclass
class TuningGrid:
    name: str = ""
    reference_hz: float = 440.0
    cents: np.ndarray = field(default_factory=lambda: np.array([]))
    octave_cents: float = 1200.0

    def frequency_at(self, degree: int, octave: int = 0) -> float:
        if degree < 0 or degree >= len(self.cents):
            return self.reference_hz
        c = self.cents[degree] + octave * self.octave_cents
        return self.reference_hz * (2.0 ** (c / 1200.0))


EQUAL_12_GRID = np.arange(12) * 100.0

JUST_RATIOS = [
    (1, 1), (16, 15), (9, 8), (6, 5), (5, 4), (4, 3),
    (45, 32), (3, 2), (8, 5), (5, 3), (9, 5), (15, 8),
]

PYTHAGOREAN_RATIOS = [
    (1, 1), (256, 243), (9, 8), (32, 27), (81, 64), (4, 3),
    (729, 512), (3, 2), (128, 81), (27, 16), (16, 9), (243, 128),
]


def ratio_to_cents_val(num: int, den: int) -> float:
    if num <= 0 or den <= 0:
        return 0.0
    return 1200.0 * np.log2(num / den)


def hz_to_cents_val(freq: float, ref: float = 440.0) -> float:
    if freq <= 0 or ref <= 0:
        return 0.0
    return 1200.0 * np.log2(freq / ref)


def cents_to_hz(cents: float, ref: float = 440.0) -> float:
    return ref * (2.0 ** (cents / 1200.0))


def build_just_grid(ref_hz: float = 440.0) -> TuningGrid:
    cents = np.array([ratio_to_cents_val(n, d) for n, d in JUST_RATIOS])
    return TuningGrid(name="Just Intonation", reference_hz=ref_hz, cents=cents)


def build_pythagorean_grid(ref_hz: float = 440.0) -> TuningGrid:
    cents = np.array([ratio_to_cents_val(n, d) for n, d in PYTHAGOREAN_RATIOS])
    return TuningGrid(name="Pythagorean", reference_hz=ref_hz, cents=cents)


def build_equal_grid(divisions: int = 12, ref_hz: float = 440.0) -> TuningGrid:
    cents = np.arange(divisions) * (1200.0 / divisions)
    return TuningGrid(name=f"{divisions}-TET", reference_hz=ref_hz, cents=cents)


def snap_to_grid(pitch_cents: float, grid: TuningGrid) -> Tuple[int, float]:
    p_mod = pitch_cents % grid.octave_cents
    if p_mod < 0:
        p_mod += grid.octave_cents
    diffs = np.abs(grid.cents - p_mod)
    wrap_diff = np.abs(grid.cents - (p_mod - grid.octave_cents))
    combined = np.minimum(diffs, wrap_diff)
    idx = int(np.argmin(combined))
    deviation = float(combined[idx])
    return idx, deviation


def measure_intonation(measurements: List[PitchMeasurement],
                       grid: TuningGrid) -> IntonationProfile:
    if not measurements:
        return IntonationProfile(system=IntonationSystem.CUSTOM)
    devs = []
    for m in measurements:
        c = hz_to_cents_val(m.frequency_hz, grid.reference_hz)
        _, dev = snap_to_grid(c, grid)
        devs.append(dev)
    devs_arr = np.array(devs)
    return IntonationProfile(
        system=IntonationSystem.CUSTOM,
        deviations_cents=devs_arr,
        mean_deviation=float(np.mean(devs_arr)),
        std_deviation=float(np.std(devs_arr)),
        max_deviation=float(np.max(devs_arr)),
        fit_score=float(max(0.0, 1.0 - np.mean(devs_arr) / 50.0)),
    )


def identify_tuning_system(measurements: List[PitchMeasurement],
                           ref_hz: float = 440.0) -> IntonationProfile:
    grids = {
        IntonationSystem.EQUAL_12: build_equal_grid(12, ref_hz),
        IntonationSystem.JUST: build_just_grid(ref_hz),
        IntonationSystem.PYTHAGOREAN: build_pythagorean_grid(ref_hz),
    }
    best = None
    for system, grid in grids.items():
        profile = measure_intonation(measurements, grid)
        profile.system = system
        if best is None or profile.fit_score > best.fit_score:
            best = profile
    return best if best else IntonationProfile(system=IntonationSystem.EQUAL_12)


def pitch_histogram(measurements: List[PitchMeasurement],
                    bin_cents: float = 10.0,
                    octave_fold: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if not measurements:
        return np.array([]), np.array([])
    cents_vals = []
    for m in measurements:
        c = hz_to_cents_val(m.frequency_hz)
        if octave_fold:
            c = c % 1200.0
        cents_vals.append(c)
    arr = np.array(cents_vals)
    if octave_fold:
        bins = np.arange(0, 1200.0 + bin_cents, bin_cents)
    else:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        bins = np.arange(lo, hi + bin_cents, bin_cents)
    hist, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, hist.astype(float)


def intonation_drift(measurements: List[PitchMeasurement],
                     grid: TuningGrid,
                     window: int = 10) -> np.ndarray:
    if len(measurements) < window:
        return np.array([])
    devs = []
    for m in measurements:
        c = hz_to_cents_val(m.frequency_hz, grid.reference_hz)
        _, dev = snap_to_grid(c, grid)
        devs.append(dev)
    devs_arr = np.array(devs)
    kernel = np.ones(window) / window
    return np.convolve(devs_arr, kernel, mode='valid')


def adaptive_reference(measurements: List[PitchMeasurement],
                       initial_ref_hz: float = 440.0,
                       alpha: float = 0.01) -> float:
    ref = initial_ref_hz
    for m in measurements:
        if m.frequency_hz > 0 and m.confidence > 0.5:
            octave_folded = m.frequency_hz
            while octave_folded > ref * 1.5:
                octave_folded /= 2.0
            while octave_folded < ref * 0.75:
                octave_folded *= 2.0
            ref = ref * (1.0 - alpha) + octave_folded * alpha
    return ref


def tuning_stability(measurements: List[PitchMeasurement],
                     grid: TuningGrid) -> float:
    if len(measurements) < 3:
        return 1.0
    devs = []
    for m in measurements:
        c = hz_to_cents_val(m.frequency_hz, grid.reference_hz)
        _, dev = snap_to_grid(c, grid)
        devs.append(dev)
    arr = np.array(devs)
    jitter = float(np.std(np.diff(arr)))
    return float(max(0.0, 1.0 - jitter / 25.0))


def comma_detection(pitches_cents: np.ndarray,
                    expected_comma_cents: float = 21.5,
                    tolerance: float = 5.0) -> List[int]:
    hits = []
    for i in range(len(pitches_cents) - 1):
        diff = abs(pitches_cents[i + 1] - pitches_cents[i])
        if abs(diff - expected_comma_cents) < tolerance:
            hits.append(i)
    return hits
