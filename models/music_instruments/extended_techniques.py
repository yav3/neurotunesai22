"""Extended instrumental techniques: harmonics, multiphonics, prepared instruments.

Models non-standard playing techniques across instrument families, including
spectral properties, difficulty estimation, and notation symbols.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class InstrumentFamily(Enum):
    STRINGS = "strings"
    WOODWINDS = "woodwinds"
    BRASS = "brass"
    PERCUSSION = "percussion"
    KEYBOARD = "keyboard"
    VOICE = "voice"
    ELECTRONIC = "electronic"


class TechniqueCategory(Enum):
    HARMONICS = "harmonics"
    MULTIPHONICS = "multiphonics"
    EXTENDED_BOW = "extended_bow"
    PREPARED = "prepared"
    MUTING = "muting"
    PERCUSSIVE = "percussive"
    MICROTONAL = "microtonal"
    BREATH = "breath"
    VOCAL_FOLD = "vocal_fold"


@dataclass(frozen=True)
class ExtendedTechnique:
    name: str
    family: InstrumentFamily
    category: TechniqueCategory
    difficulty: float  # 0-1
    spectral_brightness: float  # relative to normal playing
    dynamic_range_db: float
    notation_symbol: str
    description: str


@dataclass(frozen=True)
class HarmonicNode:
    """A natural or artificial harmonic on a string instrument."""
    string_number: int
    node_position: float  # fraction of string length (1/2, 1/3, 1/4, ...)
    sounding_pitch_midi: float
    fundamental_midi: float
    harmonic_number: int
    amplitude_ratio: float  # relative to fundamental

    @property
    def frequency_ratio(self) -> float:
        return self.harmonic_number

    @property
    def cents_from_equal_temperament(self) -> float:
        just_ratio = self.harmonic_number
        equal_semitones = 12 * math.log2(just_ratio)
        nearest_semitone = round(equal_semitones)
        return (equal_semitones - nearest_semitone) * 100


def natural_harmonic_series(fundamental_hz: float, n_harmonics: int = 16) -> np.ndarray:
    """Return frequencies of the first n harmonics of a fundamental."""
    return fundamental_hz * np.arange(1, n_harmonics + 1, dtype=np.float64)


def harmonic_amplitude_model(
    harmonic_numbers: np.ndarray,
    decay_rate: float = 1.0,
    model: str = "inverse",
) -> np.ndarray:
    """Model amplitude decay across harmonic series.

    Models:
      - "inverse": A_n = 1/n^decay_rate (typical string)
      - "sawtooth": A_n = 1/n (ideal sawtooth)
      - "square": A_n = 1/n for odd n, 0 for even
      - "triangle": A_n = 1/n^2 for odd n, 0 for even
    """
    n = harmonic_numbers.astype(np.float64)
    if model == "inverse":
        return 1.0 / np.power(n, decay_rate)
    elif model == "sawtooth":
        return 1.0 / n
    elif model == "square":
        amps = np.where(harmonic_numbers % 2 == 1, 1.0 / n, 0.0)
        return amps
    elif model == "triangle":
        amps = np.where(harmonic_numbers % 2 == 1, 1.0 / (n ** 2), 0.0)
        return amps
    return 1.0 / np.power(n, decay_rate)


def string_harmonic_nodes(
    string_length_m: float,
    fundamental_hz: float,
    max_harmonic: int = 8,
    fundamental_midi: float = 55.0,
) -> list[HarmonicNode]:
    """Compute all natural harmonic nodes on a string."""
    nodes: list[HarmonicNode] = []
    for n in range(2, max_harmonic + 1):
        for k in range(1, n):
            position = k / n
            sounding_hz = fundamental_hz * n
            sounding_midi = 69 + 12 * math.log2(sounding_hz / 440.0)
            amp = 1.0 / n
            nodes.append(HarmonicNode(
                string_number=1,
                node_position=position,
                sounding_pitch_midi=sounding_midi,
                fundamental_midi=fundamental_midi,
                harmonic_number=n,
                amplitude_ratio=amp,
            ))
    return nodes


@dataclass
class MultiphonicSpectrum:
    """Spectral description of a wind multiphonic."""
    fingering: str
    pitches_hz: list[float]
    amplitudes: list[float]
    roughness: float  # sensory dissonance measure
    stability: float  # 0=unstable, 1=stable

    @property
    def n_pitches(self) -> int:
        return len(self.pitches_hz)

    def spectral_spread_hz(self) -> float:
        if len(self.pitches_hz) < 2:
            return 0.0
        return max(self.pitches_hz) - min(self.pitches_hz)


def multiphonic_roughness(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Compute Sethares-style roughness for a multiphonic spectrum.

    Uses the Plomp-Levelt model: R = sum_pairs A_i * A_j * g(|f_i - f_j|)
    where g is a function of the critical bandwidth.
    """
    if len(freqs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            diff = abs(freqs[i] - freqs[j])
            f_avg = (freqs[i] + freqs[j]) / 2.0
            cb = 1.72 * (f_avg ** 0.65)
            s = diff / (0.24 * cb)
            d = math.exp(-3.5 * s) - math.exp(-5.75 * s)
            total += amps[i] * amps[j] * max(d, 0.0)
    return total


@dataclass
class PreparedPianoModification:
    """A modification to a piano string for prepared piano."""
    note_midi: int
    material: str  # "bolt", "screw", "rubber", "felt", "eraser"
    position_ratio: float  # where on string (0=damper, 1=bridge)
    effect_on_harmonics: dict[int, float] = field(default_factory=dict)

    def modified_spectrum(
        self,
        fundamental_hz: float,
        n_harmonics: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = natural_harmonic_series(fundamental_hz, n_harmonics)
        amps = harmonic_amplitude_model(
            np.arange(1, n_harmonics + 1),
            decay_rate=1.0,
        )
        for h_num, factor in self.effect_on_harmonics.items():
            if 1 <= h_num <= n_harmonics:
                amps[h_num - 1] *= factor
        node_harmonic = round(1.0 / self.position_ratio) if self.position_ratio > 0 else 0
        if node_harmonic > 0:
            for h in range(1, n_harmonics + 1):
                if h % node_harmonic == 0:
                    amps[h - 1] *= 1.5
                else:
                    amps[h - 1] *= 0.6
        return freqs, amps


def technique_difficulty_score(
    technique: ExtendedTechnique,
    performer_level: float = 0.5,
) -> float:
    """Estimate difficulty of a technique for a performer at a given level (0-1)."""
    base = technique.difficulty
    adjusted = base / max(performer_level, 0.1)
    return min(adjusted, 1.0)


COMMON_TECHNIQUES: list[ExtendedTechnique] = [
    ExtendedTechnique("sul ponticello", InstrumentFamily.STRINGS, TechniqueCategory.EXTENDED_BOW,
                      0.3, 1.8, 15.0, "s.p.", "Bow near the bridge for metallic tone"),
    ExtendedTechnique("col legno", InstrumentFamily.STRINGS, TechniqueCategory.EXTENDED_BOW,
                      0.2, 0.4, 10.0, "c.l.", "Strike string with wood of bow"),
    ExtendedTechnique("flutter tongue", InstrumentFamily.WOODWINDS, TechniqueCategory.BREATH,
                      0.4, 1.3, 12.0, "flz.", "Roll tongue while playing"),
    ExtendedTechnique("stopped horn", InstrumentFamily.BRASS, TechniqueCategory.MUTING,
                      0.5, 0.7, 8.0, "+", "Insert hand fully into bell"),
    ExtendedTechnique("prepared piano bolt", InstrumentFamily.KEYBOARD, TechniqueCategory.PREPARED,
                      0.1, 0.5, 20.0, "prep.", "Bolt between strings"),
]
