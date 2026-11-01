"""Historical instrument acoustic modeling: spectral profiles, ranges, decay, ensemble blend.

Models the acoustic characteristics of period instruments as they evolved
across historical eras.  Each instrument is represented by its pitch range,
harmonic envelope (relative amplitudes of partials), attack/decay profile,
and formant structure.  Ensemble blend analysis computes spectral overlap
and masking between instruments in a period-appropriate ensemble.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class InstrumentFamily(Enum):
    STRING_BOWED = "string_bowed"
    STRING_PLUCKED = "string_plucked"
    WOODWIND = "woodwind"
    BRASS = "brass"
    KEYBOARD = "keyboard"
    PERCUSSION = "percussion"
    VOICE = "voice"


class HistoricalEra(Enum):
    MEDIEVAL = "medieval"
    RENAISSANCE = "renaissance"
    BAROQUE = "baroque"
    CLASSICAL = "classical"
    ROMANTIC = "romantic"
    MODERN = "modern"


@dataclass
class HarmonicProfile:
    """Relative amplitudes of the first N harmonics (1-indexed, fundamental = index 0)."""
    amplitudes: np.ndarray

    @property
    def n_harmonics(self) -> int:
        return len(self.amplitudes)

    def spectral_centroid_ratio(self) -> float:
        """Spectral centroid as a multiple of the fundamental."""
        if self.amplitudes.sum() < 1e-12:
            return 1.0
        harmonics = np.arange(1, self.n_harmonics + 1, dtype=float)
        return float(np.dot(harmonics, self.amplitudes) / self.amplitudes.sum())

    def odd_even_ratio(self) -> float:
        """Ratio of odd-harmonic energy to even-harmonic energy."""
        odd = sum(a ** 2 for i, a in enumerate(self.amplitudes) if (i + 1) % 2 == 1)
        even = sum(a ** 2 for i, a in enumerate(self.amplitudes) if (i + 1) % 2 == 0)
        return odd / (even + 1e-12)


@dataclass
class DecayModel:
    """Amplitude envelope as attack + exponential decay."""
    attack_ms: float
    decay_rate: float
    sustain_level: float
    release_ms: float

    def envelope(self, duration_ms: float, sr_ms: float = 1.0) -> np.ndarray:
        """Generate amplitude envelope samples."""
        n = max(1, int(duration_ms / sr_ms))
        t = np.linspace(0, duration_ms, n)
        env = np.zeros(n)
        for i, ti in enumerate(t):
            if ti < self.attack_ms:
                env[i] = ti / max(self.attack_ms, 0.1)
            elif ti < duration_ms - self.release_ms:
                elapsed = ti - self.attack_ms
                env[i] = self.sustain_level + (1.0 - self.sustain_level) * math.exp(
                    -self.decay_rate * elapsed / 1000.0)
            else:
                remaining = duration_ms - ti
                frac = remaining / max(self.release_ms, 0.1)
                elapsed = ti - self.attack_ms
                base = self.sustain_level + (1.0 - self.sustain_level) * math.exp(
                    -self.decay_rate * elapsed / 1000.0)
                env[i] = base * frac
        return env


@dataclass
class FormantSpec:
    """Formant frequencies and bandwidths for resonant instruments/voice."""
    center_hz: List[float]
    bandwidth_hz: List[float]

    def response_at(self, freq: float) -> float:
        """Combined formant filter response at a given frequency."""
        total = 0.0
        for c, bw in zip(self.center_hz, self.bandwidth_hz):
            x = (freq - c) / (bw / 2.0 + 1e-6)
            total += 1.0 / (1.0 + x ** 2)
        return total


@dataclass
class HistoricalInstrument:
    """Acoustic model of a historical instrument."""
    name: str
    era: HistoricalEra
    family: InstrumentFamily
    lowest_midi: int
    highest_midi: int
    reference_pitch_hz: float
    harmonic_profile: HarmonicProfile
    decay: DecayModel
    formants: Optional[FormantSpec] = None
    description: str = ""

    @property
    def range_semitones(self) -> int:
        return self.highest_midi - self.lowest_midi

    def frequency_range(self) -> Tuple[float, float]:
        lo = self.reference_pitch_hz * 2 ** ((self.lowest_midi - 69) / 12.0)
        hi = self.reference_pitch_hz * 2 ** ((self.highest_midi - 69) / 12.0)
        return (lo, hi)

    def synthesize_note(self, midi_note: int, duration_ms: float = 500.0,
                        sample_rate: float = 44100.0) -> np.ndarray:
        """Synthesize a single note using the instrument's harmonic profile."""
        freq = self.reference_pitch_hz * 2 ** ((midi_note - 69) / 12.0)
        n_samples = int(duration_ms * sample_rate / 1000.0)
        t = np.arange(n_samples) / sample_rate
        signal = np.zeros(n_samples)
        for k, amp in enumerate(self.harmonic_profile.amplitudes):
            h_freq = freq * (k + 1)
            if h_freq > sample_rate / 2:
                break
            signal += amp * np.sin(2 * np.pi * h_freq * t)
        env = self.decay.envelope(duration_ms, sr_ms=duration_ms / n_samples)
        if len(env) != n_samples:
            env = np.interp(np.linspace(0, 1, n_samples),
                            np.linspace(0, 1, len(env)), env)
        signal *= env
        if self.formants is not None:
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
            for i, f in enumerate(freqs):
                fft[i] *= self.formants.response_at(f)
            signal = np.fft.irfft(fft, n=n_samples)
        peak = np.max(np.abs(signal))
        if peak > 1e-10:
            signal /= peak
        return signal


# ── Canonical instrument database ────────────────────────────────

def _harmonic_decay(n: int, rate: float) -> HarmonicProfile:
    """Harmonics decaying as 1/(k^rate) for k=1..n."""
    amps = np.array([1.0 / ((k + 1) ** rate) for k in range(n)])
    return HarmonicProfile(amps / amps[0])


def _odd_harmonics(n: int, rate: float) -> HarmonicProfile:
    """Only odd harmonics, decaying as 1/(k^rate)."""
    amps = np.zeros(n)
    for k in range(n):
        if (k + 1) % 2 == 1:
            amps[k] = 1.0 / ((k + 1) ** rate)
    return HarmonicProfile(amps / max(amps[0], 1e-10))


INSTRUMENT_CATALOG: Dict[str, HistoricalInstrument] = {
    "baroque_violin": HistoricalInstrument(
        "Baroque Violin", HistoricalEra.BAROQUE, InstrumentFamily.STRING_BOWED,
        55, 93, 415.0, _harmonic_decay(12, 0.8),
        DecayModel(30, 0.5, 0.7, 100),
        description="Gut strings, lower tension, shorter neck. Warmer, less projecting."
    ),
    "classical_violin": HistoricalInstrument(
        "Classical Violin", HistoricalEra.CLASSICAL, InstrumentFamily.STRING_BOWED,
        55, 96, 430.0, _harmonic_decay(14, 0.7),
        DecayModel(25, 0.4, 0.75, 80),
        description="Transitional setup. Chin rest introduced."
    ),
    "romantic_violin": HistoricalInstrument(
        "Romantic Violin", HistoricalEra.ROMANTIC, InstrumentFamily.STRING_BOWED,
        55, 100, 440.0, _harmonic_decay(16, 0.6),
        DecayModel(20, 0.3, 0.8, 60),
        description="Steel E string, modern chinrest/shoulder rest. Brilliant, projecting."
    ),
    "baroque_oboe": HistoricalInstrument(
        "Baroque Oboe", HistoricalEra.BAROQUE, InstrumentFamily.WOODWIND,
        58, 84, 415.0, _odd_harmonics(10, 0.9),
        DecayModel(15, 1.0, 0.6, 50),
        description="Two or three keys. Reedy, transparent."
    ),
    "classical_clarinet": HistoricalInstrument(
        "Classical Clarinet", HistoricalEra.CLASSICAL, InstrumentFamily.WOODWIND,
        50, 89, 430.0, _odd_harmonics(12, 1.0),
        DecayModel(20, 0.8, 0.65, 60),
        description="Five-key clarinet. Hollow chalumeau, bright clarion."
    ),
    "natural_horn": HistoricalInstrument(
        "Natural Horn", HistoricalEra.CLASSICAL, InstrumentFamily.BRASS,
        36, 77, 430.0, _harmonic_decay(8, 0.5),
        DecayModel(40, 0.6, 0.5, 120),
        description="No valves. Hand-stopping for chromatic notes."
    ),
    "valve_horn": HistoricalInstrument(
        "Valve Horn", HistoricalEra.ROMANTIC, InstrumentFamily.BRASS,
        34, 82, 440.0, _harmonic_decay(10, 0.5),
        DecayModel(35, 0.5, 0.55, 100),
        description="Rotary or piston valves. Full chromatic range."
    ),
    "harpsichord": HistoricalInstrument(
        "Harpsichord", HistoricalEra.BAROQUE, InstrumentFamily.KEYBOARD,
        29, 89, 415.0, _harmonic_decay(10, 1.2),
        DecayModel(5, 3.0, 0.1, 200),
        description="Plucked strings, no dynamics. Bright, percussive attack."
    ),
    "fortepiano": HistoricalInstrument(
        "Fortepiano", HistoricalEra.CLASSICAL, InstrumentFamily.KEYBOARD,
        28, 89, 430.0, _harmonic_decay(12, 1.0),
        DecayModel(10, 2.0, 0.15, 300),
        description="Lighter action than modern piano. Leather-covered hammers."
    ),
    "romantic_piano": HistoricalInstrument(
        "Romantic Piano", HistoricalEra.ROMANTIC, InstrumentFamily.KEYBOARD,
        21, 108, 440.0, _harmonic_decay(16, 0.9),
        DecayModel(8, 1.2, 0.2, 500),
        description="Iron frame, cross-strung, felt hammers. Extended range and power."
    ),
    "viol": HistoricalInstrument(
        "Bass Viol (Viola da Gamba)", HistoricalEra.RENAISSANCE, InstrumentFamily.STRING_BOWED,
        36, 72, 466.0, _harmonic_decay(10, 0.9),
        DecayModel(40, 0.8, 0.6, 150),
        description="Fretted, held between knees. Intimate, blending tone."
    ),
    "lute": HistoricalInstrument(
        "Renaissance Lute", HistoricalEra.RENAISSANCE, InstrumentFamily.STRING_PLUCKED,
        40, 76, 466.0, _harmonic_decay(8, 1.5),
        DecayModel(5, 4.0, 0.05, 100),
        description="Gut strings, courses. Rapid decay, warm pluck."
    ),
    "baroque_trumpet": HistoricalInstrument(
        "Baroque Trumpet", HistoricalEra.BAROQUE, InstrumentFamily.BRASS,
        60, 84, 415.0, _harmonic_decay(8, 0.4),
        DecayModel(25, 0.7, 0.55, 80),
        description="Natural trumpet. Heroic, brilliant in the clarino register."
    ),
    "recorder": HistoricalInstrument(
        "Recorder (Alto)", HistoricalEra.RENAISSANCE, InstrumentFamily.WOODWIND,
        65, 91, 466.0, _odd_harmonics(6, 1.5),
        DecayModel(10, 1.5, 0.5, 40),
        description="End-blown flute. Sweet, gentle, limited dynamics."
    ),
}


def instruments_by_era(era: HistoricalEra) -> List[HistoricalInstrument]:
    """Return all catalogued instruments from a given era."""
    return [inst for inst in INSTRUMENT_CATALOG.values() if inst.era == era]


def instruments_by_family(family: InstrumentFamily) -> List[HistoricalInstrument]:
    """Return all catalogued instruments from a given family."""
    return [inst for inst in INSTRUMENT_CATALOG.values() if inst.family == family]


# ── Spectral overlap and ensemble blend ──────────────────────────

def spectral_overlap(inst_a: HistoricalInstrument,
                     inst_b: HistoricalInstrument,
                     n_harmonics: int = 8) -> float:
    """Measure spectral overlap between two instruments in their shared range.

    Returns a value in [0, 1] where 1 = identical harmonic profiles
    and 0 = no overlap.
    """
    n = min(n_harmonics,
            inst_a.harmonic_profile.n_harmonics,
            inst_b.harmonic_profile.n_harmonics)
    a = inst_a.harmonic_profile.amplitudes[:n]
    b = inst_b.harmonic_profile.amplitudes[:n]
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def pitch_range_overlap(inst_a: HistoricalInstrument,
                        inst_b: HistoricalInstrument) -> float:
    """Fraction of shared MIDI range relative to the larger instrument's range."""
    lo = max(inst_a.lowest_midi, inst_b.lowest_midi)
    hi = min(inst_a.highest_midi, inst_b.highest_midi)
    overlap = max(0, hi - lo)
    max_range = max(inst_a.range_semitones, inst_b.range_semitones, 1)
    return overlap / max_range


def ensemble_blend_matrix(instruments: List[HistoricalInstrument]) -> np.ndarray:
    """N×N matrix of spectral blend scores for an ensemble.

    High values indicate instruments that will fuse (blend) well;
    low values indicate timbral independence (useful for clarity).
    """
    n = len(instruments)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sp_ov = spectral_overlap(instruments[i], instruments[j])
            pr_ov = pitch_range_overlap(instruments[i], instruments[j])
            blend = 0.6 * sp_ov + 0.4 * pr_ov
            mat[i, j] = blend
            mat[j, i] = blend
        mat[i, i] = 1.0
    return mat


def ensemble_spectral_diversity(instruments: List[HistoricalInstrument]) -> float:
    """Measure how diverse an ensemble is spectrally.

    Returns 1 − mean off-diagonal blend, so diverse = high.
    """
    if len(instruments) < 2:
        return 0.0
    mat = ensemble_blend_matrix(instruments)
    n = len(instruments)
    off_diag = (mat.sum() - n) / (n * (n - 1))
    return 1.0 - off_diag


# ── Timbral evolution tracking ───────────────────────────────────

def timbral_evolution(instrument_name_prefix: str) -> List[Tuple[str, float]]:
    """Track how an instrument's spectral centroid evolved across eras.

    Matches instruments by name prefix (e.g. "baroque_violin", "classical_violin",
    "romantic_violin" for prefix "violin").
    """
    matches = []
    for key, inst in INSTRUMENT_CATALOG.items():
        if instrument_name_prefix.lower() in key.lower():
            sc = inst.harmonic_profile.spectral_centroid_ratio()
            matches.append((inst.name, sc))
    return sorted(matches, key=lambda x: x[1])


def era_average_brightness(era: HistoricalEra) -> float:
    """Average spectral centroid ratio across all instruments of an era."""
    insts = instruments_by_era(era)
    if not insts:
        return 1.0
    centroids = [i.harmonic_profile.spectral_centroid_ratio() for i in insts]
    return float(np.mean(centroids))


def masking_potential(loud_inst: HistoricalInstrument,
                      soft_inst: HistoricalInstrument,
                      midi_note: int = 60) -> float:
    """Estimate how much loud_inst masks soft_inst at a given pitch.

    Based on spectral overlap weighted by the loud instrument's energy
    in the shared harmonic region. Returns 0–1 (1 = fully masked).
    """
    freq = loud_inst.reference_pitch_hz * 2 ** ((midi_note - 69) / 12.0)
    n = min(loud_inst.harmonic_profile.n_harmonics,
            soft_inst.harmonic_profile.n_harmonics)

    masking = 0.0
    soft_total = 0.0
    for k in range(n):
        h_freq = freq * (k + 1)
        loud_amp = loud_inst.harmonic_profile.amplitudes[k]
        soft_amp = soft_inst.harmonic_profile.amplitudes[k]
        spread = 0.1 * h_freq
        mask_factor = loud_amp ** 2 / (loud_amp ** 2 + soft_amp ** 2 + 1e-12)
        masking += soft_amp * mask_factor
        soft_total += soft_amp
    return float(masking / (soft_total + 1e-12))
