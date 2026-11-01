"""Orchestration: instrument ranges, doublings, balance, blend, density.

Covers standard orchestral instrument ranges (Adler 2002), family grouping,
register-overlap analysis, doubling detection, blend compatibility (spectral
centroid proximity heuristic), voicing density, and sectional balance scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class InstrumentFamily(Enum):
    WOODWIND = "woodwind"
    BRASS = "brass"
    STRINGS = "strings"
    PERCUSSION = "percussion"
    KEYBOARD = "keyboard"
    VOICE = "voice"
    OTHER = "other"


@dataclass(frozen=True)
class InstrumentRange:
    name: str
    midi_low: int
    midi_high: int
    transposition: int = 0
    family: str = "other"
    spectral_centroid_hz: float = 1000.0

    @property
    def span_semitones(self) -> int:
        return self.midi_high - self.midi_low

    @property
    def family_enum(self) -> InstrumentFamily:
        try:
            return InstrumentFamily(self.family)
        except ValueError:
            return InstrumentFamily.OTHER


@dataclass(frozen=True)
class DoublingEvent:
    instrument_a: str
    instrument_b: str
    interval: int
    pitch_class_overlap: float


@dataclass(frozen=True)
class OrchestrationProfile:
    n_instruments: int
    n_families: int
    total_range_semitones: int
    voicing_density: float
    family_distribution: dict[str, float]
    mean_blend_score: float


ORCHESTRA_RANGES: dict[str, InstrumentRange] = {
    "piccolo":      InstrumentRange("piccolo", 74, 108, 12, "woodwind", 3200.0),
    "flute":        InstrumentRange("flute", 60, 96, 0, "woodwind", 2400.0),
    "oboe":         InstrumentRange("oboe", 58, 91, 0, "woodwind", 1800.0),
    "clarinet_bb":  InstrumentRange("clarinet_bb", 50, 94, 2, "woodwind", 1500.0),
    "bassoon":      InstrumentRange("bassoon", 34, 75, 0, "woodwind", 800.0),
    "french_horn":  InstrumentRange("french_horn", 34, 77, 7, "brass", 1200.0),
    "trumpet_bb":   InstrumentRange("trumpet_bb", 55, 82, 2, "brass", 2200.0),
    "trombone":     InstrumentRange("trombone", 40, 72, 0, "brass", 1000.0),
    "tuba":         InstrumentRange("tuba", 28, 58, 0, "brass", 600.0),
    "violin":       InstrumentRange("violin", 55, 103, 0, "strings", 2000.0),
    "viola":        InstrumentRange("viola", 48, 91, 0, "strings", 1400.0),
    "cello":        InstrumentRange("cello", 36, 76, 0, "strings", 900.0),
    "double_bass":  InstrumentRange("double_bass", 28, 67, 12, "strings", 500.0),
    "timpani":      InstrumentRange("timpani", 40, 55, 0, "percussion", 400.0),
    "piano":        InstrumentRange("piano", 21, 108, 0, "keyboard", 1600.0),
    "harp":         InstrumentRange("harp", 24, 103, 0, "strings", 1100.0),
}

BLEND_COMPATIBILITY: dict[tuple[str, str], float] = {
    ("woodwind", "woodwind"): 0.90,
    ("woodwind", "strings"): 0.85,
    ("woodwind", "brass"): 0.60,
    ("strings", "strings"): 0.95,
    ("strings", "brass"): 0.55,
    ("brass", "brass"): 0.88,
    ("keyboard", "strings"): 0.70,
    ("keyboard", "woodwind"): 0.65,
    ("percussion", "brass"): 0.50,
    ("percussion", "strings"): 0.40,
}


def fits_range(pitch: int, instrument: str) -> bool:
    r = ORCHESTRA_RANGES.get(instrument)
    if r is None:
        return False
    return r.midi_low <= pitch <= r.midi_high


def suggest_transposition(pitch: int, instrument: str) -> int | None:
    r = ORCHESTRA_RANGES.get(instrument)
    if r is None:
        return None
    if pitch < r.midi_low:
        octaves_up = ((r.midi_low - pitch) + 11) // 12
        return octaves_up * 12
    if pitch > r.midi_high:
        octaves_down = ((pitch - r.midi_high) + 11) // 12
        return -octaves_down * 12
    return 0


def doubling_score(pitches_a: list[int], pitches_b: list[int]) -> float:
    if not pitches_a or not pitches_b:
        return 0.0
    pcs_a = {p % 12 for p in pitches_a}
    pcs_b = {p % 12 for p in pitches_b}
    overlap = len(pcs_a & pcs_b)
    return overlap / max(len(pcs_a | pcs_b), 1)


def blend_score(inst_a: str, inst_b: str) -> float:
    """Timbral blend heuristic from family compatibility and spectral centroid distance."""
    ra = ORCHESTRA_RANGES.get(inst_a)
    rb = ORCHESTRA_RANGES.get(inst_b)
    if ra is None or rb is None:
        return 0.5
    family_pair = tuple(sorted([ra.family, rb.family]))
    family_compat = BLEND_COMPATIBILITY.get(family_pair, 0.5)
    centroid_dist = abs(math.log2(max(ra.spectral_centroid_hz, 1.0))
                        - math.log2(max(rb.spectral_centroid_hz, 1.0)))
    centroid_sim = max(0.0, 1.0 - centroid_dist / 3.0)
    return 0.6 * family_compat + 0.4 * centroid_sim


def register_overlap(inst_a: str, inst_b: str) -> int:
    """Semitones of overlapping range between two instruments."""
    ra = ORCHESTRA_RANGES.get(inst_a)
    rb = ORCHESTRA_RANGES.get(inst_b)
    if ra is None or rb is None:
        return 0
    lo = max(ra.midi_low, rb.midi_low)
    hi = min(ra.midi_high, rb.midi_high)
    return max(0, hi - lo)


def voicing_density(pitches: list[int]) -> float:
    """Ratio of pitch-class count to total span — 1.0 = maximally dense."""
    if len(pitches) < 2:
        return 0.0
    span = max(pitches) - min(pitches)
    if span == 0:
        return 1.0
    return len(set(pitches)) / (span + 1)


def detect_doublings(
    parts: dict[str, list[int]],
) -> list[DoublingEvent]:
    """Find pitch-class doublings between instrument parts."""
    events: list[DoublingEvent] = []
    instruments = list(parts.keys())
    for i in range(len(instruments)):
        for j in range(i + 1, len(instruments)):
            a, b = instruments[i], instruments[j]
            pcs_a = {p % 12 for p in parts[a]}
            pcs_b = {p % 12 for p in parts[b]}
            overlap = pcs_a & pcs_b
            if overlap:
                pc_overlap = len(overlap) / max(len(pcs_a | pcs_b), 1)
                events.append(DoublingEvent(
                    instrument_a=a,
                    instrument_b=b,
                    interval=0,
                    pitch_class_overlap=pc_overlap,
                ))
    return events


def family_energy_distribution(
    parts: dict[str, list[int]],
) -> dict[str, float]:
    """Fraction of active pitches contributed by each instrument family."""
    family_counts: dict[str, int] = {}
    total = 0
    for inst, pitches in parts.items():
        r = ORCHESTRA_RANGES.get(inst)
        fam = r.family if r else "other"
        family_counts[fam] = family_counts.get(fam, 0) + len(pitches)
        total += len(pitches)
    if total == 0:
        return {}
    return {k: v / total for k, v in family_counts.items()}


def orchestral_balance(part_assignments: dict[str, list[int]]) -> dict[str, float]:
    family_notes: dict[str, int] = {}
    total = 0
    for inst, pitches in part_assignments.items():
        r = ORCHESTRA_RANGES.get(inst)
        fam = r.family if r else "other"
        family_notes[fam] = family_notes.get(fam, 0) + len(pitches)
        total += len(pitches)
    if total == 0:
        return {}
    return {k: v / total for k, v in family_notes.items()}


def orchestral_balance_score(parts: dict[str, list[int]]) -> float:
    """0-1 score: 1.0 when families are equally represented."""
    dist = family_energy_distribution(parts)
    if not dist:
        return 0.0
    values = list(dist.values())
    n = len(values)
    if n < 2:
        return 1.0
    ideal = 1.0 / n
    deviation = sum(abs(v - ideal) for v in values)
    max_deviation = 2.0 * (1.0 - ideal)
    return max(0.0, 1.0 - deviation / max(max_deviation, 1e-9))


def compute_orchestration_profile(
    parts: dict[str, list[int]],
) -> OrchestrationProfile:
    """Aggregate orchestration metrics for a set of instrument parts."""
    all_pitches = [p for pitches in parts.values() for p in pitches]
    families = set()
    for inst in parts:
        r = ORCHESTRA_RANGES.get(inst)
        if r:
            families.add(r.family)
    total_range = (max(all_pitches) - min(all_pitches)) if all_pitches else 0
    density = voicing_density(all_pitches) if all_pitches else 0.0
    dist = family_energy_distribution(parts)
    instruments = list(parts.keys())
    blend_scores = []
    for i in range(len(instruments)):
        for j in range(i + 1, len(instruments)):
            blend_scores.append(blend_score(instruments[i], instruments[j]))
    mean_blend = float(np.mean(blend_scores)) if blend_scores else 0.0
    return OrchestrationProfile(
        n_instruments=len(parts),
        n_families=len(families),
        total_range_semitones=total_range,
        voicing_density=density,
        family_distribution=dist,
        mean_blend_score=mean_blend,
    )
