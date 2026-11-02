"""Ethnomusicological mode/scale classification: pitch-class profiling,
mode correlation, modulation detection, and microtonality scoring."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class ScaleSystem(str, Enum):
    WESTERN_DIATONIC = "western_diatonic"
    PENTATONIC = "pentatonic"
    MAQAM = "maqam"
    RAGA = "raga"
    GAMELAN_SLENDRO = "gamelan_slendro"
    GAMELAN_PELOG = "gamelan_pelog"
    BLUES = "blues"
    WHOLE_TONE = "whole_tone"
    CHROMATIC = "chromatic"
    OCTATONIC = "octatonic"


@dataclass
class ModeProfile:
    name: str
    scale_system: ScaleSystem
    intervals_cents: np.ndarray
    tonic_strength: float = 1.0


def _c(intervals: List[int]) -> np.ndarray:
    return np.array(intervals, dtype=float)


MODE_CATALOG: Dict[str, ModeProfile] = {
    "ionian": ModeProfile("Ionian", ScaleSystem.WESTERN_DIATONIC, _c([0, 200, 400, 500, 700, 900, 1100])),
    "dorian": ModeProfile("Dorian", ScaleSystem.WESTERN_DIATONIC, _c([0, 200, 300, 500, 700, 900, 1000])),
    "phrygian": ModeProfile("Phrygian", ScaleSystem.WESTERN_DIATONIC, _c([0, 100, 300, 500, 700, 800, 1000])),
    "lydian": ModeProfile("Lydian", ScaleSystem.WESTERN_DIATONIC, _c([0, 200, 400, 600, 700, 900, 1100])),
    "mixolydian": ModeProfile("Mixolydian", ScaleSystem.WESTERN_DIATONIC, _c([0, 200, 400, 500, 700, 900, 1000])),
    "aeolian": ModeProfile("Aeolian", ScaleSystem.WESTERN_DIATONIC, _c([0, 200, 300, 500, 700, 800, 1000])),
    "locrian": ModeProfile("Locrian", ScaleSystem.WESTERN_DIATONIC, _c([0, 100, 300, 500, 600, 800, 1000])),
    "major_pentatonic": ModeProfile("Major Pentatonic", ScaleSystem.PENTATONIC, _c([0, 200, 400, 700, 900])),
    "minor_pentatonic": ModeProfile("Minor Pentatonic", ScaleSystem.PENTATONIC, _c([0, 300, 500, 700, 1000])),
    "blues": ModeProfile("Blues", ScaleSystem.BLUES, _c([0, 300, 500, 600, 700, 1000])),
    "whole_tone": ModeProfile("Whole Tone", ScaleSystem.WHOLE_TONE, _c([0, 200, 400, 600, 800, 1000])),
    "octatonic_hw": ModeProfile("Octatonic H-W", ScaleSystem.OCTATONIC, _c([0, 100, 300, 400, 600, 700, 900, 1000])),
    "maqam_hijaz": ModeProfile("Maqam Hijaz", ScaleSystem.MAQAM, _c([0, 100, 400, 500, 700, 800, 1000]), 0.9),
    "maqam_rast": ModeProfile("Maqam Rast", ScaleSystem.MAQAM, _c([0, 200, 350, 500, 700, 900, 1050]), 0.9),
    "raga_bhairav": ModeProfile("Raga Bhairav", ScaleSystem.RAGA, _c([0, 100, 400, 500, 700, 800, 1100]), 0.85),
    "raga_yaman": ModeProfile("Raga Yaman", ScaleSystem.RAGA, _c([0, 200, 400, 600, 700, 900, 1100]), 0.85),
    "slendro": ModeProfile("Slendro", ScaleSystem.GAMELAN_SLENDRO, _c([0, 240, 480, 720, 960]), 0.7),
    "pelog": ModeProfile("Pelog", ScaleSystem.GAMELAN_PELOG, _c([0, 120, 270, 540, 670, 800, 1060]), 0.7),
}


def _mode_template(mode: ModeProfile, resolution: int = 1200) -> np.ndarray:
    """Build a binary template vector at cent resolution."""
    tpl = np.zeros(resolution)
    for c in mode.intervals_cents:
        idx = int(round(c)) % resolution
        tpl[idx] = 1.0
    return tpl


def interval_histogram(pitches_cents: np.ndarray, n_bins: int = 1200) -> np.ndarray:
    pc = np.mod(pitches_cents, 1200.0)
    hist, _ = np.histogram(pc, bins=n_bins, range=(0, 1200))
    return hist.astype(float)


def pitch_class_profile(pitches_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    valid = pitches_hz[pitches_hz > 0]
    if len(valid) == 0:
        return np.zeros(12)
    cents = 1200.0 * np.log2(valid / tonic_hz)
    pc = np.mod(cents, 1200.0)
    bins = np.linspace(0, 1200, 13)
    hist, _ = np.histogram(pc, bins=bins)
    return hist.astype(float)


def mode_correlation(profile: np.ndarray, mode: ModeProfile) -> float:
    if len(profile) == 12:
        tpl = np.zeros(12)
        for c in mode.intervals_cents:
            idx = int(round(c / 100.0)) % 12
            tpl[idx] = 1.0
    else:
        tpl = _mode_template(mode, len(profile))
    na = np.linalg.norm(profile)
    nb = np.linalg.norm(tpl)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(profile, tpl) / (na * nb))


def classify_mode(
    pitches_hz: np.ndarray,
    tonic_hz: float,
    catalog: Optional[Dict[str, ModeProfile]] = None,
) -> Tuple[str, float]:
    cat = catalog or MODE_CATALOG
    profile = pitch_class_profile(pitches_hz, tonic_hz)
    best_name, best_score = "", -1.0
    for name, mode in cat.items():
        s = mode_correlation(profile, mode)
        if s > best_score:
            best_score = s
            best_name = name
    return best_name, best_score


def mode_distance(a: ModeProfile, b: ModeProfile) -> float:
    ta = _mode_template(a)
    tb = _mode_template(b)
    intersection = np.sum(np.minimum(ta, tb))
    union = np.sum(np.maximum(ta, tb))
    if union < 1e-12:
        return 1.0
    return 1.0 - intersection / union


def detect_modulation(
    pitches_hz: np.ndarray,
    tonic_hz: float,
    window_size: int = 50,
    hop: int = 25,
    catalog: Optional[Dict[str, ModeProfile]] = None,
) -> List[Tuple[int, str, float]]:
    results: List[Tuple[int, str, float]] = []
    n = len(pitches_hz)
    for start in range(0, n - window_size + 1, hop):
        chunk = pitches_hz[start : start + window_size]
        name, score = classify_mode(chunk, tonic_hz, catalog)
        results.append((start, name, score))
    return results


def microtonality_score(pitches_cents: np.ndarray) -> float:
    if len(pitches_cents) == 0:
        return 0.0
    pc = np.mod(pitches_cents, 1200.0)
    nearest_semitone = np.round(pc / 100.0) * 100.0
    deviation = np.abs(pc - nearest_semitone)
    return float(np.mean(deviation > 25.0))


def interval_class_vector(mode: ModeProfile) -> np.ndarray:
    iv = mode.intervals_cents
    icv = np.zeros(7)
    for i in range(len(iv)):
        for j in range(i + 1, len(iv)):
            diff = (iv[j] - iv[i]) % 1200
            ic = min(diff, 1200 - diff)
            idx = min(int(round(ic / 100.0)), 6)
            icv[idx] += 1
    return icv
