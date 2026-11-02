"""Dance-rhythm patterns by world tradition.

Each dance form is represented as a pulse pattern (onset grid), accent
pattern, and tempo range. Provides classification from onset features
and pattern matching utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DancePattern:
    name: str
    region: str
    time_signature: Tuple[int, int]  # (beats, beat_unit)
    cycle_pulses: int
    onsets: Tuple[int, ...]     # 1-indexed pulse positions with onsets
    accents: Tuple[int, ...]    # 1-indexed pulse positions with strong accents
    tempo_range_bpm: Tuple[float, float]
    shuffle_ratio: float = 1.0  # 1.0 = straight, 2.0 = triplet swing


# ── Pattern catalog ──────────────────────────────────────────────────────

WALTZ = DancePattern(
    "Waltz", "European", (3, 4), 3,
    onsets=(1, 2, 3), accents=(1,),
    tempo_range_bpm=(80, 180),
)

TANGO = DancePattern(
    "Tango", "Argentine", (4, 4), 8,
    onsets=(1, 4, 5, 8), accents=(1, 5),  # habanera
    tempo_range_bpm=(60, 72),
)

SAMBA = DancePattern(
    "Samba", "Brazilian", (2, 4), 16,
    onsets=(1, 4, 7, 9, 11, 13, 16), accents=(1, 9),
    tempo_range_bpm=(90, 110),
)

BOSSA_NOVA = DancePattern(
    "Bossa Nova", "Brazilian", (4, 4), 16,
    onsets=(1, 4, 7, 11, 14), accents=(1, 7),
    tempo_range_bpm=(120, 145),
)

REGGAE_ONE_DROP = DancePattern(
    "Reggae One-Drop", "Jamaican", (4, 4), 8,
    onsets=(3, 5, 7), accents=(3,),
    tempo_range_bpm=(60, 90),
)

REGGAE_STEPPERS = DancePattern(
    "Reggae Steppers", "Jamaican", (4, 4), 4,
    onsets=(1, 2, 3, 4), accents=(1, 3),
    tempo_range_bpm=(70, 85),
)

SWING = DancePattern(
    "Swing", "American", (4, 4), 8,
    onsets=(1, 3, 5, 7), accents=(1, 5),
    tempo_range_bpm=(120, 240),
    shuffle_ratio=1.5,
)

FLAMENCO_SOLEA = DancePattern(
    "Soleá", "Andalusian", (3, 4), 12,
    onsets=(1, 3, 4, 6, 7, 8, 10, 11, 12), accents=(3, 6, 8, 10, 12),
    tempo_range_bpm=(60, 120),
)

FLAMENCO_BULERIA = DancePattern(
    "Bulería", "Andalusian", (3, 4), 12,
    onsets=(1, 3, 4, 6, 7, 8, 10, 11, 12), accents=(3, 6, 8, 10, 12),
    tempo_range_bpm=(180, 280),
)

IRISH_JIG = DancePattern(
    "Jig", "Irish", (6, 8), 6,
    onsets=(1, 2, 3, 4, 5, 6), accents=(1, 4),
    tempo_range_bpm=(110, 130),
)

IRISH_REEL = DancePattern(
    "Reel", "Irish", (4, 4), 8,
    onsets=(1, 2, 3, 4, 5, 6, 7, 8), accents=(1, 5),
    tempo_range_bpm=(100, 130),
)

POLKA = DancePattern(
    "Polka", "Czech/Polish", (2, 4), 4,
    onsets=(1, 2, 3, 4), accents=(1, 3),
    tempo_range_bpm=(100, 140),
)

BOLERO = DancePattern(
    "Bolero", "Cuban", (4, 4), 8,
    onsets=(1, 4, 5, 7), accents=(1, 5),
    tempo_range_bpm=(80, 100),
)

MERENGUE = DancePattern(
    "Merengue", "Dominican", (2, 4), 4,
    onsets=(1, 2, 3, 4), accents=(1,),
    tempo_range_bpm=(120, 160),
)

CUMBIA = DancePattern(
    "Cumbia", "Colombian", (4, 4), 8,
    onsets=(1, 3, 5, 7, 8), accents=(1, 5),
    tempo_range_bpm=(80, 110),
)

SALSA = DancePattern(
    "Salsa", "Cuban/Puerto Rican", (4, 4), 16,
    onsets=(1, 4, 7, 9, 11, 13, 16), accents=(1, 9),
    tempo_range_bpm=(150, 250),
)

CHA_CHA = DancePattern(
    "Cha-Cha", "Cuban", (4, 4), 8,
    onsets=(1, 3, 5, 6, 7), accents=(1, 5),
    tempo_range_bpm=(110, 130),
)

RUMBA = DancePattern(
    "Rumba", "Cuban", (4, 4), 16,
    onsets=(1, 4, 7, 11, 13), accents=(1, 7),
    tempo_range_bpm=(100, 130),
)

AFROBEAT = DancePattern(
    "Afrobeat", "West African", (4, 4), 16,
    onsets=(1, 3, 5, 7, 9, 11, 13, 15), accents=(1, 5, 9, 13),
    tempo_range_bpm=(100, 130),
)

BHARATANATYAM_ADI = DancePattern(
    "Adi Tala", "Indian", (4, 4), 8,
    onsets=(1, 2, 3, 4, 5, 6, 7, 8), accents=(1, 5),
    tempo_range_bpm=(60, 180),
)


DANCE_CATALOG: Dict[str, DancePattern] = {
    "waltz": WALTZ, "tango": TANGO, "samba": SAMBA,
    "bossa_nova": BOSSA_NOVA, "reggae_one_drop": REGGAE_ONE_DROP,
    "reggae_steppers": REGGAE_STEPPERS, "swing": SWING,
    "solea": FLAMENCO_SOLEA, "buleria": FLAMENCO_BULERIA,
    "jig": IRISH_JIG, "reel": IRISH_REEL, "polka": POLKA,
    "bolero": BOLERO, "merengue": MERENGUE, "cumbia": CUMBIA,
    "salsa": SALSA, "cha_cha": CHA_CHA, "rumba": RUMBA,
    "afrobeat": AFROBEAT, "adi_tala": BHARATANATYAM_ADI,
}


# ── Utility ──────────────────────────────────────────────────────────────

def pattern_to_binary(pattern: DancePattern) -> np.ndarray:
    """Convert onset positions to a binary pulse vector."""
    vec = np.zeros(pattern.cycle_pulses, dtype=int)
    for p in pattern.onsets:
        if 1 <= p <= pattern.cycle_pulses:
            vec[p - 1] = 1
    return vec


def accent_to_weight(pattern: DancePattern) -> np.ndarray:
    """Convert accent positions to a weight vector (accented=1.0, onset=0.5, rest=0.0)."""
    w = np.zeros(pattern.cycle_pulses)
    for p in pattern.onsets:
        if 1 <= p <= pattern.cycle_pulses:
            w[p - 1] = 0.5
    for p in pattern.accents:
        if 1 <= p <= pattern.cycle_pulses:
            w[p - 1] = 1.0
    return w


def pattern_similarity(a: DancePattern, b: DancePattern) -> float:
    """Hamming-based similarity between two dance patterns (rotation-invariant)."""
    va = pattern_to_binary(a)
    vb = pattern_to_binary(b)
    lcm_len = int(np.lcm(len(va), len(vb)))
    va_rep = np.tile(va, lcm_len // len(va))
    vb_rep = np.tile(vb, lcm_len // len(vb))
    best = 0.0
    for shift in range(lcm_len):
        shifted = np.roll(vb_rep, shift)
        match = float(np.sum(va_rep == shifted)) / lcm_len
        if match > best:
            best = match
    return best


# ── Classification ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class DanceClassification:
    name: str
    confidence: float
    tempo_match: bool
    pattern_similarity: float


def classify_dance(
    onset_pattern: np.ndarray,
    tempo_bpm: float,
    top_k: int = 3,
) -> List[DanceClassification]:
    """Classify an onset pattern against the dance catalog."""
    results: List[DanceClassification] = []
    binary = (np.asarray(onset_pattern) > 0).astype(int)

    for name, dp in DANCE_CATALOG.items():
        ref = pattern_to_binary(dp)
        lcm_len = int(np.lcm(len(binary), len(ref)))
        b_rep = np.tile(binary, lcm_len // max(len(binary), 1))[:lcm_len]
        r_rep = np.tile(ref, lcm_len // max(len(ref), 1))[:lcm_len]

        best_sim = 0.0
        for shift in range(lcm_len):
            shifted = np.roll(r_rep, shift)
            sim = float(np.sum(b_rep == shifted)) / lcm_len
            best_sim = max(best_sim, sim)

        tempo_ok = dp.tempo_range_bpm[0] <= tempo_bpm <= dp.tempo_range_bpm[1]
        tempo_bonus = 0.15 if tempo_ok else 0.0
        conf = min(best_sim + tempo_bonus, 1.0)

        results.append(DanceClassification(
            name=dp.name,
            confidence=conf,
            tempo_match=tempo_ok,
            pattern_similarity=best_sim,
        ))

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results[:top_k]


def generate_onset_times(pattern: DancePattern, tempo_bpm: float,
                         n_bars: int = 1) -> np.ndarray:
    """Convert a dance pattern into onset times in seconds."""
    beat_dur = 60.0 / tempo_bpm
    pulses_per_beat = max(pattern.cycle_pulses / pattern.time_signature[0], 1)
    pulse_dur = beat_dur / pulses_per_beat
    onsets = []
    bar_dur = pattern.cycle_pulses * pulse_dur
    for bar in range(n_bars):
        bar_offset = bar * bar_dur
        for onset_pos in pattern.onsets:
            onsets.append(bar_offset + (onset_pos - 1) * pulse_dur)
    return np.array(onsets, dtype=float)


def syncopation_score(pattern: DancePattern) -> float:
    """Measure syncopation: fraction of onsets on weak pulses."""
    binary = pattern_to_binary(pattern)
    n = len(binary)
    ts_num = pattern.time_signature[0]
    pulses_per_beat = max(n // ts_num, 1)
    weak_onsets = 0
    total_onsets = int(np.sum(binary))
    if total_onsets == 0:
        return 0.0
    for i in range(n):
        if binary[i] == 1 and (i % pulses_per_beat) != 0:
            weak_onsets += 1
    return weak_onsets / total_onsets


def swing_feel_from_pattern(pattern: DancePattern) -> float:
    """Estimate swing ratio from accent pattern (1.0 = straight, 2.0 = triplet swing)."""
    acc = accent_to_weight(pattern)
    n = len(acc)
    if n < 2:
        return 1.0
    even_weight = float(np.mean(acc[0::2]))
    odd_weight = float(np.mean(acc[1::2]))
    if odd_weight < 1e-9:
        return 1.0
    return min(max(even_weight / odd_weight, 1.0), 3.0)


def tempo_in_range(pattern: DancePattern, bpm: float) -> bool:
    """Check whether bpm falls within the pattern's canonical tempo range."""
    lo, hi = pattern.tempo_range_bpm
    return lo <= bpm <= hi
