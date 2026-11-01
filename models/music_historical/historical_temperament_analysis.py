"""Extended historical temperament analysis: well-temperaments, key color, wolf intervals.

Complements tuning_systems.py (basic ratio/cent computations) with:
- Additional historical temperaments (Kirnberger III, Vallotti, Young II)
- Arbitrary well-temperament construction from fifth-narrowing specifications
- Key-color analysis: how intervals in each key deviate from just/equal
- Wolf-interval detection (intervals exceeding a roughness threshold)
- Temperament comparison matrices across all keys
- Adaptive just intonation tracking for real-time retuning
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


PURE_FIFTH_CENTS = 1200.0 * math.log2(3.0 / 2.0)  # 701.955
PYTHAGOREAN_COMMA_CENTS = 12 * PURE_FIFTH_CENTS - 7 * 1200.0  # ~23.46

FIFTH_CHAIN_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

NOTE_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F",
              "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]

JUST_INTERVALS = {
    0: (1, 1),
    1: (16, 15),
    2: (9, 8),
    3: (6, 5),
    4: (5, 4),
    5: (4, 3),
    6: (45, 32),
    7: (3, 2),
    8: (8, 5),
    9: (5, 3),
    10: (9, 5),
    11: (15, 8),
}


def _build_well_temperament(narrowings_quarter_comma: np.ndarray) -> np.ndarray:
    """Build a 12-note temperament from fifth-narrowing specifications.

    narrowings_quarter_comma[i] = fraction of Pythagorean comma removed
    from the i-th fifth in the chain C-G-D-A-E-B-F#-C#-G#-D#-A#-F.
    Must sum to 1.0 for a closed system (all 12 fifths circling back).
    Returns cent deviations from 12-TET for each pitch class.
    """
    cents_from_c = np.zeros(12)
    current = 0.0
    for step in range(12):
        idx = FIFTH_CHAIN_ORDER[step]
        cents_from_c[idx] = current % 1200.0
        narrowing = narrowings_quarter_comma[step] * PYTHAGOREAN_COMMA_CENTS
        current += PURE_FIFTH_CENTS - narrowing
    tet = np.arange(12) * 100.0
    deviation = cents_from_c - tet
    deviation = ((deviation + 600) % 1200) - 600
    return deviation


# ── Named historical temperaments ─────────────────────────────────

def kirnberger_iii_cents() -> np.ndarray:
    """Kirnberger III (1779).

    C-G pure, G-D pure, D-A narrowed by schisma (~1/12 PC),
    A-E narrowed by syntonic comma minus schisma (~1/4 PC − schisma).
    Simplified: distributes comma across 4 fifths: C-G, G-D, D-A, A-E
    with weights [1/4, 1/4, 1/4, 1/4] of the Pythagorean comma.
    """
    n = np.zeros(12)
    n[0] = 0.25  # C-G
    n[1] = 0.25  # G-D
    n[2] = 0.25  # D-A
    n[3] = 0.25  # A-E
    return _build_well_temperament(n)


def vallotti_cents() -> np.ndarray:
    """Vallotti temperament (Francesco Vallotti, ~1754).

    Six fifths (F-C-G-D-A-E-B) each narrowed by 1/6 Pythagorean comma;
    remaining six fifths are pure.
    """
    n = np.zeros(12)
    for i in range(6):
        idx = (11 + i) % 12  # F is the 11th in chain order: F-C-G-D-A-E-B
        n[idx] = 1.0 / 6.0
    # Actually the chain is C-G-D-A-E-B-F#-..., so fifths 0..5 are C-G-D-A-E-B
    n = np.zeros(12)
    for i in range(6):
        n[i] = 1.0 / 6.0
    return _build_well_temperament(n)


def young_ii_cents() -> np.ndarray:
    """Young's second temperament (Thomas Young, 1800).

    Distributes the comma: 1/6 each on C-G, G-D, D-A, A-E, E-B, B-F#.
    """
    n = np.zeros(12)
    for i in range(6):
        n[i] = 1.0 / 6.0
    return _build_well_temperament(n)


def equal_temperament_cents() -> np.ndarray:
    """Equal temperament: zero deviation everywhere."""
    return np.zeros(12)


NAMED_TEMPERAMENTS: Dict[str, np.ndarray] = {}


def _init_named():
    from .tuning_systems import pythagorean_cents, meantone_cents, werckmeister_iii_cents
    NAMED_TEMPERAMENTS["pythagorean"] = pythagorean_cents()
    NAMED_TEMPERAMENTS["meantone_quarter_comma"] = meantone_cents()
    NAMED_TEMPERAMENTS["werckmeister_iii"] = werckmeister_iii_cents()
    NAMED_TEMPERAMENTS["kirnberger_iii"] = kirnberger_iii_cents()
    NAMED_TEMPERAMENTS["vallotti"] = vallotti_cents()
    NAMED_TEMPERAMENTS["young_ii"] = young_ii_cents()
    NAMED_TEMPERAMENTS["equal"] = equal_temperament_cents()


def get_temperament(name: str) -> np.ndarray:
    """Return cent deviations for a named temperament."""
    if not NAMED_TEMPERAMENTS:
        _init_named()
    if name not in NAMED_TEMPERAMENTS:
        raise ValueError(f"Unknown temperament: {name}. "
                         f"Available: {list(NAMED_TEMPERAMENTS.keys())}")
    return NAMED_TEMPERAMENTS[name].copy()


# ── Key-color analysis ────────────────────────────────────────────

@dataclass
class KeyColor:
    """Characterizes how a key 'sounds' in a given temperament."""
    root: int
    temperament: str
    major_third_deviation: float
    perfect_fifth_deviation: float
    minor_third_deviation: float
    tritone_deviation: float
    overall_brightness: float
    purity_score: float


def key_color(root: int, temperament_cents: np.ndarray) -> KeyColor:
    """Analyze how a specific key sounds in a given temperament.

    Computes deviations of the main consonant intervals from their
    just-intonation targets, giving a 'color' or 'affect' for the key.
    """
    def _interval_dev(semitones: int) -> float:
        actual_cents = ((semitones * 100.0 +
                        temperament_cents[(root + semitones) % 12] -
                        temperament_cents[root]) % 1200.0)
        n, d = JUST_INTERVALS.get(semitones % 12, (0, 0))
        if d == 0:
            return 0.0
        just_cents = 1200.0 * math.log2(n / d)
        diff = actual_cents - just_cents
        if diff > 600:
            diff -= 1200
        elif diff < -600:
            diff += 1200
        return diff

    m3 = _interval_dev(4)
    p5 = _interval_dev(7)
    mn3 = _interval_dev(3)
    tri = _interval_dev(6)

    brightness = m3
    deviations = [abs(m3), abs(p5), abs(mn3)]
    purity = max(0.0, 1.0 - sum(deviations) / (3 * 22.0))

    return KeyColor(
        root=root,
        temperament="custom",
        major_third_deviation=m3,
        perfect_fifth_deviation=p5,
        minor_third_deviation=mn3,
        tritone_deviation=tri,
        overall_brightness=brightness,
        purity_score=purity,
    )


def all_key_colors(temperament_cents: np.ndarray) -> List[KeyColor]:
    """Compute key-color profiles for all 12 keys."""
    return [key_color(root, temperament_cents) for root in range(12)]


def key_ranking_by_purity(temperament_cents: np.ndarray) -> List[Tuple[int, float]]:
    """Rank keys from purest to most tempered."""
    colors = all_key_colors(temperament_cents)
    ranked = sorted([(c.root, c.purity_score) for c in colors],
                    key=lambda x: -x[1])
    return ranked


# ── Wolf-interval detection ──────────────────────────────────────

@dataclass
class WolfInterval:
    """An interval that deviates significantly from its just target."""
    note_a: int
    note_b: int
    interval_semitones: int
    deviation_cents: float
    just_target_cents: float
    actual_cents: float


def detect_wolf_intervals(temperament_cents: np.ndarray,
                          threshold_cents: float = 15.0) -> List[WolfInterval]:
    """Find intervals whose deviation from just exceeds a threshold.

    Default threshold 15 cents corresponds to roughly audible beating
    in sustained intervals.
    """
    wolves = []
    consonant_intervals = [3, 4, 5, 7, 8, 9]
    for root in range(12):
        for interval in consonant_intervals:
            target_note = (root + interval) % 12
            actual = ((interval * 100.0 + temperament_cents[target_note] -
                      temperament_cents[root]) % 1200.0)
            n, d = JUST_INTERVALS.get(interval, (0, 0))
            if d == 0:
                continue
            just = 1200.0 * math.log2(n / d)
            dev = actual - just
            if dev > 600:
                dev -= 1200
            elif dev < -600:
                dev += 1200
            if abs(dev) > threshold_cents:
                wolves.append(WolfInterval(
                    note_a=root,
                    note_b=target_note,
                    interval_semitones=interval,
                    deviation_cents=dev,
                    just_target_cents=just,
                    actual_cents=actual,
                ))
    return wolves


def wolf_fifth_location(temperament_cents: np.ndarray) -> Optional[Tuple[int, int, float]]:
    """Find the worst fifth (the 'wolf') in a temperament.

    Returns (note_a, note_b, deviation_cents) or None if all fifths
    are within 5 cents of pure.
    """
    worst_dev = 0.0
    worst_pair = None
    for root in range(12):
        target = (root + 7) % 12
        actual = ((700.0 + temperament_cents[target] -
                  temperament_cents[root]) % 1200.0)
        dev = actual - PURE_FIFTH_CENTS
        if dev > 600:
            dev -= 1200
        elif dev < -600:
            dev += 1200
        if abs(dev) > abs(worst_dev):
            worst_dev = dev
            worst_pair = (root, target)
    if worst_pair and abs(worst_dev) > 5.0:
        return (*worst_pair, worst_dev)
    return None


# ── Temperament comparison matrix ─────────────────────────────────

def temperament_distance_matrix(temperaments: Dict[str, np.ndarray]) -> np.ndarray:
    """RMS cent distance between every pair of temperaments.

    Returns an N×N symmetric matrix where entry (i,j) is the RMS
    of the 12 per-note cent differences.
    """
    names = sorted(temperaments.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = temperaments[names[i]] - temperaments[names[j]]
            rms = float(np.sqrt(np.mean(diff ** 2)))
            mat[i, j] = rms
            mat[j, i] = rms
    return mat


def temperament_key_affinity(temperament_cents: np.ndarray) -> np.ndarray:
    """12×12 matrix: how well each interval sounds in each key.

    Entry (root, interval) = purity score ∈ [0,1] where 1 = just.
    """
    affinity = np.zeros((12, 12))
    for root in range(12):
        for interval in range(12):
            n, d = JUST_INTERVALS.get(interval, (0, 0))
            if d == 0:
                affinity[root, interval] = 1.0
                continue
            just = 1200.0 * math.log2(n / d)
            actual = ((interval * 100.0 + temperament_cents[(root + interval) % 12] -
                      temperament_cents[root]) % 1200.0)
            dev = abs(actual - just)
            if dev > 600:
                dev = 1200 - dev
            affinity[root, interval] = max(0.0, 1.0 - dev / 22.0)
    return affinity


# ── Adaptive just intonation ──────────────────────────────────────

@dataclass
class AdaptiveState:
    """State for adaptive just intonation retuning."""
    reference_pitch: int
    current_offsets: np.ndarray
    history: List[np.ndarray] = field(default_factory=list)
    drift_cents: float = 0.0


def adaptive_just_retune(chord_pcs: List[int],
                         state: Optional[AdaptiveState] = None,
                         drift_limit: float = 20.0) -> AdaptiveState:
    """Compute adaptive just intonation offsets for a chord.

    Tunes the chord's intervals to just ratios relative to the root,
    while tracking cumulative drift from the original reference pitch.
    If drift exceeds drift_limit, compromises toward equal temperament.
    """
    if not chord_pcs:
        if state is None:
            return AdaptiveState(0, np.zeros(12))
        return state

    root = chord_pcs[0] % 12
    offsets = np.zeros(12)

    if state is not None:
        offsets[:] = state.current_offsets
        drift = state.drift_cents
    else:
        drift = 0.0

    for pc in chord_pcs:
        interval = (pc - root) % 12
        n, d = JUST_INTERVALS.get(interval, (0, 0))
        if d == 0 or interval == 0:
            continue
        just_cents = 1200.0 * math.log2(n / d)
        target_offset = just_cents - interval * 100.0

        if abs(drift + target_offset) > drift_limit:
            blend = max(0.0, 1.0 - abs(drift) / drift_limit)
            target_offset *= blend

        offsets[pc % 12] = target_offset

    new_drift = drift + offsets[root]

    new_state = AdaptiveState(
        reference_pitch=root,
        current_offsets=offsets.copy(),
        drift_cents=new_drift,
    )
    if state is not None:
        new_state.history = state.history + [offsets.copy()]
    else:
        new_state.history = [offsets.copy()]
    return new_state


def adaptive_drift_trajectory(chord_sequence: List[List[int]],
                              drift_limit: float = 20.0) -> np.ndarray:
    """Track cumulative pitch drift across a chord sequence under adaptive JI.

    Returns array of drift values (in cents) per chord.
    """
    state = None
    drifts = []
    for chord in chord_sequence:
        state = adaptive_just_retune(chord, state, drift_limit)
        drifts.append(state.drift_cents)
    return np.array(drifts)
