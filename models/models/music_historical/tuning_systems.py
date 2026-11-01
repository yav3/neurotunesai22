"""Historical tuning systems: Pythagorean, meantone, well-temperament, just intonation.

Implements frequency-ratio computations for period-specific tuning systems used
before equal temperament became standard (~1850).  Each system returns cent
deviations from 12-TET so the results can be compared directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SEMITONE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

# ── Pythagorean tuning ──────────────────────────────────────────────

def pythagorean_ratios() -> np.ndarray:
    """12-note Pythagorean tuning via the spiral of pure fifths (3:2).

    Starting from C, stacks perfect fifths and octave-reduces.
    Returns 12 frequency ratios relative to the tonic.
    """
    ratios = np.ones(12)
    fifth = 3.0 / 2.0
    order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    for i, idx in enumerate(order):
        r = fifth ** i
        while r >= 2.0:
            r /= 2.0
        ratios[idx] = r
    return ratios


def pythagorean_cents() -> np.ndarray:
    """Cent deviations of Pythagorean tuning from 12-TET."""
    ratios = pythagorean_ratios()
    tet_ratios = 2.0 ** (np.arange(12) / 12.0)
    return 1200.0 * np.log2(ratios / tet_ratios)


def pythagorean_comma() -> float:
    """The Pythagorean comma: (3/2)^12 / 2^7 in cents."""
    return 1200.0 * math.log2((3.0 / 2.0) ** 12 / 2.0 ** 7)


# ── Quarter-comma meantone ──────────────────────────────────────────

def quarter_comma_meantone_ratios() -> np.ndarray:
    """Quarter-comma meantone temperament.

    The fifth is narrowed by 1/4 of a syntonic comma so that four
    fifths yield a pure major third (5:4).
    """
    syntonic_comma = 81.0 / 80.0
    fifth = (3.0 / 2.0) / (syntonic_comma ** 0.25)
    ratios = np.ones(12)
    order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    for i, idx in enumerate(order):
        r = fifth ** i
        while r >= 2.0:
            r /= 2.0
        while r < 1.0:
            r *= 2.0
        ratios[idx] = r
    return ratios


def meantone_cents() -> np.ndarray:
    """Cent deviations of quarter-comma meantone from 12-TET."""
    ratios = quarter_comma_meantone_ratios()
    tet = 2.0 ** (np.arange(12) / 12.0)
    return 1200.0 * np.log2(ratios / tet)


# ── Just intonation (5-limit) ──────────────────────────────────────

JUST_RATIOS_5LIMIT = np.array([
    1.0,       # C  unison
    16.0/15,   # C# minor second
    9.0/8,     # D  major second
    6.0/5,     # D# minor third
    5.0/4,     # E  major third
    4.0/3,     # F  perfect fourth
    45.0/32,   # F# augmented fourth
    3.0/2,     # G  perfect fifth
    8.0/5,     # G# minor sixth
    5.0/3,     # A  major sixth
    9.0/5,     # A# minor seventh
    15.0/8,    # B  major seventh
])


def just_intonation_cents() -> np.ndarray:
    """Cent deviations of 5-limit just intonation from 12-TET."""
    tet = 2.0 ** (np.arange(12) / 12.0)
    return 1200.0 * np.log2(JUST_RATIOS_5LIMIT / tet)


def syntonic_comma() -> float:
    """The syntonic comma 81/80 in cents."""
    return 1200.0 * math.log2(81.0 / 80.0)


# ── Werckmeister III ────────────────────────────────────────────────

def werckmeister_iii_ratios() -> np.ndarray:
    """Werckmeister III (1691) well-temperament.

    Distributes the Pythagorean comma across four fifths (C-G, G-D,
    D-A, B-F#) each narrowed by 1/4 comma; all others are pure 3:2.
    """
    pc = pythagorean_comma()
    q = pc / 4.0
    cents_from_c = np.zeros(12)
    fifth_chain = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    narrowed = {0, 1, 2, 10}
    current = 0.0
    for step in range(12):
        idx = fifth_chain[step]
        cents_from_c[idx] = current
        if step in narrowed:
            current += 701.955 - q
        else:
            current += 701.955
        while current >= 1200.0:
            current -= 1200.0
    ratios = 2.0 ** (cents_from_c / 1200.0)
    return ratios


def werckmeister_iii_cents() -> np.ndarray:
    """Cent deviations of Werckmeister III from 12-TET."""
    ratios = werckmeister_iii_ratios()
    tet = 2.0 ** (np.arange(12) / 12.0)
    return 1200.0 * np.log2(ratios / tet)


# ── Comparison utilities ────────────────────────────────────────────

@dataclass(frozen=True)
class TuningComparison:
    system_a: str
    system_b: str
    max_deviation_cents: float
    mean_deviation_cents: float
    rms_deviation_cents: float
    per_note_cents: np.ndarray


def compare_tunings(
    cents_a: np.ndarray,
    cents_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
) -> TuningComparison:
    """Compare two tuning systems by their cent deviations from 12-TET."""
    diff = cents_a - cents_b
    return TuningComparison(
        system_a=name_a,
        system_b=name_b,
        max_deviation_cents=float(np.max(np.abs(diff))),
        mean_deviation_cents=float(np.mean(np.abs(diff))),
        rms_deviation_cents=float(np.sqrt(np.mean(diff ** 2))),
        per_note_cents=diff,
    )


def cents_to_frequency(
    base_freq: float,
    semitone: int,
    cent_deviation: float,
) -> float:
    """Convert a 12-TET semitone + cent deviation to a frequency."""
    return base_freq * 2.0 ** ((semitone + cent_deviation / 100.0) / 12.0)


def beat_frequency(f1: float, f2: float) -> float:
    """Beat frequency between two pitches (audible roughness indicator)."""
    return abs(f1 - f2)


def interval_purity(ratio: float, target_ratio: float) -> float:
    """How close an interval ratio is to a target just ratio, in cents."""
    if ratio <= 0 or target_ratio <= 0:
        return 0.0
    return abs(1200.0 * math.log2(ratio / target_ratio))
