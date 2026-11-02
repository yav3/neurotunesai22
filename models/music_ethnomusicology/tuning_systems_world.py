"""World tuning systems: makam, maqam, gamelan, raga, Thai 7-TET, microtonal analysis.

Pure numpy. Provides non-Western scale definitions, interval measurement,
scale detection, gamelan ombak analysis, raga aroha/avaroha extraction,
and maqam jins identification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Interval utilities
# ---------------------------------------------------------------------------

def freq_to_cents(f: float, ref: float = 440.0) -> float:
    """Convert frequency to cents relative to reference."""
    if f <= 0 or ref <= 0:
        return 0.0
    return 1200.0 * np.log2(f / ref)


def cents_to_freq(cents: float, ref: float = 440.0) -> float:
    """Convert cents to frequency."""
    return ref * (2.0 ** (cents / 1200.0))


def interval_cents(f1: float, f2: float) -> float:
    """Interval between two frequencies in cents."""
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return 1200.0 * np.log2(f2 / f1)


def nearest_et_deviation(freq: float, ref: float = 440.0,
                         divisions: int = 12) -> tuple[int, float]:
    """Find nearest equal-temperament pitch and deviation in cents.

    Returns (degree, deviation_cents) where degree is steps from ref.
    """
    if freq <= 0:
        return (0, 0.0)
    step_cents = 1200.0 / divisions
    total_cents = 1200.0 * np.log2(freq / ref)
    degree = int(round(total_cents / step_cents))
    dev = total_cents - degree * step_cents
    return (degree, dev)


# ---------------------------------------------------------------------------
# Tuning system catalogs
# ---------------------------------------------------------------------------

@dataclass
class TuningSystem:
    name: str
    intervals_cents: np.ndarray  # ascending from unison, including 0
    culture: str
    description: str


# Turkish makam: 53-TET (Holdrian comma ≈ 22.64 cents)
HOLDRIAN_COMMA = 1200.0 / 53

TURKISH_MAKAM_SCALES: dict[str, np.ndarray] = {
    "rast":    np.array([0, 9, 17, 22, 31, 40, 48, 53]) * HOLDRIAN_COMMA,
    "ussak":   np.array([0, 8, 13, 22, 31, 35, 44, 53]) * HOLDRIAN_COMMA,
    "hicaz":   np.array([0, 5, 17, 22, 31, 35, 48, 53]) * HOLDRIAN_COMMA,
    "huseyni": np.array([0, 8, 13, 22, 31, 40, 44, 53]) * HOLDRIAN_COMMA,
    "segah":   np.array([0, 4, 13, 22, 26, 35, 44, 53]) * HOLDRIAN_COMMA,
}

# Arabic maqam: 24-TET (quarter-tones)
QUARTER_TONE = 50.0  # cents

ARABIC_MAQAM_SCALES: dict[str, np.ndarray] = {
    "rast":     np.array([0, 4, 7, 10, 14, 18, 21, 24]) * QUARTER_TONE,
    "bayati":   np.array([0, 3, 6, 10, 14, 17, 20, 24]) * QUARTER_TONE,
    "sikah":    np.array([0, 3, 7, 10, 13, 17, 21, 24]) * QUARTER_TONE,
    "hijaz":    np.array([0, 2, 8, 10, 14, 16, 22, 24]) * QUARTER_TONE,
    "nahawand": np.array([0, 4, 6, 10, 14, 16, 22, 24]) * QUARTER_TONE,
    "kurd":     np.array([0, 2, 6, 10, 14, 16, 20, 24]) * QUARTER_TONE,
    "ajam":     np.array([0, 4, 8, 10, 14, 18, 22, 24]) * QUARTER_TONE,
    "saba":     np.array([0, 3, 6, 8, 14, 16, 20, 24]) * QUARTER_TONE,
}

# Indonesian gamelan
SLENDRO_5 = np.array([0, 240, 480, 720, 960])  # roughly equal pentatonic
PELOG_7 = np.array([0, 120, 270, 540, 660, 780, 1020])  # unequal heptatonic

# Thai classical: 7-TET
THAI_7TET = np.arange(8) * (1200.0 / 7)

# Indian classical: 22 shrutis (Pythagorean-just hybrid)
SHRUTI_22 = np.array([
    0, 90, 112, 182, 204, 294, 316, 386, 408, 498, 520,
    590, 612, 702, 792, 814, 884, 906, 996, 1018, 1088, 1110,
])

ALL_TUNING_SYSTEMS: dict[str, TuningSystem] = {
    "slendro": TuningSystem("Slendro", SLENDRO_5, "Javanese", "5-tone roughly equidistant"),
    "pelog": TuningSystem("Pelog", PELOG_7, "Javanese/Balinese", "7-tone unequal"),
    "thai_7tet": TuningSystem("Thai 7-TET", THAI_7TET, "Thai", "7-tone equal temperament"),
    "shruti_22": TuningSystem("22 Shrutis", SHRUTI_22, "Indian", "22-tone Pythagorean-just"),
}
for name, intervals in TURKISH_MAKAM_SCALES.items():
    ALL_TUNING_SYSTEMS[f"makam_{name}"] = TuningSystem(f"Makam {name}", intervals, "Turkish", f"53-TET {name}")
for name, intervals in ARABIC_MAQAM_SCALES.items():
    ALL_TUNING_SYSTEMS[f"maqam_{name}"] = TuningSystem(f"Maqam {name}", intervals, "Arabic", f"24-TET {name}")


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

def scale_deviation(pitches_cents: np.ndarray,
                    scale_cents: np.ndarray) -> float:
    """Total deviation of pitches from the nearest scale degree (in cents).

    pitches_cents: measured pitch positions in cents (mod 1200).
    scale_cents: scale degrees in cents.
    """
    total = 0.0
    for p in pitches_cents:
        p_mod = p % 1200
        diffs = np.abs(scale_cents - p_mod)
        # Also check wrapping
        diffs_wrap = np.abs(scale_cents - p_mod + 1200)
        diffs_wrap2 = np.abs(scale_cents - p_mod - 1200)
        min_diff = min(float(np.min(diffs)), float(np.min(diffs_wrap)), float(np.min(diffs_wrap2)))
        total += min_diff
    return total / max(len(pitches_cents), 1)


def detect_tuning_system(pitches_hz: np.ndarray,
                         ref: float = 440.0,
                         candidates: dict[str, TuningSystem] | None = None
                         ) -> list[tuple[str, float]]:
    """Find the best-fitting tuning system for measured pitches.

    Returns list of (system_name, mean_deviation_cents) sorted ascending.
    """
    if candidates is None:
        candidates = ALL_TUNING_SYSTEMS

    pitches_cents = np.array([freq_to_cents(f, ref) for f in pitches_hz if f > 0])
    if len(pitches_cents) == 0:
        return []

    results: list[tuple[str, float]] = []
    for name, ts in candidates.items():
        dev = scale_deviation(pitches_cents, ts.intervals_cents)
        results.append((name, dev))

    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Microtonal inflection detection
# ---------------------------------------------------------------------------

@dataclass
class MicrotonalInflection:
    position: int
    pitch_cents: float
    nearest_12tet_cents: float
    deviation_cents: float
    classification: str  # "quarter_tone", "shruti", "blue_note", "other"


def detect_microtonal_inflections(pitches_hz: np.ndarray,
                                  ref: float = 440.0,
                                  threshold_cents: float = 15.0
                                  ) -> list[MicrotonalInflection]:
    """Detect pitches that deviate from 12-TET beyond threshold."""
    inflections: list[MicrotonalInflection] = []
    for i, f in enumerate(pitches_hz):
        if f <= 0:
            continue
        degree, dev = nearest_et_deviation(f, ref, 12)
        if abs(dev) < threshold_cents:
            continue

        # Classify
        if 40 < abs(dev) < 60:
            cls = "quarter_tone"
        elif 15 < abs(dev) < 30:
            cls = "shruti"
        elif 20 < abs(dev) < 50 and (degree % 12) in (3, 7, 10):
            cls = "blue_note"
        else:
            cls = "other"

        inflections.append(MicrotonalInflection(
            position=i,
            pitch_cents=freq_to_cents(f, ref),
            nearest_12tet_cents=degree * 100.0,
            deviation_cents=dev,
            classification=cls,
        ))

    return inflections


# ---------------------------------------------------------------------------
# Gamelan analysis
# ---------------------------------------------------------------------------

@dataclass
class GamelanAnalysis:
    ombak_hz: float          # beating frequency between paired instruments
    roughness: float         # Sethares-style spectral roughness
    tuning_system: str       # "slendro" or "pelog"
    tuning_deviation: float  # mean cents deviation from ideal


def analyze_gamelan_tuning(pitches_a: np.ndarray, pitches_b: np.ndarray,
                           ref: float = 440.0) -> GamelanAnalysis:
    """Analyze gamelan tuning from paired instrument pitches."""
    n = min(len(pitches_a), len(pitches_b))
    if n == 0:
        return GamelanAnalysis(0.0, 0.0, "unknown", 0.0)

    # Ombak: mean beating frequency
    beats = np.abs(pitches_a[:n] - pitches_b[:n])
    ombak = float(np.mean(beats[beats > 0])) if np.any(beats > 0) else 0.0

    # Roughness from beating
    roughness = 0.0
    for i in range(n):
        if pitches_a[i] > 0 and pitches_b[i] > 0:
            df = abs(pitches_a[i] - pitches_b[i])
            fmin = min(pitches_a[i], pitches_b[i])
            s = 0.24 * (0.0207 * fmin + 18.96)  # critical bandwidth estimate
            x = df / (s + 1e-10)
            roughness += float(np.exp(-3.5 * x) * x)
    roughness /= max(n, 1)

    # Best fit: slendro vs pelog
    avg_pitches = (pitches_a[:n] + pitches_b[:n]) / 2.0
    cents = np.array([freq_to_cents(f, ref) for f in avg_pitches if f > 0])

    dev_s = scale_deviation(cents, SLENDRO_5)
    dev_p = scale_deviation(cents, PELOG_7)
    if dev_s < dev_p:
        system, dev = "slendro", dev_s
    else:
        system, dev = "pelog", dev_p

    return GamelanAnalysis(ombak, roughness, system, dev)


# ---------------------------------------------------------------------------
# Raga scale extraction
# ---------------------------------------------------------------------------

@dataclass
class RagaAnalysis:
    aroha: list[int]           # ascending scale degrees (pitch classes 0..11)
    avaroha: list[int]         # descending scale degrees
    vadi: int                  # dominant tone (most frequent/longest)
    samvadi: int               # subdominant
    pitch_class_weights: np.ndarray  # (12,) normalised duration weights


def extract_raga_features(pitches_midi: np.ndarray,
                          durations: np.ndarray | None = None) -> RagaAnalysis:
    """Extract raga features from a melodic pitch sequence.

    Identifies aroha/avaroha patterns and vadi/samvadi from pitch-class statistics.
    """
    if len(pitches_midi) < 3:
        return RagaAnalysis([], [], 0, 0, np.zeros(12))

    pcs = (pitches_midi.astype(int) % 12)

    # Weight by duration or count
    weights = np.zeros(12)
    if durations is not None and len(durations) == len(pitches_midi):
        for pc, d in zip(pcs, durations):
            weights[pc] += d
    else:
        for pc in pcs:
            weights[pc] += 1.0

    weights /= (np.sum(weights) + 1e-10)

    # Vadi = highest weight, samvadi = second highest
    sorted_pcs = np.argsort(-weights)
    vadi = int(sorted_pcs[0])
    samvadi = int(sorted_pcs[1]) if len(sorted_pcs) > 1 else vadi

    # Extract ascending/descending movements
    ascending_pcs = set()
    descending_pcs = set()
    for i in range(1, len(pitches_midi)):
        if pitches_midi[i] > pitches_midi[i - 1]:
            ascending_pcs.add(int(pcs[i]))
        elif pitches_midi[i] < pitches_midi[i - 1]:
            descending_pcs.add(int(pcs[i]))

    aroha = sorted(ascending_pcs)
    avaroha = sorted(descending_pcs, reverse=True)

    return RagaAnalysis(aroha, avaroha, vadi, samvadi, weights)


# ---------------------------------------------------------------------------
# Maqam jins identification
# ---------------------------------------------------------------------------

@dataclass
class JinsSegment:
    start_idx: int
    end_idx: int
    jins_name: str
    confidence: float
    intervals_cents: np.ndarray


# Catalog of common ajnas (tetrachords/trichords) in cents
AJNAS_CATALOG: dict[str, np.ndarray] = {
    "rast":     np.array([0, 200, 350, 500]),
    "bayati":   np.array([0, 150, 300, 500]),
    "sikah":    np.array([0, 150, 350, 500]),
    "hijaz":    np.array([0, 100, 400, 500]),
    "nahawand": np.array([0, 200, 300, 500]),
    "kurd":     np.array([0, 100, 300, 500]),
    "ajam":     np.array([0, 200, 400, 500]),
    "saba":     np.array([0, 150, 300, 400]),
}


def identify_jins(pitches_cents: np.ndarray,
                  window_size: int = 4) -> list[JinsSegment]:
    """Segment a melody into tetrachord jins and classify each.

    pitches_cents: pitch values in cents (absolute or relative).
    """
    n = len(pitches_cents)
    segments: list[JinsSegment] = []

    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        chunk = pitches_cents[start:end]
        # Normalize to start from 0
        intervals = chunk - chunk[0]

        best_name = "unknown"
        best_dev = float("inf")

        for name, template in AJNAS_CATALOG.items():
            if len(template) != len(intervals):
                continue
            dev = float(np.mean(np.abs(intervals - template)))
            if dev < best_dev:
                best_dev = dev
                best_name = name

        confidence = max(0.0, 1.0 - best_dev / 100.0)

        segments.append(JinsSegment(
            start_idx=start,
            end_idx=end,
            jins_name=best_name,
            confidence=confidence,
            intervals_cents=intervals,
        ))

    return segments
