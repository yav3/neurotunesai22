"""Solo / improvisation transcription analysis.

Given a note sequence representing a transcribed solo, extracts statistical
and structural features that characterise the improviser's choices:
note-choice distributions, approach-note usage, rhythmic density profile,
register usage, and bebop-vocabulary density.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --- Note-choice categories relative to a chord context ---

def _classify_note(pitch_class: int, chord_root: int, chord_pcs: set[int],
                   scale_pcs: set[int]) -> str:
    pc = pitch_class % 12
    if pc in chord_pcs:
        return "chord_tone"
    if pc in scale_pcs:
        return "scale_tone"
    return "chromatic"


def note_choice_distribution(
    pitches: list[int],
    chord_root: int,
    chord_pcs: set[int],
    scale_pcs: set[int],
) -> dict[str, float]:
    """Fraction of notes that are chord tones, scale tones, chromatic."""
    if not pitches:
        return {"chord_tone": 0.0, "scale_tone": 0.0, "chromatic": 0.0}
    counts: dict[str, int] = {"chord_tone": 0, "scale_tone": 0, "chromatic": 0}
    for p in pitches:
        cat = _classify_note(p, chord_root, chord_pcs, scale_pcs)
        counts[cat] += 1
    n = len(pitches)
    return {k: v / n for k, v in counts.items()}


# --- Approach-note analysis ---

@dataclass
class ApproachEvent:
    """An approach note leading to a target."""
    approach_pitch: int
    target_pitch: int
    interval: int
    approach_type: str  # "chromatic_above", "chromatic_below", "diatonic", "enclosure", "leap"


def detect_approaches(
    pitches: list[int],
    chord_pcs: set[int],
) -> list[ApproachEvent]:
    """Find approach notes — non-chord tones resolving stepwise to chord tones."""
    events: list[ApproachEvent] = []
    for i in range(len(pitches) - 1):
        target_pc = pitches[i + 1] % 12
        if target_pc not in chord_pcs:
            continue
        curr_pc = pitches[i] % 12
        if curr_pc in chord_pcs:
            continue
        iv = pitches[i + 1] - pitches[i]
        if iv == 1:
            atype = "chromatic_below"
        elif iv == -1:
            atype = "chromatic_above"
        elif iv == 2 or iv == -2:
            atype = "diatonic"
        elif abs(iv) > 4:
            atype = "leap"
        else:
            atype = "diatonic"
        events.append(ApproachEvent(pitches[i], pitches[i + 1], iv, atype))
    return events


def approach_type_distribution(events: list[ApproachEvent]) -> dict[str, float]:
    """Fraction of each approach type."""
    if not events:
        return {}
    counts: dict[str, int] = {}
    for e in events:
        counts[e.approach_type] = counts.get(e.approach_type, 0) + 1
    n = len(events)
    return {k: v / n for k, v in counts.items()}


# --- Enclosure detection ---

def detect_enclosures(
    pitches: list[int],
    chord_pcs: set[int],
) -> list[tuple[int, int, int]]:
    """Detect enclosure patterns: (above, below, target) or (below, above, target).

    Returns list of (index_of_first_approach, index_of_second_approach, index_of_target).
    """
    results: list[tuple[int, int, int]] = []
    for i in range(len(pitches) - 2):
        target_pc = pitches[i + 2] % 12
        if target_pc not in chord_pcs:
            continue
        iv1 = pitches[i + 2] - pitches[i]
        iv2 = pitches[i + 2] - pitches[i + 1]
        if (iv1 > 0 and iv2 < 0) or (iv1 < 0 and iv2 > 0):
            if abs(iv1) <= 3 and abs(iv2) <= 3:
                results.append((i, i + 1, i + 2))
    return results


# --- Register and range analysis ---

@dataclass
class RegisterProfile:
    """Statistics about how a soloist uses pitch register."""
    lowest: int
    highest: int
    range_semitones: int
    mean_pitch: float
    std_pitch: float
    register_histogram: np.ndarray  # 10 octave bins (C0-C9)


def register_profile(pitches: list[int]) -> RegisterProfile:
    """Analyse the pitch-register usage of a solo."""
    if not pitches:
        return RegisterProfile(0, 0, 0, 0.0, 0.0, np.zeros(10))
    arr = np.array(pitches)
    hist = np.zeros(10)
    for p in pitches:
        octave = min(max(p // 12, 0), 9)
        hist[octave] += 1
    if len(pitches) > 0:
        hist /= len(pitches)
    return RegisterProfile(
        lowest=int(arr.min()),
        highest=int(arr.max()),
        range_semitones=int(arr.max() - arr.min()),
        mean_pitch=float(arr.mean()),
        std_pitch=float(arr.std()),
        register_histogram=hist,
    )


# --- Rhythmic density ---

def rhythmic_density_curve(
    onset_times: list[float],
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
) -> np.ndarray:
    """Note density (notes/sec) in sliding windows over a solo.

    Parameters
    ----------
    onset_times : list[float]
        Note onset times in seconds.
    window_sec : float
        Window width.
    hop_sec : float
        Hop between windows.

    Returns
    -------
    np.ndarray
        Density curve, one value per window position.
    """
    if not onset_times:
        return np.array([0.0])
    onsets = np.array(onset_times)
    end = onsets[-1] + window_sec
    positions = np.arange(0, end, hop_sec)
    densities = np.zeros(len(positions))
    for i, pos in enumerate(positions):
        count = np.sum((onsets >= pos) & (onsets < pos + window_sec))
        densities[i] = count / window_sec
    return densities


# --- Interval distribution ---

def interval_histogram(pitches: list[int], max_interval: int = 12) -> np.ndarray:
    """Histogram of absolute interval sizes, 0..max_interval (inclusive)."""
    hist = np.zeros(max_interval + 1)
    for i in range(len(pitches) - 1):
        iv = abs(pitches[i + 1] - pitches[i])
        iv = min(iv, max_interval)
        hist[iv] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# --- High-level solo summary ---

@dataclass
class SoloSummary:
    """Aggregate statistics for a transcribed improvisation."""
    n_notes: int
    duration_sec: float
    avg_density: float
    note_choice: dict[str, float]
    approach_dist: dict[str, float]
    n_enclosures: int
    register: RegisterProfile
    interval_hist: np.ndarray
    chromatic_ratio: float


def summarise_solo(
    pitches: list[int],
    onset_times: list[float] | None = None,
    chord_root: int = 0,
    chord_pcs: set[int] | None = None,
    scale_pcs: set[int] | None = None,
) -> SoloSummary:
    """One-call summary of a transcribed solo."""
    if chord_pcs is None:
        chord_pcs = {0, 4, 7, 10}
    if scale_pcs is None:
        scale_pcs = {0, 2, 4, 5, 7, 9, 10}
    chord_pcs_shifted = {(chord_root + p) % 12 for p in chord_pcs}
    scale_pcs_shifted = {(chord_root + p) % 12 for p in scale_pcs}

    nc = note_choice_distribution(pitches, chord_root, chord_pcs_shifted, scale_pcs_shifted)
    approaches = detect_approaches(pitches, chord_pcs_shifted)
    ad = approach_type_distribution(approaches)
    enclosures = detect_enclosures(pitches, chord_pcs_shifted)
    reg = register_profile(pitches)
    ih = interval_histogram(pitches)

    if onset_times and len(onset_times) >= 2:
        dur = onset_times[-1] - onset_times[0]
        density = len(pitches) / max(dur, 0.01)
    else:
        dur = 0.0
        density = 0.0

    return SoloSummary(
        n_notes=len(pitches),
        duration_sec=dur,
        avg_density=density,
        note_choice=nc,
        approach_dist=ad,
        n_enclosures=len(enclosures),
        register=reg,
        interval_hist=ih,
        chromatic_ratio=nc.get("chromatic", 0.0),
    )
