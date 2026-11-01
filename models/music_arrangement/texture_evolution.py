"""Texture evolution analysis: tutti/solo detection, Rimsky-Korsakov balance,
instrumentation density curves, colla parte tracking, register weight functions.

Models orchestral texture as a time-varying phenomenon: how many instruments
are active, what families dominate, whether voices double or diverge, and
whether the register distribution follows established balance principles.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class TextureCategory(Enum):
    SOLO = "solo"
    DUET = "duet"
    SMALL_ENSEMBLE = "small_ensemble"    # 3–6 parts
    CHAMBER = "chamber"                  # 7–12 parts
    FULL_ORCHESTRA = "full_orchestra"    # 13+ parts
    TUTTI = "tutti"                      # all available parts active


class RegisterBand(Enum):
    SUB_BASS = "sub_bass"      # MIDI 21–35  (A0–B1)
    BASS = "bass"              # MIDI 36–47  (C2–B2)
    LOW_MID = "low_mid"        # MIDI 48–59  (C3–B3)
    MID = "mid"                # MIDI 60–71  (C4–B4)
    HIGH_MID = "high_mid"      # MIDI 72–83  (C5–B5)
    HIGH = "high"              # MIDI 84–96  (C6–C7)
    BRILLIANCE = "brilliance"  # MIDI 97–108 (C#7–C8)


REGISTER_BOUNDS = {
    RegisterBand.SUB_BASS: (21, 35),
    RegisterBand.BASS: (36, 47),
    RegisterBand.LOW_MID: (48, 59),
    RegisterBand.MID: (60, 71),
    RegisterBand.HIGH_MID: (72, 83),
    RegisterBand.HIGH: (84, 96),
    RegisterBand.BRILLIANCE: (97, 108),
}

# Rimsky-Korsakov balance principles: ideal energy distribution by register
# when writing for full orchestra. Bass foundation supports upper voices;
# too much low-mid creates muddiness; high-mid is most projecting.
RK_IDEAL_BALANCE = {
    RegisterBand.SUB_BASS: 0.08,
    RegisterBand.BASS: 0.18,
    RegisterBand.LOW_MID: 0.12,
    RegisterBand.MID: 0.20,
    RegisterBand.HIGH_MID: 0.22,
    RegisterBand.HIGH: 0.15,
    RegisterBand.BRILLIANCE: 0.05,
}

# Muddy interval thresholds: minor/major seconds and minor thirds
# below C3 (MIDI 48) create muddiness. Wider intervals needed below this point.
MUDDY_THRESHOLD_MIDI = 48
MUDDY_INTERVALS = {1, 2, 3}  # semitones


@dataclass
class TextureSnapshot:
    """Texture state at a single time point."""
    time: float
    active_parts: int
    total_parts: int
    category: TextureCategory
    register_distribution: Dict[RegisterBand, float]
    family_distribution: Dict[str, float]
    density: float          # active_parts / total_parts
    rk_balance_score: float # adherence to Rimsky-Korsakov balance


@dataclass
class CollaParteEvent:
    """Detected colla parte (doubling at unison or octave)."""
    part_a: int
    part_b: int
    start_time: float
    end_time: float
    interval: int           # 0 = unison, 12 = octave, -12 = octave below


@dataclass
class TextureEvolution:
    """Full texture analysis over a piece."""
    snapshots: List[TextureSnapshot]
    colla_parte: List[CollaParteEvent]
    density_curve: np.ndarray
    times: np.ndarray


def classify_texture(n_active: int, n_total: int) -> TextureCategory:
    """Classify texture category from active/total part counts."""
    if n_active <= 0:
        return TextureCategory.SOLO
    if n_active == 1:
        return TextureCategory.SOLO
    if n_active == 2:
        return TextureCategory.DUET
    if n_active <= 6:
        return TextureCategory.SMALL_ENSEMBLE
    if n_active <= 12:
        return TextureCategory.CHAMBER
    if n_total > 0 and n_active >= n_total * 0.9:
        return TextureCategory.TUTTI
    return TextureCategory.FULL_ORCHESTRA


def note_to_register(midi: int) -> RegisterBand:
    """Map MIDI note to register band."""
    for band, (lo, hi) in REGISTER_BOUNDS.items():
        if lo <= midi <= hi:
            return band
    if midi < 21:
        return RegisterBand.SUB_BASS
    return RegisterBand.BRILLIANCE


def register_distribution(notes_midi: np.ndarray) -> Dict[RegisterBand, float]:
    """Compute normalised energy distribution across register bands."""
    counts = {band: 0 for band in RegisterBand}
    for n in notes_midi:
        counts[note_to_register(int(n))] += 1
    total = max(sum(counts.values()), 1)
    return {band: c / total for band, c in counts.items()}


def rk_balance_score(distribution: Dict[RegisterBand, float]) -> float:
    """Score adherence to Rimsky-Korsakov ideal balance (0 = poor, 1 = perfect)."""
    mse = 0.0
    for band in RegisterBand:
        actual = distribution.get(band, 0.0)
        ideal = RK_IDEAL_BALANCE.get(band, 0.0)
        mse += (actual - ideal) ** 2
    rmse = np.sqrt(mse / len(RegisterBand))
    return float(max(0.0, 1.0 - rmse * 5.0))


def muddy_interval_count(chord_midi: np.ndarray) -> int:
    """Count intervals in the low register that would sound muddy."""
    low_notes = np.sort(chord_midi[chord_midi < MUDDY_THRESHOLD_MIDI])
    count = 0
    for i in range(len(low_notes)):
        for j in range(i + 1, len(low_notes)):
            interval = abs(int(low_notes[j]) - int(low_notes[i])) % 12
            if interval in MUDDY_INTERVALS:
                count += 1
    return count


def detect_colla_parte(parts: List[np.ndarray],
                       tolerance: int = 0) -> List[CollaParteEvent]:
    """Detect unison and octave doublings between parts.

    Parameters
    ----------
    parts : list of arrays, each shape (T,) of MIDI pitches (0 = rest)
    tolerance : allowed deviation in semitones (0 = exact)
    """
    events = []
    n_parts = len(parts)
    min_len = min(len(p) for p in parts) if parts else 0

    for i in range(n_parts):
        for j in range(i + 1, n_parts):
            in_doubling = False
            start_t = 0
            interval = 0
            for t in range(min_len):
                a, b = int(parts[i][t]), int(parts[j][t])
                if a == 0 or b == 0:
                    if in_doubling:
                        events.append(CollaParteEvent(i, j, float(start_t), float(t), interval))
                        in_doubling = False
                    continue
                diff = a - b
                is_doubling = False
                for oct in [0, 12, -12, 24, -24]:
                    if abs(diff - oct) <= tolerance:
                        is_doubling = True
                        interval = oct
                        break
                if is_doubling and not in_doubling:
                    in_doubling = True
                    start_t = t
                elif not is_doubling and in_doubling:
                    events.append(CollaParteEvent(i, j, float(start_t), float(t), interval))
                    in_doubling = False
            if in_doubling:
                events.append(CollaParteEvent(i, j, float(start_t), float(min_len), interval))
    return events


def instrumentation_density_curve(parts: List[np.ndarray]) -> np.ndarray:
    """Normalised density curve: fraction of parts active at each time step."""
    if not parts:
        return np.array([])
    min_len = min(len(p) for p in parts)
    n_parts = len(parts)
    density = np.zeros(min_len)
    for t in range(min_len):
        active = sum(1 for p in parts if int(p[t]) > 0)
        density[t] = active / n_parts
    return density


def family_distribution_over_time(
    parts: List[np.ndarray],
    families: List[str],
) -> Dict[str, np.ndarray]:
    """Per-family activity fraction over time."""
    if not parts:
        return {}
    min_len = min(len(p) for p in parts)
    unique_fam = sorted(set(families))
    result = {f: np.zeros(min_len) for f in unique_fam}
    for t in range(min_len):
        active_fam: Dict[str, int] = {f: 0 for f in unique_fam}
        n_active = 0
        for p, fam in zip(parts, families):
            if int(p[t]) > 0:
                active_fam[fam] += 1
                n_active += 1
        if n_active > 0:
            for f in unique_fam:
                result[f][t] = active_fam[f] / n_active
    return result


def register_weight_curve(parts: List[np.ndarray]) -> Dict[RegisterBand, np.ndarray]:
    """Register band activation weight over time."""
    if not parts:
        return {}
    min_len = min(len(p) for p in parts)
    result = {band: np.zeros(min_len) for band in RegisterBand}
    for t in range(min_len):
        notes = np.array([int(p[t]) for p in parts if int(p[t]) > 0])
        if len(notes) == 0:
            continue
        dist = register_distribution(notes)
        for band, val in dist.items():
            result[band][t] = val
    return result


def texture_change_points(density_curve: np.ndarray,
                          threshold: float = 0.15) -> List[int]:
    """Detect points where texture density changes significantly."""
    if len(density_curve) < 2:
        return []
    diff = np.abs(np.diff(density_curve))
    return [int(i + 1) for i in range(len(diff)) if diff[i] >= threshold]


def analyze_texture_evolution(
    parts: List[np.ndarray],
    families: Optional[List[str]] = None,
    times: Optional[np.ndarray] = None,
) -> TextureEvolution:
    """Full texture evolution analysis over a multi-part score."""
    if not parts:
        return TextureEvolution([], [], np.array([]), np.array([]))

    min_len = min(len(p) for p in parts)
    n_parts = len(parts)
    if families is None:
        families = ["unknown"] * n_parts
    if times is None:
        times = np.arange(min_len, dtype=float)

    density = instrumentation_density_curve(parts)
    colla_parte = detect_colla_parte(parts)

    snapshots = []
    for t in range(min_len):
        notes = np.array([int(p[t]) for p in parts if int(p[t]) > 0])
        n_active = len(notes)
        cat = classify_texture(n_active, n_parts)
        if n_active > 0:
            reg_dist = register_distribution(notes)
            rk_score = rk_balance_score(reg_dist)
        else:
            reg_dist = {band: 0.0 for band in RegisterBand}
            rk_score = 0.0
        fam_dist: Dict[str, float] = {}
        for p, fam in zip(parts, families):
            if int(p[t]) > 0:
                fam_dist[fam] = fam_dist.get(fam, 0) + 1
        total_active = max(sum(fam_dist.values()), 1)
        fam_dist = {f: c / total_active for f, c in fam_dist.items()}

        snapshots.append(TextureSnapshot(
            time=float(times[t]) if t < len(times) else float(t),
            active_parts=n_active,
            total_parts=n_parts,
            category=cat,
            register_distribution=reg_dist,
            family_distribution=fam_dist,
            density=density[t],
            rk_balance_score=rk_score,
        ))

    return TextureEvolution(snapshots, colla_parte, density, times[:min_len])


def rk_balance_summary(evolution: TextureEvolution) -> Dict[str, float]:
    """Aggregate Rimsky-Korsakov balance statistics over the piece."""
    if not evolution.snapshots:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    scores = np.array([s.rk_balance_score for s in evolution.snapshots])
    return {
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "std": float(np.std(scores)),
    }


def texture_variety_index(evolution: TextureEvolution) -> float:
    """Shannon entropy of texture categories (higher = more variety)."""
    if not evolution.snapshots:
        return 0.0
    cats = [s.category.value for s in evolution.snapshots]
    unique, counts = np.unique(cats, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-15)))


def colla_parte_fraction(evolution: TextureEvolution) -> float:
    """Fraction of total part-time spent in colla parte doubling."""
    if not evolution.snapshots or not evolution.colla_parte:
        return 0.0
    total_time = len(evolution.snapshots)
    doubled_time = sum(e.end_time - e.start_time for e in evolution.colla_parte)
    n_parts = evolution.snapshots[0].total_parts if evolution.snapshots else 1
    return float(doubled_time / max(total_time * n_parts, 1))
