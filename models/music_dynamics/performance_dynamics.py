"""Performance dynamics: MIDI-level expressive analysis.

Velocity-to-loudness perception via Stevens (1957) power law, expressive
timing deviations per Repp (1992, 1998), agogic accent detection, rubato
quantification, and phrasing arc scoring. Operates on MIDI-level note
events (velocity, onset time, duration) rather than raw audio.
Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class DynamicCurve:
    time_points: np.ndarray
    velocity_values: np.ndarray
    tempo_curve: np.ndarray


@dataclass
class ADSREnvelope:
    attack_slope: float
    decay_slope: float
    sustain_level: float
    release_slope: float
    peak_velocity: float
    peak_time: float


@dataclass
class ExpressiveTimingFeatures:
    rubato: float
    agogic_ratio: float
    tempo_variance: float
    dynamic_range_db: float
    phrasing_arc: float
    mean_velocity: float


# ---------------------------------------------------------------------------
# Stevens (1957) loudness power law
# ---------------------------------------------------------------------------

def velocity_to_loudness(velocity: float, exponent: float = 0.6,
                         ref_velocity: float = 64.0,
                         ref_sone: float = 1.0) -> float:
    v = max(float(velocity), 0.0)
    if v < 1.0:
        return 0.0
    ratio = v / ref_velocity
    return ref_sone * (ratio ** exponent)


def velocity_to_db(velocity: float) -> float:
    v = max(float(velocity), 1.0)
    return 20.0 * np.log10(v / 127.0)


# ---------------------------------------------------------------------------
# Dynamic range and contrast
# ---------------------------------------------------------------------------

def dynamic_range_db(velocities: Sequence[float]) -> float:
    arr = np.asarray(velocities, dtype=float)
    if len(arr) < 2:
        return 0.0
    arr = np.clip(arr, 1.0, 127.0)
    db = 20.0 * np.log10(arr / 127.0)
    return float(np.percentile(db, 95) - np.percentile(db, 5))


def dynamic_contrast(velocities: Sequence[float],
                     threshold: float = 64.0) -> float:
    arr = np.asarray(velocities, dtype=float)
    loud = arr[arr >= threshold]
    soft = arr[arr < threshold]
    if len(loud) == 0 or len(soft) == 0:
        return 0.0
    rms_loud = np.sqrt(np.mean(loud ** 2))
    rms_soft = np.sqrt(np.mean(soft ** 2))
    if rms_soft < 1e-10:
        return float('inf')
    return float(rms_loud / rms_soft)


# ---------------------------------------------------------------------------
# Crescendo / sforzando detection
# ---------------------------------------------------------------------------

def crescendo_detector(velocities: Sequence[float],
                       window: int = 8,
                       min_slope: float = 0.5) -> List[Tuple[int, int, float]]:
    arr = np.asarray(velocities, dtype=float)
    n = len(arr)
    if n < window:
        return []
    results: List[Tuple[int, int, float]] = []
    run_start: Optional[int] = None
    for i in range(n - window + 1):
        seg = arr[i:i + window]
        x = np.arange(window, dtype=float)
        slope = float(np.polyfit(x, seg, 1)[0])
        if slope >= min_slope:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_end = i + window - 2
                total_rise = float(arr[min(run_end, n - 1)] - arr[run_start])
                results.append((run_start, run_end, total_rise))
                run_start = None
    if run_start is not None:
        run_end = n - 1
        total_rise = float(arr[run_end] - arr[run_start])
        results.append((run_start, run_end, total_rise))
    return results


def sforzando_detector(velocities: Sequence[float],
                       times: Sequence[float],
                       vel_threshold: float = 40.0,
                       time_threshold: float = 0.1) -> List[int]:
    vel = np.asarray(velocities, dtype=float)
    t = np.asarray(times, dtype=float)
    n = len(vel)
    hits: List[int] = []
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        dv = vel[i] - vel[i - 1]
        if dv >= vel_threshold and dt <= time_threshold:
            hits.append(i)
    return hits


# ---------------------------------------------------------------------------
# Rubato and expressive timing (Repp 1992/1998)
# ---------------------------------------------------------------------------

def rubato_measure(onset_times: Sequence[float],
                   nominal_ioi: Optional[float] = None) -> float:
    t = np.asarray(onset_times, dtype=float)
    if len(t) < 3:
        return 0.0
    iois = np.diff(t)
    iois = iois[iois > 0]
    if len(iois) < 2:
        return 0.0
    if nominal_ioi is None:
        nominal_ioi = float(np.median(iois))
    if nominal_ioi < 1e-10:
        return 0.0
    ratios = iois / nominal_ioi
    return float(np.std(ratios))


def agogic_accent_detector(onset_times: Sequence[float],
                           durations: Sequence[float],
                           neighborhood: int = 3,
                           threshold: float = 1.3) -> List[int]:
    dur = np.asarray(durations, dtype=float)
    n = len(dur)
    if n < 2 * neighborhood + 1:
        return []
    accents: List[int] = []
    for i in range(neighborhood, n - neighborhood):
        local = np.concatenate([dur[i - neighborhood:i],
                                dur[i + 1:i + neighborhood + 1]])
        local_mean = np.mean(local)
        if local_mean > 1e-10 and dur[i] / local_mean >= threshold:
            accents.append(i)
    return accents


def tempo_from_onsets(onset_times: Sequence[float]) -> np.ndarray:
    t = np.asarray(onset_times, dtype=float)
    if len(t) < 2:
        return np.array([])
    iois = np.diff(t)
    iois = np.where(iois < 1e-10, 1e-10, iois)
    return 60.0 / iois


# ---------------------------------------------------------------------------
# Dynamic envelope (ADSR from velocity contour)
# ---------------------------------------------------------------------------

def dynamic_envelope(velocities: Sequence[float],
                     times: Sequence[float]) -> ADSREnvelope:
    vel = np.asarray(velocities, dtype=float)
    t = np.asarray(times, dtype=float)
    n = len(vel)
    if n < 2:
        return ADSREnvelope(0.0, 0.0, float(vel[0]) if n > 0 else 0.0,
                            0.0, float(vel[0]) if n > 0 else 0.0,
                            float(t[0]) if n > 0 else 0.0)

    peak_idx = int(np.argmax(vel))
    peak_vel = float(vel[peak_idx])
    peak_time = float(t[peak_idx])

    if peak_idx > 0:
        dt = t[peak_idx] - t[0]
        attack_slope = (peak_vel - vel[0]) / max(dt, 1e-10)
    else:
        attack_slope = 0.0

    if peak_idx < n - 1:
        mid = (peak_idx + n) // 2
        mid = max(mid, peak_idx + 1)
        dt_decay = t[min(mid, n - 1)] - t[peak_idx]
        decay_slope = (vel[min(mid, n - 1)] - peak_vel) / max(dt_decay, 1e-10)
        sustain_level = float(np.mean(vel[mid:]) if mid < n else vel[-1])
        dt_release = t[-1] - t[max(mid, 0)]
        release_slope = (vel[-1] - sustain_level) / max(dt_release, 1e-10)
    else:
        decay_slope = 0.0
        sustain_level = peak_vel
        release_slope = 0.0

    return ADSREnvelope(float(attack_slope), float(decay_slope),
                        float(sustain_level), float(release_slope),
                        peak_vel, peak_time)


# ---------------------------------------------------------------------------
# Phrasing arc score
# ---------------------------------------------------------------------------

def phrasing_arc_score(velocities: Sequence[float]) -> float:
    vel = np.asarray(velocities, dtype=float)
    n = len(vel)
    if n < 3:
        return 0.0
    x = np.linspace(0.0, np.pi, n)
    ideal_arc = np.sin(x)
    v_min, v_max = vel.min(), vel.max()
    if v_max - v_min < 1e-10:
        return 0.0
    vel_norm = (vel - v_min) / (v_max - v_min)
    corr = np.corrcoef(vel_norm, ideal_arc)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Aggregate feature extraction
# ---------------------------------------------------------------------------

def expressive_timing_features(
    velocities: Sequence[float],
    onset_times: Sequence[float],
    durations: Sequence[float],
) -> ExpressiveTimingFeatures:
    vel = np.asarray(velocities, dtype=float)
    rub = rubato_measure(onset_times)
    accents = agogic_accent_detector(onset_times, durations)
    agogic_ratio = len(accents) / max(len(vel), 1)
    tempo = tempo_from_onsets(onset_times)
    tempo_var = float(np.var(tempo)) if len(tempo) > 1 else 0.0
    dr = dynamic_range_db(velocities)
    arc = phrasing_arc_score(velocities)
    mean_vel = float(np.mean(vel)) if len(vel) > 0 else 0.0

    return ExpressiveTimingFeatures(
        rubato=rub,
        agogic_ratio=agogic_ratio,
        tempo_variance=tempo_var,
        dynamic_range_db=dr,
        phrasing_arc=arc,
        mean_velocity=mean_vel,
    )


# ---------------------------------------------------------------------------
# Velocity contour analysis
# ---------------------------------------------------------------------------

def build_dynamic_curve(
    velocities: Sequence[float],
    onset_times: Sequence[float],
    nominal_ioi: Optional[float] = None,
) -> DynamicCurve:
    vel = np.asarray(velocities, dtype=float)
    t = np.asarray(onset_times, dtype=float)
    tempo = tempo_from_onsets(onset_times)
    if len(tempo) < len(t):
        tempo = np.append(tempo, tempo[-1] if len(tempo) > 0 else 120.0)
    return DynamicCurve(
        time_points=t,
        velocity_values=vel,
        tempo_curve=tempo,
    )
