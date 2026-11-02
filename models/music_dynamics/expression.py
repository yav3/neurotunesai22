"""Musical expression dynamics: crescendo/diminuendo, accents, rubato, phrase arcs.

Models the expressive dimension of music performance including dynamic shaping,
accent detection, timing deviations (rubato), and humanization metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DynamicEventType(Enum):
    CRESCENDO = "crescendo"
    DIMINUENDO = "diminuendo"
    SFORZANDO = "sforzando"
    ACCENT = "accent"
    SUBITO_PIANO = "subito_piano"
    SUBITO_FORTE = "subito_forte"


@dataclass(frozen=True)
class DynamicEvent:
    event_type: DynamicEventType
    start_time: float
    end_time: float
    start_level: float
    end_level: float
    magnitude: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class PhraseArc:
    start_time: float
    end_time: float
    peak_time: float
    peak_level: float
    start_level: float
    end_level: float
    symmetry: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class ExpressionProfile:
    mean_velocity: float
    velocity_std: float
    timing_jitter_ms: float
    articulation_variance: float
    rubato_strength: float
    n_crescendi: int
    n_diminuendi: int
    n_accents: int
    phrase_arc_symmetry: float


def detect_crescendo_diminuendo(
    loudness_curve: np.ndarray,
    times: np.ndarray,
    min_duration: float = 0.5,
    min_change_db: float = 3.0,
) -> list[DynamicEvent]:
    """Detect crescendo and diminuendo regions from a loudness envelope."""
    events: list[DynamicEvent] = []
    if len(loudness_curve) < 3:
        return events
    diff = np.diff(loudness_curve)
    sign = np.sign(diff)
    i = 0
    while i < len(sign):
        if sign[i] == 0:
            i += 1
            continue
        direction = sign[i]
        start = i
        while i < len(sign) and sign[i] == direction:
            i += 1
        end = i
        duration = times[end] - times[start]
        change = loudness_curve[end] - loudness_curve[start]
        if duration >= min_duration and abs(change) >= min_change_db:
            etype = DynamicEventType.CRESCENDO if change > 0 else DynamicEventType.DIMINUENDO
            events.append(DynamicEvent(
                event_type=etype,
                start_time=float(times[start]),
                end_time=float(times[end]),
                start_level=float(loudness_curve[start]),
                end_level=float(loudness_curve[end]),
                magnitude=abs(change),
            ))
    return events


def detect_accents(
    velocities: np.ndarray,
    onset_times: np.ndarray,
    threshold_factor: float = 1.5,
) -> list[DynamicEvent]:
    """Detect accents as velocity spikes above local context."""
    events: list[DynamicEvent] = []
    if len(velocities) < 3:
        return events
    window = 5
    for i in range(len(velocities)):
        lo = max(0, i - window)
        hi = min(len(velocities), i + window + 1)
        context = np.concatenate([velocities[lo:i], velocities[i + 1:hi]])
        if len(context) == 0:
            continue
        mean_ctx = float(np.mean(context))
        if velocities[i] > mean_ctx * threshold_factor:
            spike = velocities[i] - mean_ctx
            if spike > 20:
                etype = DynamicEventType.SFORZANDO
            else:
                etype = DynamicEventType.ACCENT
            events.append(DynamicEvent(
                event_type=etype,
                start_time=float(onset_times[i]),
                end_time=float(onset_times[i]),
                start_level=mean_ctx,
                end_level=float(velocities[i]),
                magnitude=float(spike),
            ))
    return events


def rubato_strength(
    onset_times: np.ndarray,
    tempo_bpm: float,
) -> float:
    """Quantify rubato as normalized timing deviation from a steady grid."""
    if len(onset_times) < 2 or tempo_bpm <= 0:
        return 0.0
    beat_dur = 60.0 / tempo_bpm
    ioi = np.diff(onset_times)
    expected = np.round(ioi / beat_dur) * beat_dur
    expected = np.maximum(expected, beat_dur * 0.25)
    deviations = (ioi - expected) / expected
    return float(np.std(deviations))


def timing_jitter(onset_times: np.ndarray, grid_times: np.ndarray) -> float:
    """RMS timing deviation from a quantized grid, in milliseconds."""
    if len(onset_times) == 0 or len(grid_times) == 0:
        return 0.0
    min_len = min(len(onset_times), len(grid_times))
    deviations = onset_times[:min_len] - grid_times[:min_len]
    return float(np.sqrt(np.mean(deviations ** 2)) * 1000.0)


def extract_phrase_arcs(
    loudness_curve: np.ndarray,
    times: np.ndarray,
    min_phrase_duration: float = 2.0,
) -> list[PhraseArc]:
    """Extract phrase-level dynamic arcs (rise-peak-fall patterns)."""
    arcs: list[PhraseArc] = []
    if len(loudness_curve) < 5:
        return arcs
    smoothed = np.convolve(loudness_curve, np.ones(5) / 5, mode='same')
    diff = np.diff(smoothed)
    sign_changes = []
    for i in range(1, len(diff)):
        if diff[i - 1] > 0 and diff[i] <= 0:
            sign_changes.append(('peak', i))
        elif diff[i - 1] < 0 and diff[i] >= 0:
            sign_changes.append(('valley', i))
    valleys = [sc for sc in sign_changes if sc[0] == 'valley']
    for i in range(len(valleys) - 1):
        start_idx = valleys[i][1]
        end_idx = valleys[i + 1][1]
        duration = times[end_idx] - times[start_idx]
        if duration < min_phrase_duration:
            continue
        segment = smoothed[start_idx:end_idx + 1]
        peak_local = int(np.argmax(segment))
        peak_idx = start_idx + peak_local
        rise_dur = times[peak_idx] - times[start_idx]
        fall_dur = times[end_idx] - times[peak_idx]
        total = rise_dur + fall_dur
        symmetry = 1.0 - abs(rise_dur - fall_dur) / max(total, 1e-6)
        arcs.append(PhraseArc(
            start_time=float(times[start_idx]),
            end_time=float(times[end_idx]),
            peak_time=float(times[peak_idx]),
            peak_level=float(smoothed[peak_idx]),
            start_level=float(smoothed[start_idx]),
            end_level=float(smoothed[end_idx]),
            symmetry=symmetry,
        ))
    return arcs


def velocity_humanization(velocities: np.ndarray) -> dict[str, float]:
    """Compute humanization metrics from a sequence of MIDI velocities."""
    if len(velocities) == 0:
        return {"mean": 0.0, "std": 0.0, "range": 0.0, "cv": 0.0}
    mean = float(np.mean(velocities))
    std = float(np.std(velocities))
    vel_range = float(np.max(velocities) - np.min(velocities))
    cv = std / max(mean, 1e-6)
    return {"mean": mean, "std": std, "range": vel_range, "cv": cv}


def compute_expression_profile(
    velocities: np.ndarray,
    onset_times: np.ndarray,
    loudness_curve: np.ndarray | None = None,
    loudness_times: np.ndarray | None = None,
    grid_times: np.ndarray | None = None,
    tempo_bpm: float = 120.0,
) -> ExpressionProfile:
    """Aggregate expression metrics into a profile."""
    vh = velocity_humanization(velocities)
    jitter = timing_jitter(onset_times, grid_times) if grid_times is not None else 0.0
    rubato = rubato_strength(onset_times, tempo_bpm)
    n_cresc = 0
    n_dim = 0
    n_acc = 0
    arc_sym = 0.0
    if loudness_curve is not None and loudness_times is not None:
        events = detect_crescendo_diminuendo(loudness_curve, loudness_times)
        n_cresc = sum(1 for e in events if e.event_type == DynamicEventType.CRESCENDO)
        n_dim = sum(1 for e in events if e.event_type == DynamicEventType.DIMINUENDO)
        arcs = extract_phrase_arcs(loudness_curve, loudness_times)
        if arcs:
            arc_sym = float(np.mean([a.symmetry for a in arcs]))
    accents = detect_accents(velocities, onset_times)
    n_acc = len(accents)
    ioi = np.diff(onset_times) if len(onset_times) > 1 else np.array([0.0])
    art_var = float(np.std(ioi)) if len(ioi) > 1 else 0.0
    return ExpressionProfile(
        mean_velocity=vh["mean"],
        velocity_std=vh["std"],
        timing_jitter_ms=jitter,
        articulation_variance=art_var,
        rubato_strength=rubato,
        n_crescendi=n_cresc,
        n_diminuendi=n_dim,
        n_accents=n_acc,
        phrase_arc_symmetry=arc_sym,
    )
