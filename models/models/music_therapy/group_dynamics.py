"""Group music-making dynamics analysis."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class GroupSession:
    n_participants: int = 4
    duration_sec: float = 600.0
    instrument_types: list[str] = field(default_factory=list)


def rhythmic_cohesion(participant_tap_times: list[list[float]]) -> float:
    if len(participant_tap_times) < 2:
        return 1.0
    ious = []
    for i in range(len(participant_tap_times)):
        for j in range(i + 1, len(participant_tap_times)):
            a = np.array(participant_tap_times[i])
            b = np.array(participant_tap_times[j])
            if len(a) == 0 or len(b) == 0:
                ious.append(0.0)
                continue
            diffs = []
            for t in a:
                diffs.append(float(np.min(np.abs(b - t))))
            mean_diff = np.mean(diffs)
            period = np.median(np.diff(a)) if len(a) > 1 else 1.0
            ious.append(float(max(0, 1.0 - mean_diff / max(period, 1e-6))))
    return float(np.mean(ious)) if ious else 0.0


def turn_taking_balance(solo_durations: list[float]) -> float:
    if not solo_durations or all(d == 0 for d in solo_durations):
        return 0.0
    arr = np.array(solo_durations, dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    proportions = arr / total
    n = len(proportions)
    ideal = 1.0 / n
    max_deviation = 1.0 - ideal
    deviation = float(np.max(np.abs(proportions - ideal)))
    return float(max(0.0, 1.0 - deviation / max(max_deviation, 1e-6)))


def collective_tempo_drift(
    group_beat_times: list[float], metronome_times: list[float]
) -> dict[str, float]:
    if len(group_beat_times) < 2 or len(metronome_times) < 2:
        return {"drift_bpm": 0.0, "stability": 1.0}
    g = np.array(group_beat_times)
    m = np.array(metronome_times)
    g_ibi = np.diff(g)
    m_ibi = np.diff(m)
    min_len = min(len(g_ibi), len(m_ibi))
    g_ibi = g_ibi[:min_len]
    m_ibi = m_ibi[:min_len]
    g_bpm = 60.0 / np.clip(g_ibi, 0.1, 10)
    m_bpm = 60.0 / np.clip(m_ibi, 0.1, 10)
    drift = float(np.mean(g_bpm) - np.mean(m_bpm))
    stability = float(1.0 / (1.0 + np.std(g_bpm - m_bpm)))
    return {"drift_bpm": drift, "stability": stability}


def synchronization_index(tap_times_list: list[list[float]]) -> float:
    if len(tap_times_list) < 2:
        return 1.0
    all_phases = []
    for taps in tap_times_list:
        if len(taps) < 2:
            continue
        ibi = np.diff(taps)
        med_ibi = np.median(ibi)
        if med_ibi <= 0:
            continue
        phases = (np.array(taps) % med_ibi) / med_ibi * 2 * np.pi
        all_phases.extend(phases.tolist())
    if not all_phases:
        return 0.0
    z = np.mean(np.exp(1j * np.array(all_phases)))
    return float(np.abs(z))
