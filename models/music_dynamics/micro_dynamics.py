"""Micro-dynamics: note-level velocity shaping, expressive timing, humanization models."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PerformanceStyle(str, Enum):
    MECHANICAL = "mechanical"
    SUBTLE = "subtle"
    EXPRESSIVE = "expressive"
    ROMANTIC = "romantic"
    JAZZ = "jazz"


STYLE_PARAMS: Dict[PerformanceStyle, Dict[str, float]] = {
    PerformanceStyle.MECHANICAL: {"vel_jitter": 0.0, "timing_jitter_ms": 0.0, "swing": 0.0},
    PerformanceStyle.SUBTLE: {"vel_jitter": 3.0, "timing_jitter_ms": 5.0, "swing": 0.0},
    PerformanceStyle.EXPRESSIVE: {"vel_jitter": 8.0, "timing_jitter_ms": 15.0, "swing": 0.0},
    PerformanceStyle.ROMANTIC: {"vel_jitter": 12.0, "timing_jitter_ms": 25.0, "swing": 0.0},
    PerformanceStyle.JAZZ: {"vel_jitter": 10.0, "timing_jitter_ms": 12.0, "swing": 0.6},
}


@dataclass
class NoteEvent:
    onset_ms: float
    duration_ms: float
    velocity: int
    pitch: int = 60


@dataclass
class HumanizedNote:
    original: NoteEvent
    onset_ms: float
    duration_ms: float
    velocity: int


@dataclass
class MicroDynamicProfile:
    velocity_std: float
    timing_std_ms: float
    velocity_range: Tuple[int, int]
    avg_velocity: float
    downbeat_accent: float


def humanize_velocity(velocity: int, jitter: float, seed: Optional[int] = None) -> int:
    rng = random.Random(seed)
    offset = rng.gauss(0, jitter)
    return max(1, min(127, round(velocity + offset)))


def humanize_timing(onset_ms: float, jitter_ms: float, seed: Optional[int] = None) -> float:
    rng = random.Random(seed)
    return max(0.0, onset_ms + rng.gauss(0, jitter_ms))


def apply_swing(notes: List[NoteEvent], swing_ratio: float = 0.6,
                beat_duration_ms: float = 500.0) -> List[NoteEvent]:
    if swing_ratio <= 0.5:
        return list(notes)
    result: List[NoteEvent] = []
    for note in notes:
        beat_pos = (note.onset_ms % beat_duration_ms) / beat_duration_ms
        if 0.45 < beat_pos < 0.55:
            swing_offset = (swing_ratio - 0.5) * beat_duration_ms
            new_onset = note.onset_ms + swing_offset
            result.append(NoteEvent(onset_ms=new_onset, duration_ms=note.duration_ms,
                                    velocity=note.velocity, pitch=note.pitch))
        else:
            result.append(note)
    return result


def humanize(notes: List[NoteEvent], style: PerformanceStyle = PerformanceStyle.EXPRESSIVE,
             seed: Optional[int] = None) -> List[HumanizedNote]:
    params = STYLE_PARAMS[style]
    rng = random.Random(seed)
    result: List[HumanizedNote] = []
    for note in notes:
        vel = humanize_velocity(note.velocity, params["vel_jitter"], rng.randint(0, 2**31))
        onset = humanize_timing(note.onset_ms, params["timing_jitter_ms"], rng.randint(0, 2**31))
        dur_jitter = params["timing_jitter_ms"] * 0.5
        dur = max(10.0, note.duration_ms + rng.gauss(0, dur_jitter))
        result.append(HumanizedNote(original=note, onset_ms=onset,
                                    duration_ms=dur, velocity=vel))
    if params["swing"] > 0.5:
        swung = apply_swing(
            [NoteEvent(h.onset_ms, h.duration_ms, h.velocity, h.original.pitch)
             for h in result],
            params["swing"]
        )
        for i, s in enumerate(swung):
            result[i] = HumanizedNote(original=result[i].original,
                                      onset_ms=s.onset_ms, duration_ms=s.duration_ms,
                                      velocity=s.velocity)
    return result


def downbeat_accent(notes: List[NoteEvent], beat_duration_ms: float = 500.0,
                    accent_amount: int = 10) -> List[NoteEvent]:
    result: List[NoteEvent] = []
    for note in notes:
        beat_pos = (note.onset_ms % beat_duration_ms) / beat_duration_ms
        if beat_pos < 0.05:
            vel = min(127, note.velocity + accent_amount)
        else:
            vel = note.velocity
        result.append(NoteEvent(onset_ms=note.onset_ms, duration_ms=note.duration_ms,
                                velocity=vel, pitch=note.pitch))
    return result


def analyze_micro_dynamics(notes: List[NoteEvent], beat_duration_ms: float = 500.0) -> MicroDynamicProfile:
    if not notes:
        return MicroDynamicProfile(0.0, 0.0, (0, 0), 0.0, 0.0)
    vels = [n.velocity for n in notes]
    avg_v = sum(vels) / len(vels)
    vel_std = math.sqrt(sum((v - avg_v) ** 2 for v in vels) / len(vels))
    onsets = sorted(n.onset_ms for n in notes)
    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    if iois:
        avg_ioi = sum(iois) / len(iois)
        timing_std = math.sqrt(sum((t - avg_ioi) ** 2 for t in iois) / len(iois))
    else:
        timing_std = 0.0
    downbeat_vels = [n.velocity for n in notes
                     if (n.onset_ms % beat_duration_ms) / beat_duration_ms < 0.05]
    offbeat_vels = [n.velocity for n in notes
                    if (n.onset_ms % beat_duration_ms) / beat_duration_ms >= 0.05]
    accent = 0.0
    if downbeat_vels and offbeat_vels:
        accent = (sum(downbeat_vels) / len(downbeat_vels)) - (sum(offbeat_vels) / len(offbeat_vels))
    return MicroDynamicProfile(velocity_std=vel_std, timing_std_ms=timing_std,
                                velocity_range=(min(vels), max(vels)),
                                avg_velocity=avg_v, downbeat_accent=accent)
