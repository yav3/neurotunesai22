"""Rhythmic patterns for improvisation: swing, syncopation, metric displacement."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class RhythmFeel(Enum):
    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    BOSSA = "bossa"
    CLAVE_23 = "clave_23"
    CLAVE_32 = "clave_32"
    TRESILLO = "tresillo"


CLAVE_PATTERNS: dict[str, list[float]] = {
    "son_32": [0.0, 1.5, 2.5, 4.0, 5.0],
    "son_23": [0.0, 1.0, 2.5, 4.0, 5.5],
    "rumba_32": [0.0, 1.5, 2.5, 4.0, 5.25],
    "rumba_23": [0.0, 0.75, 2.5, 4.0, 5.5],
    "bossa": [0.0, 1.5, 3.0, 4.0, 5.5],
}


@dataclass
class RhythmicEvent:
    onset_beat: float
    duration_beat: float
    accent: float = 0.8

    @property
    def offset_beat(self) -> float:
        return self.onset_beat + self.duration_beat


def swing_offset(position: float, swing_ratio: float = 0.67) -> float:
    """Apply swing feel to an eighth-note position.

    swing_ratio=0.5 → straight, 0.67 → triplet swing, 0.75 → hard swing.
    """
    beat_pos = position % 1.0
    beat_floor = math.floor(position)
    if beat_pos < 0.5:
        return beat_floor + beat_pos
    offset_in_half = (beat_pos - 0.5) / 0.5
    return beat_floor + swing_ratio + offset_in_half * (1.0 - swing_ratio)


def apply_swing(events: list[RhythmicEvent],
                swing_ratio: float = 0.67) -> list[RhythmicEvent]:
    """Apply swing to a list of rhythmic events."""
    return [
        RhythmicEvent(
            onset_beat=swing_offset(e.onset_beat, swing_ratio),
            duration_beat=e.duration_beat,
            accent=e.accent,
        )
        for e in events
    ]


def syncopation_score(onsets: list[float], meter: int = 4) -> float:
    """Measure syncopation: fraction of onsets on weak beats."""
    if not onsets:
        return 0.0
    weak_count = 0
    for o in onsets:
        pos_in_bar = o % meter
        if pos_in_bar % 1.0 != 0.0:
            weak_count += 1
        elif int(pos_in_bar) % 2 != 0:
            weak_count += 0.5
    return weak_count / len(onsets)


def metric_displacement(events: list[RhythmicEvent],
                        shift_beats: float) -> list[RhythmicEvent]:
    """Displace all events by a fixed number of beats."""
    return [
        RhythmicEvent(
            onset_beat=e.onset_beat + shift_beats,
            duration_beat=e.duration_beat,
            accent=e.accent,
        )
        for e in events
    ]


def clave_alignment(onsets: list[float], clave: str = "son_32") -> float:
    """Score how well onsets align with a clave pattern (0-1)."""
    pattern = CLAVE_PATTERNS.get(clave, CLAVE_PATTERNS["son_32"])
    if not onsets:
        return 0.0
    aligned = 0
    for o in onsets:
        pos = o % 8.0
        for cp in pattern:
            if abs(pos - cp) < 0.25 or abs(pos - (cp + 8.0)) < 0.25:
                aligned += 1
                break
    return aligned / len(onsets)


def generate_clave_pattern(
    clave: str = "son_32",
    bars: int = 2,
    duration: float = 0.5,
) -> list[RhythmicEvent]:
    """Generate rhythmic events from a clave pattern over N bars."""
    pattern = CLAVE_PATTERNS.get(clave, CLAVE_PATTERNS["son_32"])
    events = []
    bar_len = 4.0
    for bar in range(bars):
        for onset in pattern:
            events.append(RhythmicEvent(
                onset_beat=bar * bar_len * 2 + onset,
                duration_beat=duration,
                accent=1.0 if onset == 0.0 else 0.7,
            ))
    return events


def tresillo_pattern(bars: int = 1) -> list[RhythmicEvent]:
    """3+3+2 tresillo pattern."""
    pattern_eighths = [0.0, 1.5, 3.0]
    events = []
    for bar in range(bars):
        for onset in pattern_eighths:
            events.append(RhythmicEvent(
                onset_beat=bar * 4.0 + onset,
                duration_beat=0.5,
                accent=0.9 if onset == 0.0 else 0.7,
            ))
    return events


def anticipation_pattern(
    target_beats: list[float],
    anticipation_amount: float = 0.5,
) -> list[RhythmicEvent]:
    """Create anticipation events arriving before target beats."""
    return [
        RhythmicEvent(
            onset_beat=t - anticipation_amount,
            duration_beat=anticipation_amount + 0.25,
            accent=0.85,
        )
        for t in target_beats
    ]


def density_curve(events: list[RhythmicEvent],
                  window_beats: float = 4.0) -> list[tuple[float, float]]:
    """Compute rhythmic density over sliding windows. Returns (window_start, events_per_beat)."""
    if not events:
        return []
    max_time = max(e.offset_beat for e in events)
    result = []
    t = 0.0
    while t <= max_time:
        count = sum(1 for e in events if t <= e.onset_beat < t + window_beats)
        result.append((t, count / window_beats))
        t += window_beats / 2.0
    return result
