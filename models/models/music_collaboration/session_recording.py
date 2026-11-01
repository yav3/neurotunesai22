"""Session recording — capture, replay, and analyze collaborative music sessions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math


class EventType(Enum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    CONTROL_CHANGE = "control_change"
    TEMPO_CHANGE = "tempo_change"
    MARKER = "marker"
    CHAT = "chat"


@dataclass
class SessionEvent:
    timestamp: float
    event_type: EventType
    player_id: str
    data: dict = field(default_factory=dict)


@dataclass
class SessionRecording:
    session_id: str
    events: list[SessionEvent] = field(default_factory=list)
    start_time: float = 0.0

    @property
    def duration(self) -> float:
        if not self.events:
            return 0.0
        return max(e.timestamp for e in self.events) - self.start_time

    @property
    def player_ids(self) -> set[str]:
        return {e.player_id for e in self.events}

    @property
    def event_count(self) -> int:
        return len(self.events)


def record_event(recording: SessionRecording, event: SessionEvent) -> None:
    recording.events.append(event)


def events_in_range(recording: SessionRecording, start: float, end: float) -> list[SessionEvent]:
    return [e for e in recording.events if start <= e.timestamp <= end]


def player_activity(recording: SessionRecording, player_id: str) -> list[SessionEvent]:
    return [e for e in recording.events if e.player_id == player_id]


def activity_density(recording: SessionRecording, window_sec: float = 1.0) -> list[float]:
    if not recording.events or recording.duration <= 0:
        return []
    n_windows = max(1, int(math.ceil(recording.duration / window_sec)))
    density = [0.0] * n_windows
    for e in recording.events:
        idx = min(int((e.timestamp - recording.start_time) / window_sec), n_windows - 1)
        density[idx] += 1
    return density


def player_contribution(recording: SessionRecording) -> dict[str, float]:
    total = len(recording.events)
    if total == 0:
        return {}
    counts: dict[str, int] = {}
    for e in recording.events:
        counts[e.player_id] = counts.get(e.player_id, 0) + 1
    return {pid: c / total for pid, c in counts.items()}


def interaction_matrix(recording: SessionRecording, window_sec: float = 2.0) -> dict[tuple[str, str], int]:
    pairs: dict[tuple[str, str], int] = {}
    sorted_events = sorted(recording.events, key=lambda e: e.timestamp)
    for i, e1 in enumerate(sorted_events):
        for j in range(i + 1, len(sorted_events)):
            e2 = sorted_events[j]
            if e2.timestamp - e1.timestamp > window_sec:
                break
            if e1.player_id != e2.player_id:
                key = tuple(sorted([e1.player_id, e2.player_id]))
                pairs[key] = pairs.get(key, 0) + 1
    return pairs


def session_fingerprint(recording: SessionRecording) -> str:
    data = f"{recording.session_id}:{recording.event_count}:{recording.duration}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def merge_recordings(a: SessionRecording, b: SessionRecording, new_id: str = "merged") -> SessionRecording:
    merged_events = sorted(a.events + b.events, key=lambda e: e.timestamp)
    start = min(a.start_time, b.start_time) if a.events and b.events else a.start_time
    return SessionRecording(session_id=new_id, events=merged_events, start_time=start)


def tempo_changes(recording: SessionRecording) -> list[tuple[float, float]]:
    return [
        (e.timestamp, e.data.get("bpm", 120.0))
        for e in recording.events
        if e.event_type == EventType.TEMPO_CHANGE
    ]


def note_events_only(recording: SessionRecording) -> list[SessionEvent]:
    return [e for e in recording.events if e.event_type in (EventType.NOTE_ON, EventType.NOTE_OFF)]
