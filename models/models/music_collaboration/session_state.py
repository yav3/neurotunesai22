"""Live session management for collaborative music-making."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SessionRole(Enum):
    LEADER = "leader"
    ACCOMPANIST = "accompanist"
    SOLOIST = "soloist"
    RHYTHM_SECTION = "rhythm_section"
    LISTENER = "listener"


class AudioQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"


@dataclass
class Participant:
    id: str
    role: SessionRole = SessionRole.ACCOMPANIST
    instrument: str = "unknown"
    latency_ms: float = 0.0
    reliability_score: float = 1.0


@dataclass
class SessionState:
    tempo_bpm: float = 120.0
    key: str = "C"
    time_sig: tuple[int, int] = (4, 4)
    active_participants: list[Participant] = field(default_factory=list)
    measure_number: int = 1
    is_recording: bool = False

    def add_participant(self, p: Participant) -> None:
        self.active_participants.append(p)

    def remove_participant(self, pid: str) -> None:
        self.active_participants = [p for p in self.active_participants if p.id != pid]

    @property
    def n_participants(self) -> int:
        return len(self.active_participants)

    def beat_duration_ms(self) -> float:
        if self.tempo_bpm <= 0:
            return 500.0
        return 60000.0 / self.tempo_bpm

    def measure_duration_ms(self) -> float:
        return self.beat_duration_ms() * self.time_sig[0]


def compute_latency_offset(participant: Participant) -> float:
    return participant.latency_ms / 2.0


def sync_point(participants: list[Participant]) -> float:
    if not participants:
        return 0.0
    latencies = [p.latency_ms for p in participants]
    return max(latencies) / 2.0


def role_assignment(
    participants: list[Participant],
    piece_requirements: dict[str, int] | None = None,
) -> dict[str, SessionRole]:
    if not participants:
        return {}
    sorted_p = sorted(participants, key=lambda p: p.latency_ms)
    assignments: dict[str, SessionRole] = {}
    assignments[sorted_p[0].id] = SessionRole.LEADER
    rhythm_instruments = {"drums", "bass", "percussion", "drum kit", "bass guitar"}
    solo_instruments = {"violin", "flute", "trumpet", "voice", "saxophone", "guitar"}
    for p in sorted_p[1:]:
        inst_lower = p.instrument.lower()
        if inst_lower in rhythm_instruments:
            assignments[p.id] = SessionRole.RHYTHM_SECTION
        elif inst_lower in solo_instruments and SessionRole.SOLOIST not in assignments.values():
            assignments[p.id] = SessionRole.SOLOIST
        else:
            assignments[p.id] = SessionRole.ACCOMPANIST
    return assignments


QUALITY_BITRATES = {
    AudioQuality.LOW: 64,
    AudioQuality.MEDIUM: 128,
    AudioQuality.HIGH: 256,
    AudioQuality.LOSSLESS: 1411,
}


def bandwidth_estimate(n_participants: int, quality: AudioQuality = AudioQuality.MEDIUM) -> float:
    per_stream_kbps = QUALITY_BITRATES.get(quality, 128)
    send = per_stream_kbps
    receive = per_stream_kbps * max(0, n_participants - 1)
    return float(send + receive)


def max_participants_for_bandwidth(
    available_kbps: float, quality: AudioQuality = AudioQuality.MEDIUM,
) -> int:
    per_stream = QUALITY_BITRATES.get(quality, 128)
    if per_stream <= 0:
        return 0
    return max(1, int(available_kbps / per_stream))


def jitter_buffer_size(latency_ms: float, jitter_ms: float = 10.0) -> float:
    return latency_ms + 2 * jitter_ms
