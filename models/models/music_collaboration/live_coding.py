"""Live coding collaboration: real-time shared musical programming and performance."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class CodeLanguage(str, Enum):
    SUPERCOLLIDER = "supercollider"
    TIDAL_CYCLES = "tidal_cycles"
    SONIC_PI = "sonic_pi"
    FOXDOT = "foxdot"
    CUSTOM = "custom"


class SyncMode(str, Enum):
    QUANTIZED = "quantized"
    IMMEDIATE = "immediate"
    NEXT_BAR = "next_bar"
    NEXT_PHRASE = "next_phrase"


@dataclass
class CodeSnippet:
    code: str
    author: str
    language: CodeLanguage
    timestamp: float = field(default_factory=time.time)
    bar_number: int = 0

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()[:12]


@dataclass
class PerformerState:
    name: str
    active: bool = True
    current_snippet: Optional[CodeSnippet] = None
    muted: bool = False
    volume_db: float = 0.0
    history: List[CodeSnippet] = field(default_factory=list)


@dataclass
class SharedClock:
    bpm: float = 120.0
    beats_per_bar: int = 4
    current_beat: float = 0.0

    @property
    def beat_duration_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_duration_s(self) -> float:
        return self.beat_duration_s * self.beats_per_bar

    def next_bar_time(self, current_time: float, start_time: float = 0.0) -> float:
        elapsed = current_time - start_time
        bar_dur = self.bar_duration_s
        bars_elapsed = elapsed / bar_dur
        next_bar = (int(bars_elapsed) + 1) * bar_dur
        return start_time + next_bar


@dataclass
class LiveSession:
    performers: Dict[str, PerformerState] = field(default_factory=dict)
    clock: SharedClock = field(default_factory=SharedClock)
    sync_mode: SyncMode = SyncMode.NEXT_BAR
    history: List[Tuple[str, CodeSnippet]] = field(default_factory=list)


def submit_code(session: LiveSession, performer_name: str, snippet: CodeSnippet) -> bool:
    if performer_name not in session.performers:
        session.performers[performer_name] = PerformerState(name=performer_name)
    state = session.performers[performer_name]
    state.current_snippet = snippet
    state.history.append(snippet)
    session.history.append((performer_name, snippet))
    return True


def code_diff(old: str, new: str) -> Tuple[List[str], List[str], List[str]]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    added = [l for l in new_lines if l not in old_lines]
    removed = [l for l in old_lines if l not in new_lines]
    unchanged = [l for l in old_lines if l in new_lines]
    return (added, removed, unchanged)


def performer_activity(session: LiveSession, performer_name: str,
                       window_seconds: float = 60.0) -> float:
    state = session.performers.get(performer_name)
    if not state:
        return 0.0
    now = time.time()
    recent = [s for s in state.history if now - s.timestamp <= window_seconds]
    return len(recent) / max(window_seconds / 10.0, 1.0)


def session_entropy(session: LiveSession) -> float:
    import math
    total = len(session.history)
    if total == 0:
        return 0.0
    counts: Dict[str, int] = {}
    for name, _ in session.history:
        counts[name] = counts.get(name, 0) + 1
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def quantize_to_bar(clock: SharedClock, timestamp: float, start_time: float = 0.0) -> int:
    elapsed = timestamp - start_time
    return int(elapsed / clock.bar_duration_s)


def active_performers(session: LiveSession) -> List[str]:
    return [name for name, state in session.performers.items() if state.active and not state.muted]
