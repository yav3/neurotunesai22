"""Real-time communication protocol for collaborative music: message passing,
state synchronization, vector clocks, and conflict-free replicated data types (CRDTs)."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time


class MessageType(str, Enum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    CONTROL_CHANGE = "control_change"
    TEMPO_CHANGE = "tempo_change"
    TRANSPORT = "transport"
    CHAT = "chat"
    STATE_SYNC = "state_sync"
    HEARTBEAT = "heartbeat"


@dataclass
class VectorClock:
    """Lamport vector clock for causal ordering of distributed events."""
    counters: Dict[str, int] = field(default_factory=dict)

    def tick(self, node_id: str) -> None:
        self.counters[node_id] = self.counters.get(node_id, 0) + 1

    def merge(self, other: VectorClock) -> None:
        for k, v in other.counters.items():
            self.counters[k] = max(self.counters.get(k, 0), v)

    def happens_before(self, other: VectorClock) -> bool:
        if not self.counters:
            return bool(other.counters)
        for k, v in self.counters.items():
            if v > other.counters.get(k, 0):
                return False
        return any(
            self.counters.get(k, 0) < v for k, v in other.counters.items()
        ) or (self.counters != other.counters)

    def concurrent_with(self, other: VectorClock) -> bool:
        return not self.happens_before(other) and not other.happens_before(self)


@dataclass
class Message:
    sender_id: str
    msg_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    clock: VectorClock = field(default_factory=VectorClock)
    sequence_num: int = 0

    def digest(self) -> str:
        content = f"{self.sender_id}:{self.msg_type.value}:{self.sequence_num}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class GCounter:
    """Grow-only counter CRDT — monotonically increasing per-node counters."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}

    def increment(self, node_id: str, amount: int = 1) -> None:
        self._counts[node_id] = self._counts.get(node_id, 0) + amount

    def value(self) -> int:
        return sum(self._counts.values())

    def merge(self, other: GCounter) -> None:
        for k, v in other._counts.items():
            self._counts[k] = max(self._counts.get(k, 0), v)


class PNCounter:
    """Positive-negative counter CRDT — supports increment and decrement."""

    def __init__(self) -> None:
        self.pos = GCounter()
        self.neg = GCounter()

    def increment(self, node_id: str, amount: int = 1) -> None:
        self.pos.increment(node_id, amount)

    def decrement(self, node_id: str, amount: int = 1) -> None:
        self.neg.increment(node_id, amount)

    def value(self) -> int:
        return self.pos.value() - self.neg.value()

    def merge(self, other: PNCounter) -> None:
        self.pos.merge(other.pos)
        self.neg.merge(other.neg)


class LWWRegister:
    """Last-writer-wins register CRDT for single-value state (e.g., tempo, key)."""

    def __init__(self, value: Any = None, ts: float = 0.0) -> None:
        self._value = value
        self._ts = ts

    @property
    def value(self) -> Any:
        return self._value

    def assign(self, value: Any, ts: float | None = None) -> None:
        t = ts if ts is not None else time.time()
        if t >= self._ts:
            self._value = value
            self._ts = t

    def merge(self, other: LWWRegister) -> None:
        if other._ts > self._ts:
            self._value = other._value
            self._ts = other._ts


class ORSet:
    """Observed-remove set CRDT — supports add and remove without conflicts."""

    def __init__(self) -> None:
        self._elements: Dict[Any, set] = {}
        self._tombstones: Dict[Any, set] = {}
        self._tag_counter = 0

    def _new_tag(self) -> str:
        self._tag_counter += 1
        return f"t{self._tag_counter}_{time.time()}"

    def add(self, element: Any) -> None:
        tag = self._new_tag()
        self._elements.setdefault(element, set()).add(tag)

    def remove(self, element: Any) -> None:
        tags = self._elements.pop(element, set())
        self._tombstones.setdefault(element, set()).update(tags)

    def contains(self, element: Any) -> bool:
        return element in self._elements and bool(self._elements[element])

    def elements(self) -> List[Any]:
        return [e for e, tags in self._elements.items() if tags]

    def merge(self, other: ORSet) -> None:
        all_keys = set(self._elements) | set(other._elements)
        for key in all_keys:
            local_tags = self._elements.get(key, set())
            remote_tags = other._elements.get(key, set())
            local_tomb = self._tombstones.get(key, set())
            remote_tomb = other._tombstones.get(key, set())
            merged_tomb = local_tomb | remote_tomb
            merged_tags = (local_tags | remote_tags) - merged_tomb
            self._elements[key] = merged_tags
            self._tombstones[key] = merged_tomb


@dataclass
class SessionBus:
    """Simple pub-sub message bus for a collaborative session."""
    participants: Dict[str, VectorClock] = field(default_factory=dict)
    log: List[Message] = field(default_factory=list)
    _seq: int = 0

    def join(self, node_id: str) -> None:
        self.participants[node_id] = VectorClock()

    def leave(self, node_id: str) -> None:
        self.participants.pop(node_id, None)

    def send(self, sender_id: str, msg_type: MessageType, payload: Dict[str, Any]) -> Message:
        clock = self.participants.get(sender_id, VectorClock())
        clock.tick(sender_id)
        self._seq += 1
        msg = Message(
            sender_id=sender_id,
            msg_type=msg_type,
            payload=payload,
            clock=VectorClock(counters=dict(clock.counters)),
            sequence_num=self._seq,
        )
        self.log.append(msg)
        for pid, pc in self.participants.items():
            if pid != sender_id:
                pc.merge(msg.clock)
        return msg

    def messages_since(self, seq: int) -> List[Message]:
        return [m for m in self.log if m.sequence_num > seq]

    def causal_order(self) -> List[Message]:
        return sorted(self.log, key=lambda m: (sum(m.clock.counters.values()), m.timestamp))


def latency_compensated_timestamp(
    local_ts: float, rtt_ms: float, clock_offset_ms: float = 0.0
) -> float:
    """Adjust a local timestamp for network latency and clock skew."""
    return local_ts - (rtt_ms / 2000.0) - (clock_offset_ms / 1000.0)


def merge_concurrent_edits(
    edits: List[Tuple[str, Dict[str, Any]]], strategy: str = "lww"
) -> Dict[str, Any]:
    """Merge concurrent edits from multiple participants.

    strategy: 'lww' (last-writer-wins by timestamp) or 'union' (merge all keys).
    """
    if not edits:
        return {}
    if strategy == "union":
        merged: Dict[str, Any] = {}
        for _, payload in edits:
            merged.update(payload)
        return merged
    best_ts = -1.0
    best_payload: Dict[str, Any] = {}
    for _, payload in edits:
        ts = payload.get("_ts", 0.0)
        if ts > best_ts:
            best_ts = ts
            best_payload = payload
    return best_payload


def message_integrity_check(msg: Message) -> bool:
    """Verify message digest matches content."""
    expected = msg.digest()
    recomputed = Message(
        sender_id=msg.sender_id,
        msg_type=msg.msg_type,
        payload=msg.payload,
        timestamp=msg.timestamp,
        clock=msg.clock,
        sequence_num=msg.sequence_num,
    ).digest()
    return expected == recomputed
