"""Conflict resolution for concurrent collaborative music editing."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EditOp:
    """A single edit operation on a musical score region."""
    user_id: str
    timestamp: float
    region_start: float  # beat position
    region_end: float
    op_type: str  # 'insert', 'delete', 'modify'
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        raw = f"{self.user_id}:{self.timestamp}:{self.region_start}:{self.region_end}:{self.op_type}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def regions_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def detect_conflicts(ops: list[EditOp]) -> list[tuple[EditOp, EditOp]]:
    """Find all pairs of overlapping edit operations from different users."""
    conflicts: list[tuple[EditOp, EditOp]] = []
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            if ops[i].user_id == ops[j].user_id:
                continue
            if regions_overlap(ops[i].region_start, ops[i].region_end,
                               ops[j].region_start, ops[j].region_end):
                conflicts.append((ops[i], ops[j]))
    return conflicts


def resolve_last_writer_wins(conflicts: list[tuple[EditOp, EditOp]]) -> list[EditOp]:
    """Resolve each conflict by keeping the latest-timestamped edit."""
    winners: list[EditOp] = []
    seen: set[str] = set()
    for a, b in conflicts:
        winner = a if a.timestamp >= b.timestamp else b
        if winner.fingerprint not in seen:
            winners.append(winner)
            seen.add(winner.fingerprint)
    return winners


def merge_non_overlapping(ops: list[EditOp]) -> list[EditOp]:
    """Return a de-duplicated, chronologically sorted list of non-conflicting ops."""
    sorted_ops = sorted(ops, key=lambda o: (o.region_start, o.timestamp))
    merged: list[EditOp] = []
    for op in sorted_ops:
        if not merged or not regions_overlap(merged[-1].region_start, merged[-1].region_end,
                                              op.region_start, op.region_end):
            merged.append(op)
        else:
            if op.timestamp > merged[-1].timestamp:
                merged[-1] = op
    return merged


@dataclass
class ConflictReport:
    total_ops: int
    conflicts_found: int
    resolved_ops: list[EditOp]
    dropped_ops: list[EditOp]


def full_resolution_pipeline(ops: list[EditOp]) -> ConflictReport:
    """Run full conflict detection and resolution."""
    conflicts = detect_conflicts(ops)
    if not conflicts:
        return ConflictReport(
            total_ops=len(ops), conflicts_found=0,
            resolved_ops=sorted(ops, key=lambda o: o.region_start),
            dropped_ops=[],
        )
    winners = resolve_last_writer_wins(conflicts)
    loser_fps = set()
    for a, b in conflicts:
        winner = a if a.timestamp >= b.timestamp else b
        loser = b if winner is a else a
        loser_fps.add(loser.fingerprint)

    all_fps = {op.fingerprint for op in ops}
    conflict_fps = set()
    for a, b in conflicts:
        conflict_fps.add(a.fingerprint)
        conflict_fps.add(b.fingerprint)
    non_conflict = [op for op in ops if op.fingerprint not in conflict_fps]
    resolved = sorted(non_conflict + winners, key=lambda o: o.region_start)
    dropped = [op for op in ops if op.fingerprint in loser_fps]
    return ConflictReport(
        total_ops=len(ops), conflicts_found=len(conflicts),
        resolved_ops=resolved, dropped_ops=dropped,
    )
