"""Musical version control — diff, merge, and revision tracking for collaborative scores."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DiffType(Enum):
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"


@dataclass
class NoteEvent:
    pitch: int
    onset_beat: float
    duration_beats: float
    voice: int = 0


@dataclass
class MusicalDiff:
    measure: int
    voice: int
    diff_type: DiffType
    old_notes: list[NoteEvent] = field(default_factory=list)
    new_notes: list[NoteEvent] = field(default_factory=list)


@dataclass
class MergeConflict:
    measure: int
    voice: int
    diff_a: MusicalDiff
    diff_b: MusicalDiff


@dataclass
class MusicalRevision:
    revision_id: str
    timestamp: float
    author: str
    diffs: list[MusicalDiff]
    parent_revision: Optional[str] = None


def _events_in_measure(events: list[NoteEvent], measure: int, beats_per_measure: float = 4.0) -> list[NoteEvent]:
    lo = measure * beats_per_measure
    hi = lo + beats_per_measure
    return [e for e in events if lo <= e.onset_beat < hi]


def _events_equal(a: list[NoteEvent], b: list[NoteEvent]) -> bool:
    if len(a) != len(b):
        return False
    sa = sorted(a, key=lambda e: (e.pitch, e.onset_beat, e.duration_beats, e.voice))
    sb = sorted(b, key=lambda e: (e.pitch, e.onset_beat, e.duration_beats, e.voice))
    return all(
        x.pitch == y.pitch and abs(x.onset_beat - y.onset_beat) < 1e-9
        and abs(x.duration_beats - y.duration_beats) < 1e-9 and x.voice == y.voice
        for x, y in zip(sa, sb)
    )


def compute_musical_diff(
    old_events: list[NoteEvent],
    new_events: list[NoteEvent],
    beats_per_measure: float = 4.0,
) -> list[MusicalDiff]:
    if not old_events and not new_events:
        return []
    all_events = old_events + new_events
    max_beat = max(e.onset_beat + e.duration_beats for e in all_events)
    n_measures = int(max_beat / beats_per_measure) + 1
    voices = sorted({e.voice for e in all_events})
    diffs: list[MusicalDiff] = []
    for m in range(n_measures):
        for v in voices:
            old_m = [e for e in _events_in_measure(old_events, m, beats_per_measure) if e.voice == v]
            new_m = [e for e in _events_in_measure(new_events, m, beats_per_measure) if e.voice == v]
            if _events_equal(old_m, new_m):
                continue
            if not old_m and new_m:
                diffs.append(MusicalDiff(m, v, DiffType.ADD, [], new_m))
            elif old_m and not new_m:
                diffs.append(MusicalDiff(m, v, DiffType.DELETE, old_m, []))
            else:
                diffs.append(MusicalDiff(m, v, DiffType.MODIFY, old_m, new_m))
    return diffs


def apply_diffs(events: list[NoteEvent], diffs: list[MusicalDiff], beats_per_measure: float = 4.0) -> list[NoteEvent]:
    result = list(events)
    for d in diffs:
        lo = d.measure * beats_per_measure
        hi = lo + beats_per_measure
        if d.diff_type in (DiffType.DELETE, DiffType.MODIFY):
            result = [e for e in result if not (lo <= e.onset_beat < hi and e.voice == d.voice)]
        if d.diff_type in (DiffType.ADD, DiffType.MODIFY):
            result.extend(d.new_notes)
    return result


def merge_revisions(
    base: list[NoteEvent],
    rev_a: list[NoteEvent],
    rev_b: list[NoteEvent],
    beats_per_measure: float = 4.0,
) -> tuple[list[NoteEvent], list[MergeConflict]]:
    diffs_a = compute_musical_diff(base, rev_a, beats_per_measure)
    diffs_b = compute_musical_diff(base, rev_b, beats_per_measure)
    a_keys = {(d.measure, d.voice): d for d in diffs_a}
    b_keys = {(d.measure, d.voice): d for d in diffs_b}
    conflicts: list[MergeConflict] = []
    merged_diffs: list[MusicalDiff] = []
    for key, da in a_keys.items():
        if key in b_keys:
            db = b_keys[key]
            if _events_equal(da.new_notes, db.new_notes):
                merged_diffs.append(da)
            else:
                conflicts.append(MergeConflict(key[0], key[1], da, db))
        else:
            merged_diffs.append(da)
    for key, db in b_keys.items():
        if key not in a_keys:
            merged_diffs.append(db)
    merged = apply_diffs(base, merged_diffs, beats_per_measure)
    return merged, conflicts


def revision_distance(events_a: list[NoteEvent], events_b: list[NoteEvent], beats_per_measure: float = 4.0) -> int:
    diffs = compute_musical_diff(events_a, events_b, beats_per_measure)
    return len({d.measure for d in diffs})


def make_revision(
    author: str,
    old_events: list[NoteEvent],
    new_events: list[NoteEvent],
    parent_revision: Optional[str] = None,
    beats_per_measure: float = 4.0,
) -> MusicalRevision:
    diffs = compute_musical_diff(old_events, new_events, beats_per_measure)
    content = f"{author}{time.time()}{len(diffs)}"
    rid = hashlib.sha256(content.encode()).hexdigest()[:12]
    return MusicalRevision(rid, time.time(), author, diffs, parent_revision)
