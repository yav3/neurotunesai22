"""Score annotation and commenting — threaded discussions anchored to musical locations.

Supports:
- Anchoring annotations to beat/measure/voice/pitch ranges
- Threaded comment hierarchies with resolve/reopen lifecycle
- Annotation categories (performance, articulation, dynamics, harmony, rhythm, text)
- Annotation density and coverage metrics
- Search/filter across annotation sets
- Annotation merging across collaborator copies
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import numpy as np


class AnnotationCategory(Enum):
    PERFORMANCE = "performance"
    ARTICULATION = "articulation"
    DYNAMICS = "dynamics"
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    TEXT = "text"
    FINGERING = "fingering"
    PEDALING = "pedaling"
    BOWING = "bowing"
    GENERAL = "general"


class AnnotationStatus(Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class MusicAnchor:
    """A location range in a musical score.

    Positions use (measure, beat) coordinates.  beat is 1-based within
    the measure.  voice and pitch_range are optional narrowing filters.
    """
    measure_start: int
    beat_start: float = 1.0
    measure_end: int = -1      # -1 → same as start
    beat_end: float = -1.0     # -1 → same as beat_start
    voice: Optional[int] = None
    pitch_low: Optional[int] = None   # MIDI
    pitch_high: Optional[int] = None

    def effective_end(self) -> Tuple[int, float]:
        m = self.measure_end if self.measure_end >= 0 else self.measure_start
        b = self.beat_end if self.beat_end >= 0 else self.beat_start
        return m, b

    def overlaps(self, other: MusicAnchor) -> bool:
        s1 = (self.measure_start, self.beat_start)
        e1 = self.effective_end()
        s2 = (other.measure_start, other.beat_start)
        e2 = other.effective_end()
        if e1 < s2 or e2 < s1:
            return False
        if self.voice is not None and other.voice is not None:
            if self.voice != other.voice:
                return False
        return True

    def span_beats(self, beats_per_measure: float = 4.0) -> float:
        me, be = self.effective_end()
        total_start = (self.measure_start - 1) * beats_per_measure + self.beat_start
        total_end = (me - 1) * beats_per_measure + be
        return max(0.0, total_end - total_start)


@dataclass
class Annotation:
    """A single annotation/comment on a score region."""
    annotation_id: str
    author: str
    anchor: MusicAnchor
    category: AnnotationCategory
    text: str
    status: AnnotationStatus = AnnotationStatus.OPEN
    created_at: float = 0.0
    updated_at: float = 0.0
    parent_id: Optional[str] = None  # for threaded replies
    tags: List[str] = field(default_factory=list)
    priority: int = 0  # 0=normal, 1=important, 2=critical

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at


def make_annotation_id(author: str, text: str, anchor: MusicAnchor) -> str:
    blob = f"{author}:{text}:{anchor.measure_start}:{anchor.beat_start}:{time.time()}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class AnnotationSet:
    """Collection of annotations on a score, with indexing for queries."""
    annotations: Dict[str, Annotation] = field(default_factory=dict)

    def add(self, ann: Annotation) -> None:
        self.annotations[ann.annotation_id] = ann

    def remove(self, annotation_id: str) -> bool:
        return self.annotations.pop(annotation_id, None) is not None

    def get(self, annotation_id: str) -> Optional[Annotation]:
        return self.annotations.get(annotation_id)

    def resolve(self, annotation_id: str) -> bool:
        ann = self.annotations.get(annotation_id)
        if ann is None:
            return False
        ann.status = AnnotationStatus.RESOLVED
        ann.updated_at = time.time()
        return True

    def reopen(self, annotation_id: str) -> bool:
        ann = self.annotations.get(annotation_id)
        if ann is None:
            return False
        ann.status = AnnotationStatus.OPEN
        ann.updated_at = time.time()
        return True

    def thread(self, parent_id: str) -> List[Annotation]:
        return sorted(
            [a for a in self.annotations.values() if a.parent_id == parent_id],
            key=lambda a: a.created_at,
        )

    def roots(self) -> List[Annotation]:
        return sorted(
            [a for a in self.annotations.values() if a.parent_id is None],
            key=lambda a: (a.anchor.measure_start, a.anchor.beat_start),
        )

    def by_anchor_range(self, measure_lo: int, measure_hi: int) -> List[Annotation]:
        out = []
        for a in self.annotations.values():
            me, _ = a.anchor.effective_end()
            if a.anchor.measure_start <= measure_hi and me >= measure_lo:
                out.append(a)
        return sorted(out, key=lambda a: (a.anchor.measure_start, a.anchor.beat_start))

    def by_category(self, cat: AnnotationCategory) -> List[Annotation]:
        return [a for a in self.annotations.values() if a.category == cat]

    def by_author(self, author: str) -> List[Annotation]:
        return [a for a in self.annotations.values() if a.author == author]

    def by_status(self, status: AnnotationStatus) -> List[Annotation]:
        return [a for a in self.annotations.values() if a.status == status]

    def by_tag(self, tag: str) -> List[Annotation]:
        return [a for a in self.annotations.values() if tag in a.tags]


def annotation_density(aset: AnnotationSet, total_measures: int,
                       beats_per_measure: float = 4.0) -> np.ndarray:
    """Per-measure annotation density (count of annotations touching each measure)."""
    density = np.zeros(total_measures)
    for a in aset.annotations.values():
        m_start = max(0, a.anchor.measure_start - 1)
        me, _ = a.anchor.effective_end()
        m_end = min(total_measures, me)
        for m in range(m_start, m_end):
            density[m] += 1
    return density


def annotation_coverage(aset: AnnotationSet, total_measures: int) -> float:
    """Fraction of measures that have at least one annotation."""
    touched: Set[int] = set()
    for a in aset.annotations.values():
        m_start = a.anchor.measure_start
        me, _ = a.anchor.effective_end()
        for m in range(m_start, me + 1):
            touched.add(m)
    if total_measures <= 0:
        return 0.0
    return len(touched) / total_measures


def category_distribution(aset: AnnotationSet) -> Dict[str, int]:
    """Count annotations per category."""
    counts: Dict[str, int] = {}
    for a in aset.annotations.values():
        key = a.category.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def author_activity(aset: AnnotationSet) -> Dict[str, int]:
    """Count annotations per author."""
    counts: Dict[str, int] = {}
    for a in aset.annotations.values():
        counts[a.author] = counts.get(a.author, 0) + 1
    return counts


def resolution_rate(aset: AnnotationSet) -> float:
    """Fraction of root annotations that are resolved."""
    roots = [a for a in aset.annotations.values() if a.parent_id is None]
    if not roots:
        return 1.0
    resolved = sum(1 for a in roots if a.status == AnnotationStatus.RESOLVED)
    return resolved / len(roots)


def priority_summary(aset: AnnotationSet) -> Dict[int, int]:
    """Count open annotations per priority level."""
    counts: Dict[int, int] = {}
    for a in aset.annotations.values():
        if a.status == AnnotationStatus.OPEN:
            counts[a.priority] = counts.get(a.priority, 0) + 1
    return counts


def merge_annotation_sets(base: AnnotationSet, theirs: AnnotationSet) -> AnnotationSet:
    """Merge two annotation sets, preferring newer updates for shared IDs."""
    merged = AnnotationSet()
    all_ids = set(base.annotations) | set(theirs.annotations)
    for aid in all_ids:
        a_base = base.annotations.get(aid)
        a_theirs = theirs.annotations.get(aid)
        if a_base is None:
            merged.add(a_theirs)
        elif a_theirs is None:
            merged.add(a_base)
        else:
            winner = a_theirs if a_theirs.updated_at >= a_base.updated_at else a_base
            merged.add(winner)
    return merged


def hotspot_measures(aset: AnnotationSet, total_measures: int,
                     top_k: int = 5) -> List[Tuple[int, int]]:
    """Return the top-k measures with most annotations (measure_number, count)."""
    density = annotation_density(aset, total_measures)
    indices = np.argsort(-density)[:top_k]
    return [(int(i) + 1, int(density[i])) for i in indices if density[i] > 0]


def unresolved_critical(aset: AnnotationSet) -> List[Annotation]:
    """All open annotations with priority >= 2 (critical)."""
    return [
        a for a in aset.annotations.values()
        if a.status == AnnotationStatus.OPEN and a.priority >= 2
    ]
