"""Metadata reconciliation: fuzzy matching across sources, conflict resolution.

Given metadata for the same track from multiple sources (e.g. MusicBrainz,
Discogs, user tags, audio analysis), reconcile into a single canonical record
with provenance tracking and confidence estimates. Pure numpy/scipy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ConflictStrategy(Enum):
    TRUST_ORDER = "trust_order"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    MOST_COMPLETE = "most_complete"


class FieldType(Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    DATE = "date"


FIELD_TYPES: dict[str, FieldType] = {
    "title": FieldType.TEXT, "artist": FieldType.TEXT,
    "album": FieldType.TEXT, "album_artist": FieldType.TEXT,
    "composer": FieldType.TEXT, "lyricist": FieldType.TEXT,
    "producer": FieldType.TEXT, "label": FieldType.TEXT,
    "genre": FieldType.CATEGORICAL, "mood": FieldType.CATEGORICAL,
    "language": FieldType.CATEGORICAL, "key": FieldType.CATEGORICAL,
    "bpm": FieldType.NUMERIC, "duration": FieldType.NUMERIC,
    "track_number": FieldType.NUMERIC, "disc_number": FieldType.NUMERIC,
    "year": FieldType.DATE,
    "isrc": FieldType.IDENTIFIER, "iswc": FieldType.IDENTIFIER,
    "musicbrainz_id": FieldType.IDENTIFIER, "acoustid": FieldType.IDENTIFIER,
    "barcode": FieldType.IDENTIFIER,
}


@dataclass
class SourceRecord:
    source_id: str
    metadata: dict[str, Any]
    trust_weight: float = 1.0
    timestamp: float | None = None


@dataclass
class FieldResolution:
    field_name: str
    chosen_value: Any
    confidence: float
    source_id: str
    alternatives: list[tuple[str, Any, float]]
    conflict: bool


@dataclass
class ReconciliationResult:
    canonical: dict[str, Any]
    field_resolutions: list[FieldResolution]
    overall_confidence: float
    conflict_count: int
    source_coverage: dict[str, float]


# --- fuzzy text matching ---

def _ngrams(s: str, n: int = 2) -> set[str]:
    s = s.lower().strip()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def fuzzy_similarity(a: str, b: str) -> float:
    """Dice coefficient on character bigrams — fast fuzzy text similarity."""
    if not a or not b:
        return 0.0
    if a.lower().strip() == b.lower().strip():
        return 1.0
    ga, gb = _ngrams(a), _ngrams(b)
    if not ga or not gb:
        return 0.0
    return 2.0 * len(ga & gb) / (len(ga) + len(gb))


def _normalise_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def text_similarity(a: str, b: str) -> float:
    """Combined exact + fuzzy similarity for text fields."""
    na, nb = _normalise_text(a), _normalise_text(b)
    if na == nb:
        return 1.0
    return fuzzy_similarity(na, nb)


def numeric_similarity(a: float, b: float, tolerance: float = 0.05) -> float:
    """Relative similarity for numeric fields."""
    if a == b:
        return 1.0
    denom = max(abs(a), abs(b), 1e-9)
    rel_diff = abs(a - b) / denom
    if rel_diff <= tolerance:
        return 1.0
    return float(np.clip(1.0 - rel_diff, 0.0, 1.0))


def identifier_match(a: str, b: str) -> float:
    """Exact match for identifiers (case-insensitive, stripped)."""
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.0


def field_similarity(a: Any, b: Any, field_name: str) -> float:
    """Compute similarity between two field values based on field type."""
    ft = FIELD_TYPES.get(field_name, FieldType.TEXT)
    if a is None or b is None:
        return 0.0
    if ft == FieldType.TEXT:
        return text_similarity(str(a), str(b))
    elif ft == FieldType.NUMERIC or ft == FieldType.DATE:
        try:
            return numeric_similarity(float(a), float(b))
        except (ValueError, TypeError):
            return text_similarity(str(a), str(b))
    elif ft == FieldType.IDENTIFIER:
        return identifier_match(str(a), str(b))
    elif ft == FieldType.CATEGORICAL:
        return text_similarity(str(a), str(b))
    return 0.0


# --- record-level matching ---

def record_similarity(
    a: dict[str, Any],
    b: dict[str, Any],
    field_weights: dict[str, float] | None = None,
) -> float:
    """Overall similarity between two metadata records.

    Weighted average of per-field similarities, normalised by total weight
    of fields present in both records.
    """
    default_weights = {"title": 0.3, "artist": 0.3, "album": 0.15, "duration": 0.1, "year": 0.1, "isrc": 0.05}
    weights = field_weights or default_weights

    total_w = 0.0
    total_sim = 0.0
    for f, w in weights.items():
        va, vb = a.get(f), b.get(f)
        if va is not None and vb is not None:
            total_w += w
            total_sim += w * field_similarity(va, vb, f)

    return total_sim / total_w if total_w > 0 else 0.0


def find_duplicates(
    records: list[dict[str, Any]],
    threshold: float = 0.85,
) -> list[tuple[int, int, float]]:
    """Find candidate duplicate pairs above similarity threshold.

    Returns list of (idx_a, idx_b, similarity).
    """
    n = len(records)
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = record_similarity(records[i], records[j])
            if sim >= threshold:
                pairs.append((i, j, sim))
    pairs.sort(key=lambda x: -x[2])
    return pairs


# --- conflict resolution ---

def _resolve_trust_order(
    values: list[tuple[str, Any, float]],
) -> tuple[Any, str, float]:
    """Pick the value from the highest-trust source."""
    best = max(values, key=lambda x: x[2])
    return best[1], best[0], best[2] / sum(v[2] for v in values)


def _resolve_majority_vote(
    values: list[tuple[str, Any, float]],
    field_name: str,
) -> tuple[Any, str, float]:
    """Pick the value agreed upon by the most sources (weighted)."""
    clusters: list[tuple[Any, str, float]] = []
    for src, val, w in values:
        placed = False
        for i, (cv, cs, cw) in enumerate(clusters):
            if field_similarity(val, cv, field_name) > 0.85:
                clusters[i] = (cv, cs, cw + w)
                placed = True
                break
        if not placed:
            clusters.append((val, src, w))

    best = max(clusters, key=lambda x: x[2])
    total_w = sum(c[2] for c in clusters)
    return best[0], best[1], best[2] / total_w if total_w > 0 else 0.5


def _resolve_confidence_weighted(
    values: list[tuple[str, Any, float]],
    field_name: str,
) -> tuple[Any, str, float]:
    """For numeric fields: weighted mean. For text: weighted majority."""
    ft = FIELD_TYPES.get(field_name, FieldType.TEXT)

    if ft == FieldType.NUMERIC:
        try:
            nums = [(float(v), w) for _, v, w in values]
            total_w = sum(w for _, w in nums)
            if total_w > 0:
                weighted_mean = sum(n * w for n, w in nums) / total_w
                weighted_var = sum(w * (n - weighted_mean) ** 2 for n, w in nums) / total_w
                std = math.sqrt(weighted_var) if weighted_var > 0 else 0
                rel_std = std / (abs(weighted_mean) + 1e-9)
                conf = float(np.clip(1.0 - rel_std, 0.0, 1.0))
                best_src = max(values, key=lambda x: x[2])[0]
                return weighted_mean, best_src, conf
        except (ValueError, TypeError):
            pass

    return _resolve_majority_vote(values, field_name)


def resolve_field(
    field_name: str,
    values: list[tuple[str, Any, float]],
    strategy: ConflictStrategy = ConflictStrategy.CONFIDENCE_WEIGHTED,
) -> FieldResolution:
    """Resolve a single field across multiple sources."""
    non_null = [(s, v, w) for s, v, w in values if v is not None and str(v).strip()]

    if not non_null:
        return FieldResolution(
            field_name=field_name,
            chosen_value=None,
            confidence=0.0,
            source_id="",
            alternatives=[],
            conflict=False,
        )

    if len(non_null) == 1:
        s, v, w = non_null[0]
        return FieldResolution(
            field_name=field_name,
            chosen_value=v,
            confidence=min(w, 1.0),
            source_id=s,
            alternatives=[],
            conflict=False,
        )

    # detect conflict
    ref_val = non_null[0][1]
    has_conflict = any(
        field_similarity(v, ref_val, field_name) < 0.85 for _, v, _ in non_null[1:]
    )

    if strategy == ConflictStrategy.TRUST_ORDER:
        chosen, src, conf = _resolve_trust_order(non_null)
    elif strategy == ConflictStrategy.MAJORITY_VOTE:
        chosen, src, conf = _resolve_majority_vote(non_null, field_name)
    elif strategy == ConflictStrategy.CONFIDENCE_WEIGHTED:
        chosen, src, conf = _resolve_confidence_weighted(non_null, field_name)
    else:  # MOST_COMPLETE — pick source with most non-null fields
        chosen, src, conf = _resolve_trust_order(non_null)

    alternatives = [(s, v, w) for s, v, w in non_null if s != src]

    return FieldResolution(
        field_name=field_name,
        chosen_value=chosen,
        confidence=conf,
        source_id=src,
        alternatives=alternatives,
        conflict=has_conflict,
    )


# --- full reconciliation ---

def reconcile(
    sources: list[SourceRecord],
    strategy: ConflictStrategy = ConflictStrategy.CONFIDENCE_WEIGHTED,
    fields: list[str] | None = None,
) -> ReconciliationResult:
    """Reconcile metadata from multiple sources into a canonical record.

    Parameters
    ----------
    sources : list of SourceRecord
        Each source contributes a metadata dict with a trust weight.
    strategy : ConflictStrategy
        How to resolve disagreements.
    fields : optional list of field names
        If None, uses union of all fields across all sources.

    Returns
    -------
    ReconciliationResult with canonical record, per-field resolutions,
    overall confidence, and source coverage statistics.
    """
    if not sources:
        return ReconciliationResult(
            canonical={}, field_resolutions=[], overall_confidence=0.0,
            conflict_count=0, source_coverage={},
        )

    all_fields = fields or sorted(
        set().union(*(s.metadata.keys() for s in sources))
    )

    resolutions: list[FieldResolution] = []
    canonical: dict[str, Any] = {}

    for f in all_fields:
        values = [
            (s.source_id, s.metadata.get(f), s.trust_weight)
            for s in sources
        ]
        res = resolve_field(f, values, strategy)
        resolutions.append(res)
        if res.chosen_value is not None:
            canonical[f] = res.chosen_value

    conflict_count = sum(1 for r in resolutions if r.conflict)
    confidences = [r.confidence for r in resolutions if r.chosen_value is not None]
    overall_conf = float(np.mean(confidences)) if confidences else 0.0

    coverage: dict[str, float] = {}
    for s in sources:
        present = sum(1 for f in all_fields if _has_value(s.metadata, f))
        coverage[s.source_id] = present / len(all_fields) if all_fields else 0.0

    return ReconciliationResult(
        canonical=canonical,
        field_resolutions=resolutions,
        overall_confidence=overall_conf,
        conflict_count=conflict_count,
        source_coverage=coverage,
    )


def _has_value(meta: dict[str, Any], key: str) -> bool:
    v = meta.get(key)
    return v is not None and str(v).strip() != ""


# --- similarity matrix for catalogue dedup ---

def pairwise_similarity_matrix(
    records: list[dict[str, Any]],
) -> np.ndarray:
    """Compute n×n pairwise similarity matrix for a catalogue.

    Uses record_similarity with default weights.
    """
    n = len(records)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = record_similarity(records[i], records[j])
            sim[i, j] = s
            sim[j, i] = s
    return sim


def cluster_duplicates(
    records: list[dict[str, Any]],
    threshold: float = 0.85,
) -> list[list[int]]:
    """Group records into clusters of probable duplicates.

    Simple single-linkage on the similarity matrix.
    """
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if record_similarity(records[i], records[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return [idxs for idxs in groups.values() if len(idxs) > 1]


def merge_sources(
    sources: list[SourceRecord],
    strategy: ConflictStrategy = ConflictStrategy.CONFIDENCE_WEIGHTED,
) -> dict[str, Any]:
    """Convenience: reconcile and return just the canonical dict."""
    return reconcile(sources, strategy).canonical
