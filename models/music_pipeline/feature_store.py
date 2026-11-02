"""Feature store: cache, version, and serve pre-computed audio features.

Content-addressed storage for beat-synchronous, frame-level, and
track-level features. Supports schema versioning, lazy loading,
batch retrieval, and staleness detection for ML pipeline feeds.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class FeatureSchema:
    name: str
    version: int
    dtype: str
    shape_hint: Tuple[int, ...]
    description: str = ""


@dataclass
class FeatureRecord:
    track_id: str
    schema: FeatureSchema
    data: np.ndarray
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def content_hash(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


class InMemoryFeatureStore:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, FeatureRecord]] = {}
        self._schemas: Dict[str, FeatureSchema] = {}

    def register_schema(self, schema: FeatureSchema) -> None:
        self._schemas[schema.name] = schema

    def put(self, track_id: str, feature_name: str,
            data: np.ndarray, metadata: Optional[Dict] = None) -> str:
        schema = self._schemas.get(feature_name)
        if schema is None:
            schema = FeatureSchema(name=feature_name, version=1,
                                   dtype=str(data.dtype),
                                   shape_hint=data.shape)
            self._schemas[feature_name] = schema
        ch = content_hash(data)
        record = FeatureRecord(
            track_id=track_id, schema=schema,
            data=data, content_hash=ch,
            metadata=metadata or {},
        )
        if track_id not in self._store:
            self._store[track_id] = {}
        self._store[track_id][feature_name] = record
        return ch

    def get(self, track_id: str, feature_name: str) -> Optional[FeatureRecord]:
        return self._store.get(track_id, {}).get(feature_name)

    def get_batch(self, track_ids: List[str],
                  feature_name: str) -> Dict[str, Optional[FeatureRecord]]:
        return {tid: self.get(tid, feature_name) for tid in track_ids}

    def has(self, track_id: str, feature_name: str) -> bool:
        return feature_name in self._store.get(track_id, {})

    def list_tracks(self) -> List[str]:
        return list(self._store.keys())

    def list_features(self, track_id: str) -> List[str]:
        return list(self._store.get(track_id, {}).keys())

    def is_stale(self, track_id: str, feature_name: str,
                 current_schema_version: int) -> bool:
        record = self.get(track_id, feature_name)
        if record is None:
            return True
        return record.schema.version < current_schema_version

    def invalidate(self, track_id: str,
                   feature_name: Optional[str] = None) -> int:
        if track_id not in self._store:
            return 0
        if feature_name:
            if feature_name in self._store[track_id]:
                del self._store[track_id][feature_name]
                return 1
            return 0
        count = len(self._store[track_id])
        del self._store[track_id]
        return count

    def stats(self) -> Dict[str, Any]:
        n_tracks = len(self._store)
        n_records = sum(len(v) for v in self._store.values())
        total_bytes = sum(r.data.nbytes
                         for feats in self._store.values()
                         for r in feats.values())
        return {
            "n_tracks": n_tracks,
            "n_records": n_records,
            "total_bytes": total_bytes,
            "n_schemas": len(self._schemas),
        }


def feature_matrix(store: InMemoryFeatureStore,
                   track_ids: List[str],
                   feature_name: str) -> Optional[np.ndarray]:
    records = store.get_batch(track_ids, feature_name)
    arrays = []
    for tid in track_ids:
        r = records.get(tid)
        if r is None:
            return None
        arrays.append(r.data.flatten())
    return np.stack(arrays)


def stale_report(store: InMemoryFeatureStore,
                 schemas: Dict[str, int]) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    for tid in store.list_tracks():
        for fname, version in schemas.items():
            if store.is_stale(tid, fname, version):
                report.setdefault(fname, []).append(tid)
    return report
