"""Data versioning and lineage tracking for music datasets.

Content-addressed hashing, transform provenance recording,
and dataset snapshot management.  Pure Python + numpy.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TransformKind(str, Enum):
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    IMPUTE = "impute"
    FILTER = "filter"
    AUGMENT = "augment"
    MERGE = "merge"
    SPLIT = "split"
    SELECT = "select"
    CUSTOM = "custom"


@dataclass
class TransformRecord:
    """Single transform operation in a lineage chain."""
    kind: TransformKind
    params: dict[str, Any]
    input_hash: str
    output_hash: str
    timestamp: float
    n_rows_in: int
    n_rows_out: int
    description: str = ""


@dataclass
class DatasetSnapshot:
    """Immutable snapshot of a dataset state."""
    content_hash: str
    n_samples: int
    n_features: int
    feature_names: list[str]
    created_at: float
    parent_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageGraph:
    """Full provenance graph for a dataset's history."""
    snapshots: dict[str, DatasetSnapshot] = field(default_factory=dict)
    transforms: list[TransformRecord] = field(default_factory=list)

    @property
    def root_hash(self) -> str | None:
        roots = [h for h, s in self.snapshots.items() if s.parent_hash is None]
        return roots[0] if roots else None

    @property
    def latest_hash(self) -> str | None:
        if not self.transforms:
            return self.root_hash
        return self.transforms[-1].output_hash

    def chain_length(self) -> int:
        return len(self.transforms)

    def get_ancestors(self, content_hash: str) -> list[str]:
        """Walk parent pointers back to root."""
        ancestors: list[str] = []
        current = content_hash
        while current in self.snapshots:
            parent = self.snapshots[current].parent_hash
            if parent is None:
                break
            ancestors.append(parent)
            current = parent
        return ancestors

    def transforms_for(self, content_hash: str) -> list[TransformRecord]:
        """All transforms that produced a given snapshot."""
        return [t for t in self.transforms if t.output_hash == content_hash]


def content_hash(X: np.ndarray, algorithm: str = "sha256") -> str:
    """Compute a deterministic content hash of a numpy array.

    The array is converted to C-contiguous bytes in a canonical dtype
    (float64) so the hash is reproducible across sessions.
    """
    arr = np.ascontiguousarray(X, dtype=np.float64)
    h = hashlib.new(algorithm)
    h.update(arr.tobytes())
    h.update(np.array(arr.shape, dtype=np.int64).tobytes())
    return h.hexdigest()


def metadata_hash(meta: dict[str, Any], algorithm: str = "sha256") -> str:
    """Deterministic hash of a JSON-serialisable metadata dict."""
    canonical = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.new(algorithm, canonical.encode()).hexdigest()


def create_snapshot(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    parent_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DatasetSnapshot:
    """Create an immutable snapshot of the current dataset state."""
    ch = content_hash(X)
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    return DatasetSnapshot(
        content_hash=ch,
        n_samples=n_samples,
        n_features=n_features,
        feature_names=feature_names,
        created_at=time.time(),
        parent_hash=parent_hash,
        metadata=metadata or {},
    )


def record_transform(
    kind: TransformKind,
    X_in: np.ndarray,
    X_out: np.ndarray,
    params: dict[str, Any] | None = None,
    description: str = "",
) -> TransformRecord:
    """Record a transform operation with input/output hashes."""
    return TransformRecord(
        kind=kind,
        params=params or {},
        input_hash=content_hash(X_in),
        output_hash=content_hash(X_out),
        timestamp=time.time(),
        n_rows_in=X_in.shape[0],
        n_rows_out=X_out.shape[0],
        description=description,
    )


class VersionedDataset:
    """Dataset wrapper with automatic lineage tracking."""

    def __init__(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._data = np.array(X, dtype=np.float64)
        snap = create_snapshot(self._data, feature_names, metadata=metadata)
        self.lineage = LineageGraph()
        self.lineage.snapshots[snap.content_hash] = snap
        self._current_hash = snap.content_hash

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def current_hash(self) -> str:
        return self._current_hash

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def apply(
        self,
        transform_fn,
        kind: TransformKind = TransformKind.CUSTOM,
        params: dict[str, Any] | None = None,
        description: str = "",
    ) -> VersionedDataset:
        """Apply a transform, record provenance, return new versioned dataset."""
        X_out = transform_fn(self._data)
        rec = record_transform(kind, self._data, X_out, params, description)
        self.lineage.transforms.append(rec)
        snap = create_snapshot(
            X_out,
            feature_names=self.lineage.snapshots[self._current_hash].feature_names,
            parent_hash=self._current_hash,
        )
        self.lineage.snapshots[snap.content_hash] = snap
        result = VersionedDataset.__new__(VersionedDataset)
        result._data = X_out
        result._current_hash = snap.content_hash
        result.lineage = self.lineage
        return result

    def diff_summary(self, other_hash: str) -> dict[str, Any]:
        """Summary of changes between current state and a prior snapshot."""
        if other_hash not in self.lineage.snapshots:
            return {"error": "snapshot not found"}
        other_snap = self.lineage.snapshots[other_hash]
        current_snap = self.lineage.snapshots[self._current_hash]
        transforms_between = [
            t for t in self.lineage.transforms
            if t.input_hash == other_hash or t.output_hash == self._current_hash
        ]
        return {
            "from_hash": other_hash[:12],
            "to_hash": self._current_hash[:12],
            "rows_delta": current_snap.n_samples - other_snap.n_samples,
            "features_delta": current_snap.n_features - other_snap.n_features,
            "n_transforms": len(transforms_between),
            "transform_kinds": [t.kind.value for t in transforms_between],
        }


def dataset_fingerprint(
    X: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Quick fingerprint for dataset identity checks.

    Returns content hash plus lightweight shape/distribution summary
    so two datasets can be compared without loading both fully.
    """
    ch = content_hash(X)
    col_means = np.nanmean(X, axis=0).tolist()
    col_stds = np.nanstd(X, axis=0).tolist()
    fp: dict[str, Any] = {
        "content_hash": ch,
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "col_means": col_means[:10],
        "col_stds": col_stds[:10],
        "nan_fraction": float(np.isnan(X).sum() / X.size) if X.size > 0 else 0.0,
    }
    if labels is not None:
        unique_labels = np.unique(labels)
        fp["n_classes"] = int(len(unique_labels))
        fp["label_hash"] = hashlib.sha256(
            np.ascontiguousarray(labels).tobytes()
        ).hexdigest()[:16]
    return fp
