"""Hash-chained evidence lineage — every solver run gets a tamper-evident record.

Uses the canonical chain_hash formula from neurotunes3/evidence_protocol/hashing.py:
json.dumps({prev_hash, record_id, problem_spec, answer_values, timestamp}, sort_keys=True)
→ SHA-256. This ensures cross-repo schema alignment checks pass.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class _NumpySafeEncoder(json.JSONEncoder):
    """Handles numpy types so chain_hash works with array-valued specs."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return np.round(o, 6).tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return round(float(o), 6)
        return super().default(o)


def chain_hash(
    prev_hash: str,
    record_id: str,
    problem_spec: dict,
    answer_values: dict,
    timestamp: float,
) -> str:
    """Canonical chain hash — matches neurotunes3/evidence_protocol/hashing.chain_hash.

    Same json.dumps({...}, sort_keys=True) → SHA-256 formula. Extended with
    a numpy-safe encoder so NeuroTunesAI callers can pass array-valued specs
    without pre-converting. For pure-dict inputs the output is byte-identical
    to the canonical function.
    """
    payload = json.dumps(
        {
            "prev_hash": prev_hash,
            "record_id": record_id,
            "problem_spec": problem_spec,
            "answer_values": answer_values,
            "timestamp": timestamp,
        },
        sort_keys=True,
        cls=_NumpySafeEncoder,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def content_hash(data: bytes) -> str:
    """Content-addressed hash — matches neurotunes3/evidence_protocol/hashing.content_hash."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class EvidenceRecord:
    """One immutable record in the evidence chain."""
    record_id: str
    problem_spec: dict[str, Any]
    answer_values: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    prev_hash: str = ""
    record_hash: str = ""

    def __post_init__(self):
        if not self.record_hash:
            self.record_hash = self.compute_hash()

    def compute_hash(self) -> str:
        return chain_hash(
            self.prev_hash,
            self.record_id,
            self.problem_spec,
            self.answer_values,
            self.timestamp,
        )

    def verify(self) -> bool:
        return self.record_hash == self.compute_hash()

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "problem_spec": self.problem_spec,
            "answer_values": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.answer_values.items()
            },
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }


class LineageChain:
    """Append-only hash chain of evidence records."""

    def __init__(self):
        self._records: list[EvidenceRecord] = []

    def append(
        self,
        record_id: str = "",
        problem_spec: dict | None = None,
        answer_values: dict | None = None,
        timestamp: float | None = None,
    ) -> EvidenceRecord:
        prev_hash = self._records[-1].record_hash if self._records else ""
        record = EvidenceRecord(
            record_id=record_id or f"rec-{len(self._records)}",
            problem_spec=problem_spec or {},
            answer_values=answer_values or {},
            timestamp=timestamp if timestamp is not None else time.time(),
            prev_hash=prev_hash,
        )
        self._records.append(record)
        return record

    def verify_chain(self) -> bool:
        for i, rec in enumerate(self._records):
            if not rec.verify():
                return False
            if i > 0 and rec.prev_hash != self._records[i - 1].record_hash:
                return False
        return True

    @property
    def head_hash(self) -> str:
        return self._records[-1].record_hash if self._records else ""

    def __len__(self) -> int:
        return len(self._records)

    def export(self) -> list[dict]:
        return [r.to_dict() for r in self._records]
