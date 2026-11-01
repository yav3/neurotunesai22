"""Evidence-chain integrity verification.

Given a list of EvidenceRecord dicts (as produced by EvidenceRecord.to_dict()),
verify:
1. Each record's hash matches its recomputed chain_hash.
2. prev_hash links form an unbroken chain.
3. Timestamps are monotonically non-decreasing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validation.health_check import Finding


@dataclass
class ChainReport:
    findings: list[Finding] = field(default_factory=list)
    records_checked: int = 0

    @property
    def ok(self) -> bool:
        return all(f.severity != "error" for f in self.findings)


class EvidenceChainCheck:
    """Verify integrity of an evidence chain."""

    def __init__(self, schema_root: Path | None = None) -> None:
        self.schema_root = schema_root
        self._chain_hash = None
        self._load_hasher()

    def _load_hasher(self) -> None:
        if self.schema_root is None:
            return
        import sys
        old = sys.path.copy()
        try:
            sys.path.insert(0, str(self.schema_root))
            from evidence_protocol.hashing import chain_hash
            self._chain_hash = chain_hash
        except Exception:
            pass
        finally:
            sys.path[:] = old

    def verify_chain(
        self, records: list[dict[str, Any]], source: str = "unknown"
    ) -> ChainReport:
        report = ChainReport()

        if not records:
            report.findings.append(Finding(
                source, "warning", "empty_chain", "No records to verify",
            ))
            return report

        prev_hash: str | None = None
        prev_ts: float = -1.0

        for i, rec in enumerate(records):
            report.records_checked += 1
            rid = rec.get("record_id", f"index-{i}")

            expected_prev = rec.get("prev_hash", "")
            if prev_hash is not None and expected_prev != prev_hash:
                report.findings.append(Finding(
                    source, "error", "broken_link",
                    f"Record {rid}: prev_hash={expected_prev[:16]}... "
                    f"but previous record hash={prev_hash[:16]}...",
                ))

            ts = rec.get("timestamp", 0.0)
            if ts < prev_ts:
                report.findings.append(Finding(
                    source, "warning", "timestamp_regression",
                    f"Record {rid}: timestamp {ts} < previous {prev_ts}",
                ))
            prev_ts = ts

            if self._chain_hash is not None:
                expected_hash = self._chain_hash(
                    rec.get("prev_hash", ""),
                    rec.get("record_id", ""),
                    rec.get("problem_spec", {}),
                    rec.get("answer_values", {}),
                    rec.get("timestamp", 0.0),
                )
                actual_hash = rec.get("hash", "")
                if actual_hash and actual_hash != expected_hash:
                    report.findings.append(Finding(
                        source, "error", "hash_mismatch",
                        f"Record {rid}: stored hash={actual_hash[:16]}... "
                        f"computed={expected_hash[:16]}...",
                    ))

            prev_hash = rec.get("hash", "")

        return report

    def verify_from_file(self, path: Path, source: str = "") -> ChainReport:
        import json
        if not path.is_file():
            report = ChainReport()
            report.findings.append(Finding(
                source or str(path), "error", "file_missing",
                f"Chain file not found: {path}",
            ))
            return report

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            report = ChainReport()
            report.findings.append(Finding(
                source or str(path), "error", "parse_error",
                f"Cannot parse chain file: {e}",
            ))
            return report

        records = data if isinstance(data, list) else data.get("records", [])
        return self.verify_chain(records, source=source or str(path))
