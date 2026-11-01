"""Study registry — pre-registered experiments with declared criteria.

Mirrors merlinaineuralplus2/backend/records/studies/ pattern:
each study has an ID, pre-registered criterion, declared data source,
and a verdict (met/not met/pending) that cannot be changed after the
first run without generating a new record.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StudyRecord:
    """A single pre-registered study record."""
    study_id: str
    title: str
    criterion: str
    data_kind: str = "synthetic"
    synthetic: bool = True
    run_date: str = ""
    data: str = ""
    measured: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    verdict: str = "pending"
    verdict_note: str = ""
    provenance: dict[str, list[str]] = field(default_factory=lambda: {"records": [], "scripts": []})

    def to_dict(self) -> dict:
        return {
            "study": self.study_id,
            "synthetic": self.synthetic,
            "data_kind": self.data_kind,
            "title": self.title,
            "criterion": self.criterion,
            "run_date": self.run_date,
            "data": self.data,
            "measured": self.measured,
            "summary": self.summary,
            "verdict": self.verdict,
            "verdict_note": self.verdict_note,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StudyRecord:
        return cls(
            study_id=d["study"],
            title=d["title"],
            criterion=d["criterion"],
            data_kind=d.get("data_kind", "synthetic"),
            synthetic=d.get("synthetic", True),
            run_date=d.get("run_date", ""),
            data=d.get("data", ""),
            measured=d.get("measured", {}),
            summary=d.get("summary", ""),
            verdict=d.get("verdict", "pending"),
            verdict_note=d.get("verdict_note", ""),
            provenance=d.get("provenance", {"records": [], "scripts": []}),
        )


class StudyRegistry:
    """Registry of pre-registered studies.

    Enforces: declare criterion before running, record verdict immutably.
    """

    def __init__(self, records_dir: Path | None = None):
        self._studies: dict[str, StudyRecord] = {}
        self._records_dir = records_dir
        if records_dir and records_dir.exists():
            self._load_from_dir(records_dir)

    def _load_from_dir(self, path: Path):
        for f in path.glob("*.json"):
            if f.name == "INDEX.json":
                continue
            with open(f) as fh:
                data = json.load(fh)
            if "study" in data:
                rec = StudyRecord.from_dict(data)
                self._studies[rec.study_id] = rec

    def register(
        self,
        study_id: str,
        title: str,
        criterion: str,
        data_kind: str = "synthetic",
        synthetic: bool = True,
    ) -> StudyRecord:
        if study_id in self._studies:
            raise ValueError(f"Study {study_id} already registered")
        rec = StudyRecord(
            study_id=study_id,
            title=title,
            criterion=criterion,
            data_kind=data_kind,
            synthetic=synthetic,
        )
        self._studies[study_id] = rec
        return rec

    def record_verdict(
        self,
        study_id: str,
        measured: dict,
        summary: str,
        verdict: str,
        verdict_note: str = "",
    ) -> StudyRecord:
        if study_id not in self._studies:
            raise KeyError(f"Study {study_id} not registered — register before running")
        rec = self._studies[study_id]
        if rec.verdict != "pending":
            raise ValueError(
                f"Study {study_id} verdict already recorded as '{rec.verdict}'. "
                "Create a new study ID for re-runs."
            )
        rec.measured = measured
        rec.summary = summary
        rec.verdict = verdict
        rec.verdict_note = verdict_note
        rec.run_date = time.strftime("%Y-%m-%d")
        return rec

    def get(self, study_id: str) -> StudyRecord | None:
        return self._studies.get(study_id)

    def list_studies(self) -> list[str]:
        return list(self._studies.keys())

    def pending(self) -> list[str]:
        return [k for k, v in self._studies.items() if v.verdict == "pending"]

    def export_all(self) -> dict[str, dict]:
        return {k: v.to_dict() for k, v in self._studies.items()}


EvidenceRegistry = StudyRegistry
