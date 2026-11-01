"""Evaluation harness — calibration + truth eval for the math kernel.

Implements the three-loop verification from Merlin's eval system:
  1. Citation harness: every claim has a source tag (form)
  2. Calibration harness: confidence matches accuracy (honesty)
  3. Truth eval: answers match human-labelled ground truth (substance)

This module provides the calibration and truth-eval loops adapted for
the Problem→Answer interface in models/interface/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CalibrationEntry:
    """One labelled prediction for calibration."""
    entry_id: str
    predicted_confidence: float
    label: str  # correct | partial | incorrect | unanswerable

    @property
    def empirical_value(self) -> float | None:
        return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(self.label)


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int = 0
    mean_predicted: float = 0.0
    empirical_accuracy: float = 0.0

    @property
    def gap(self) -> float:
        return abs(self.mean_predicted - self.empirical_accuracy)


BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def calibration_report(entries: list[CalibrationEntry]) -> dict[str, Any]:
    """Compute ECE, MCE, Brier score, and per-bin calibration."""
    scored = [e for e in entries if e.empirical_value is not None]
    if not scored:
        return {"n_total": len(entries), "n_scored": 0, "bins": []}

    bins: list[CalibrationBin] = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        in_bin = [e for e in scored if lo <= e.predicted_confidence < hi]
        if not in_bin:
            bins.append(CalibrationBin(lo=lo, hi=hi))
            continue
        mean_pred = np.mean([e.predicted_confidence for e in in_bin])
        emp_acc = np.mean([e.empirical_value for e in in_bin])
        bins.append(CalibrationBin(
            lo=lo, hi=hi, n=len(in_bin),
            mean_predicted=float(mean_pred),
            empirical_accuracy=float(emp_acc),
        ))

    n_total = len(scored)
    ece = sum(b.n / n_total * b.gap for b in bins if b.n > 0)
    nonempty = [b for b in bins if b.n >= 3]
    mce = max((b.gap for b in nonempty), default=0.0)

    preds = np.array([e.predicted_confidence for e in scored])
    labels = np.array([e.empirical_value for e in scored])
    brier = float(np.mean((preds - labels) ** 2))
    eps = 1e-15
    log_loss = float(-np.mean(
        labels * np.log(np.clip(preds, eps, 1 - eps))
        + (1 - labels) * np.log(np.clip(1 - preds, eps, 1 - eps))
    ))

    return {
        "n_total": len(entries),
        "n_scored": len(scored),
        "n_excluded": len(entries) - len(scored),
        "expected_calibration_error": round(ece, 6),
        "maximum_calibration_error": round(mce, 6),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "bins": [
            {"lo": b.lo, "hi": b.hi, "n": b.n,
             "mean_predicted": round(b.mean_predicted, 4),
             "empirical_accuracy": round(b.empirical_accuracy, 4),
             "gap": round(b.gap, 4)}
            for b in bins
        ],
    }


@dataclass
class TruthEntry:
    """One labelled question for truth evaluation."""
    entry_id: str
    query: dict[str, Any]
    expected_claims: list[dict[str, str]]
    forbidden_substrings: list[str] = field(default_factory=list)
    min_confidence: float = 0.0


@dataclass
class TruthResult:
    entry_id: str
    expected_total: int
    expected_matched: int
    recall: float
    produced_total: int
    forbidden_hits: list[str]
    confidence_ok: bool
    status: str  # PASS | WARN | FAIL


def truth_eval(
    entries: list[TruthEntry],
    produced: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate produced answers against labelled ground truth.

    Args:
        entries: labelled gold set
        produced: {entry_id: {"claims": [...], "confidence": float, "text": str}}
    """
    results = []
    for e in entries:
        p = produced.get(e.entry_id, {"claims": [], "confidence": 0.0, "text": ""})
        claims = p.get("claims", [])
        text = p.get("text", "")
        conf = p.get("confidence", 0.0)

        matched = 0
        for exp in e.expected_claims:
            pred_str = exp.get("predicate", "")
            subj_str = exp.get("subject", "").lower()
            obj_str = exp.get("object", "").lower()
            for c in claims:
                c_pred = c.get("predicate", "")
                c_subj = c.get("subject", "").lower()
                c_obj = c.get("object", "").lower()
                if c_pred == pred_str:
                    if (not subj_str or subj_str in c_subj) and (not obj_str or obj_str in c_obj):
                        matched += 1
                        break

        total = len(e.expected_claims)
        recall = matched / total if total > 0 else 1.0
        forbidden = [s for s in e.forbidden_substrings if s.lower() in text.lower()]
        conf_ok = conf >= e.min_confidence

        if recall >= 0.9 and not forbidden and conf_ok:
            status = "PASS"
        elif recall >= 0.5 and not forbidden:
            status = "WARN"
        else:
            status = "FAIL"

        results.append(TruthResult(
            entry_id=e.entry_id,
            expected_total=total,
            expected_matched=matched,
            recall=round(recall, 4),
            produced_total=len(claims),
            forbidden_hits=forbidden,
            confidence_ok=conf_ok,
            status=status,
        ))

    n = len(results)
    return {
        "n": n,
        "pass": sum(1 for r in results if r.status == "PASS"),
        "warn": sum(1 for r in results if r.status == "WARN"),
        "fail": sum(1 for r in results if r.status == "FAIL"),
        "macro_recall": round(np.mean([r.recall for r in results]), 4) if results else 0.0,
        "any_forbidden_hit": any(r.forbidden_hits for r in results),
        "results": [
            {"id": r.entry_id, "status": r.status, "recall": r.recall,
             "expected": r.expected_total, "matched": r.expected_matched,
             "forbidden": r.forbidden_hits}
            for r in results
        ],
    }
