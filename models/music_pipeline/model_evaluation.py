"""Comprehensive model evaluation metrics for music ML pipelines:
classification, regression, ranking, calibration, and music-specific metrics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ClassificationReport:
    accuracy: float
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    class_names: List[str]


@dataclass
class RegressionReport:
    mae: float
    rmse: float
    r_squared: float
    median_ae: float
    max_error: float
    explained_variance: float


@dataclass
class RankingReport:
    mean_reciprocal_rank: float
    ndcg_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    map_score: float


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> ClassificationReport:
    """Full classification report with per-class and macro/weighted metrics."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n)
    acc = float(np.trace(cm)) / max(cm.sum(), 1)

    prec, rec, f1 = {}, {}, {}
    supports = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        prec[name] = p
        rec[name] = r
        f1[name] = _safe_div(2 * p * r, p + r)
        supports[name] = int(cm[i, :].sum())

    total = sum(supports.values())
    macro_f1 = sum(f1.values()) / max(n, 1)
    weighted_f1 = sum(f1[c] * supports[c] for c in class_names) / max(total, 1)

    return ClassificationReport(
        accuracy=acc,
        precision_per_class=prec,
        recall_per_class=rec,
        f1_per_class=f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=cm,
        class_names=class_names,
    )


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionReport:
    """Full regression report."""
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    median_ae = float(np.median(np.abs(errors)))
    max_err = float(np.max(np.abs(errors)))
    var_y = float(np.var(y_true))
    explained = 1.0 - float(np.var(errors)) / max(var_y, 1e-12)

    return RegressionReport(
        mae=mae, rmse=rmse, r_squared=r2,
        median_ae=median_ae, max_error=max_err, explained_variance=explained,
    )


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted cumulative gain at position k."""
    rel = relevances[:k]
    positions = np.arange(1, len(rel) + 1)
    return float(np.sum(rel / np.log2(positions + 1)))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Normalized DCG at position k."""
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    return _safe_div(actual, ideal)


def mean_reciprocal_rank(ranked_relevant: List[List[bool]]) -> float:
    """MRR over a set of queries. Each query is a list of bool (relevant or not) in rank order."""
    rrs = []
    for ranks in ranked_relevant:
        for i, rel in enumerate(ranks):
            if rel:
                rrs.append(1.0 / (i + 1))
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs)) if rrs else 0.0


def precision_at_k(relevant: np.ndarray, k: int) -> float:
    """Precision at position k (binary relevance)."""
    return float(np.mean(relevant[:k])) if k > 0 else 0.0


def average_precision(relevant: np.ndarray) -> float:
    """Average precision for a single query."""
    hits = 0
    ap = 0.0
    for i, r in enumerate(relevant):
        if r:
            hits += 1
            ap += hits / (i + 1)
    return _safe_div(ap, hits)


def ranking_report(
    queries_relevances: List[np.ndarray],
    ks: List[int] = (1, 5, 10),
) -> RankingReport:
    """Full ranking report across multiple queries."""
    mrr = mean_reciprocal_rank([list(r > 0) for r in queries_relevances])
    ndcgs = {}
    precs = {}
    for k in ks:
        ndcgs[k] = float(np.mean([ndcg_at_k(r, k) for r in queries_relevances]))
        precs[k] = float(np.mean([precision_at_k(r > 0, k) for r in queries_relevances]))
    map_val = float(np.mean([average_precision(r > 0) for r in queries_relevances]))

    return RankingReport(
        mean_reciprocal_rank=mrr,
        ndcg_at_k=ndcgs,
        precision_at_k=precs,
        map_score=map_val,
    )


def calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error (ECE) for probabilistic predictions."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (n_bin / total) * abs(avg_acc - avg_conf)
    return ece


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score for probabilistic binary classification."""
    return float(np.mean((y_prob - y_true) ** 2))


def onset_detection_f1(
    true_onsets: np.ndarray, pred_onsets: np.ndarray, tolerance_sec: float = 0.05
) -> Tuple[float, float, float]:
    """F1 score for onset detection with a time tolerance window.

    Returns (precision, recall, f1).
    """
    if len(pred_onsets) == 0 and len(true_onsets) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_onsets) == 0:
        return 0.0, 0.0, 0.0
    if len(true_onsets) == 0:
        return 0.0, 0.0, 0.0

    matched_true = set()
    tp = 0
    for p in pred_onsets:
        dists = np.abs(true_onsets - p)
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= tolerance_sec and best_idx not in matched_true:
            tp += 1
            matched_true.add(best_idx)

    prec = _safe_div(tp, len(pred_onsets))
    rec = _safe_div(tp, len(true_onsets))
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return prec, rec, f1


def pitch_accuracy(
    true_pitches: np.ndarray, pred_pitches: np.ndarray, tolerance_cents: float = 50.0
) -> float:
    """Raw pitch accuracy: fraction of frames within tolerance (cents)."""
    if len(true_pitches) == 0:
        return 1.0
    diff_cents = 1200 * np.log2((pred_pitches + 1e-6) / (true_pitches + 1e-6))
    return float(np.mean(np.abs(diff_cents) <= tolerance_cents))


def beat_tracking_f1(
    true_beats: np.ndarray, pred_beats: np.ndarray, tolerance_sec: float = 0.07
) -> Tuple[float, float, float]:
    """F1 for beat tracking evaluation."""
    return onset_detection_f1(true_beats, pred_beats, tolerance_sec)


def chord_recognition_accuracy(
    true_chords: List[str], pred_chords: List[str]
) -> float:
    """Frame-level chord recognition accuracy (exact match)."""
    if not true_chords:
        return 1.0
    correct = sum(1 for t, p in zip(true_chords, pred_chords) if t == p)
    return correct / len(true_chords)


def segment_boundary_f1(
    true_boundaries: np.ndarray,
    pred_boundaries: np.ndarray,
    tolerance_sec: float = 0.5,
) -> Tuple[float, float, float]:
    """F1 for structural segmentation boundary detection."""
    return onset_detection_f1(true_boundaries, pred_boundaries, tolerance_sec)


def source_separation_sdr(
    reference: np.ndarray, estimated: np.ndarray
) -> float:
    """Signal-to-distortion ratio (SDR) in dB for source separation evaluation."""
    noise = estimated - reference
    ref_energy = float(np.sum(reference ** 2))
    noise_energy = float(np.sum(noise ** 2))
    if noise_energy < 1e-12:
        return 100.0
    return float(10 * np.log10(ref_energy / noise_energy))
