"""Evaluation metrics: standard IR metrics for music retrieval systems.

Compute precision, recall, MAP, NDCG, MRR, and R-precision for ranked
retrieval lists. Includes genre-weighted variants and inter-annotator
agreement measures.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set


def precision_at_k(retrieved: List[str], relevant: Set[str],
                   k: int) -> float:
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / max(k, 1)


def recall_at_k(retrieved: List[str], relevant: Set[str],
                k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / len(relevant)


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r < 1e-10:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    ap = 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant)


def mean_average_precision(queries: List[List[str]],
                           relevants: List[Set[str]]) -> float:
    aps = [average_precision(q, r) for q, r in zip(queries, relevants)]
    return float(np.mean(aps)) if aps else 0.0


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(queries: List[List[str]],
                         relevants: List[Set[str]]) -> float:
    rrs = [reciprocal_rank(q, r) for q, r in zip(queries, relevants)]
    return float(np.mean(rrs)) if rrs else 0.0


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    scores = relevance_scores[:k]
    return sum(s / np.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    dcg = dcg_at_k(relevance_scores, k)
    ideal = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    return dcg / max(ideal, 1e-10)


def r_precision(retrieved: List[str], relevant: Set[str]) -> float:
    return precision_at_k(retrieved, relevant, len(relevant))


def genre_weighted_precision(retrieved: List[str],
                             genres: Dict[str, str],
                             target_genre: str, k: int) -> float:
    top_k = retrieved[:k]
    correct = sum(1 for r in top_k if genres.get(r) == target_genre)
    return correct / max(k, 1)


def inter_annotator_agreement(annotations_a: List[str],
                               annotations_b: List[str]) -> float:
    if len(annotations_a) != len(annotations_b):
        n = min(len(annotations_a), len(annotations_b))
        annotations_a = annotations_a[:n]
        annotations_b = annotations_b[:n]
    agree = sum(1 for a, b in zip(annotations_a, annotations_b) if a == b)
    return agree / max(len(annotations_a), 1)


def evaluation_report(retrieved: List[str], relevant: Set[str],
                      ks: List[int] = [5, 10, 20]) -> Dict[str, float]:
    report = {
        "ap": average_precision(retrieved, relevant),
        "rr": reciprocal_rank(retrieved, relevant),
        "r_precision": r_precision(retrieved, relevant),
    }
    for k in ks:
        report[f"p@{k}"] = precision_at_k(retrieved, relevant, k)
        report[f"r@{k}"] = recall_at_k(retrieved, relevant, k)
    return report
