"""Evaluation metrics for music models."""
from __future__ import annotations

import numpy as np


def genre_accuracy(predicted: np.ndarray, true: np.ndarray) -> float:
    if len(predicted) == 0:
        return 0.0
    return float(np.mean(predicted == true))


def mood_quadrant_accuracy(
    predicted_vad: np.ndarray,
    true_vad: np.ndarray,
) -> float:
    if len(predicted_vad) == 0:
        return 0.0
    pred_q = (predicted_vad[:, 0] >= 0.5).astype(int) * 2 + (predicted_vad[:, 1] >= 0.5).astype(int)
    true_q = (true_vad[:, 0] >= 0.5).astype(int) * 2 + (true_vad[:, 1] >= 0.5).astype(int)
    return float(np.mean(pred_q == true_q))


KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_detection_accuracy(
    predicted_keys: list[str],
    true_keys: list[str],
    allow_relative: bool = True,
) -> float:
    if not predicted_keys:
        return 0.0
    correct = 0
    for pred, true in zip(predicted_keys, true_keys):
        p = pred.strip().upper()
        t = true.strip().upper()
        if p == t:
            correct += 1
        elif allow_relative:
            p_base = p.split()[0] if " " in p else p
            t_base = t.split()[0] if " " in t else t
            if p_base in KEY_NAMES and t_base in KEY_NAMES:
                pi = KEY_NAMES.index(p_base)
                ti = KEY_NAMES.index(t_base)
                if (pi - ti) % 12 in (3, 9):
                    correct += 1
    return correct / len(predicted_keys)


def tempo_estimation_accuracy(
    predicted_bpm: np.ndarray,
    true_bpm: np.ndarray,
    tolerance: float = 0.04,
) -> float:
    if len(predicted_bpm) == 0:
        return 0.0
    ratios = predicted_bpm / np.maximum(true_bpm, 1e-8)
    correct = np.zeros(len(ratios), dtype=bool)
    for mult in [0.5, 1.0, 2.0, 3.0]:
        correct |= np.abs(ratios - mult) <= tolerance * mult
    return float(np.mean(correct))


def embedding_quality(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n = len(embeddings)
    if n < 3 or len(np.unique(labels)) < 2:
        return 0.0

    overall_mean = np.mean(embeddings, axis=0)
    unique_labels = np.unique(labels)

    a_scores = np.zeros(n)
    b_scores = np.full(n, np.inf)

    for lbl in unique_labels:
        mask = labels == lbl
        cluster = embeddings[mask]
        n_c = len(cluster)
        if n_c == 0:
            continue

        cluster_mean = np.mean(cluster, axis=0)

        for i in np.where(mask)[0]:
            if n_c > 1:
                dists = np.linalg.norm(cluster - embeddings[i], axis=1)
                a_scores[i] = np.sum(dists) / (n_c - 1)
            else:
                a_scores[i] = 0.0

        for other_lbl in unique_labels:
            if other_lbl == lbl:
                continue
            other_mask = labels == other_lbl
            other_cluster = embeddings[other_mask]
            for i in np.where(mask)[0]:
                mean_dist = np.mean(np.linalg.norm(other_cluster - embeddings[i], axis=1))
                b_scores[i] = min(b_scores[i], mean_dist)

    sil = np.zeros(n)
    for i in range(n):
        denom = max(a_scores[i], b_scores[i])
        if denom > 0:
            sil[i] = (b_scores[i] - a_scores[i]) / denom
    return float(np.mean(sil))


def recommendation_ndcg(
    predicted_ranking: list[list[str]],
    relevant_items: list[set[str]],
    k: int = 10,
) -> float:
    if not predicted_ranking:
        return 0.0

    ndcg_scores = []
    for pred, rel in zip(predicted_ranking, relevant_items):
        dcg = 0.0
        for i, item in enumerate(pred[:k]):
            if item in rel:
                dcg += 1.0 / np.log2(i + 2)

        ideal = sorted([1] * min(len(rel), k) + [0] * max(0, k - len(rel)), reverse=True)
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal) if g > 0)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores))
