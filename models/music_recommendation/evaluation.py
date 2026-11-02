"""Recommendation evaluation: offline metrics, ranking quality, calibration, and fairness.

Extends the basic precision/recall/NDCG in collaborative_filtering.py with
full offline evaluation pipelines: leave-one-out, temporal split, stratified
fold evaluation, popularity debiasing (IPS), catalog coverage, Gini index
for fairness, calibration error, and statistical significance testing.

Pure numpy/scipy — no external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Sequence, Callable
from scipy.stats import mannwhitneyu, ttest_ind


@dataclass
class RankingMetrics:
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map_score: float
    mrr: float
    hit_rate: float
    k: int


@dataclass
class DiversityMetrics:
    intra_list_diversity: float
    coverage: float
    gini_index: float
    popularity_bias: float
    novelty: float
    serendipity: float


@dataclass
class CalibrationMetrics:
    genre_calibration_error: float
    mood_calibration_error: float
    overall_calibration: float


@dataclass
class EvaluationResult:
    ranking: RankingMetrics
    diversity: DiversityMetrics
    calibration: CalibrationMetrics
    n_users: int
    n_items: int


def dcg(relevances: np.ndarray) -> float:
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg(predicted_relevances: np.ndarray, ideal_relevances: np.ndarray) -> float:
    ideal = dcg(np.sort(ideal_relevances)[::-1])
    if ideal < 1e-12:
        return 0.0
    return dcg(predicted_relevances) / ideal


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision(recommended: List[int], relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant)


def reciprocal_rank(recommended: List[int], relevant: set) -> float:
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(recommended: List[int], relevant: set, k: int) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def compute_ranking_metrics(
    recommended: List[int], relevant: set, k: int,
) -> RankingMetrics:
    return RankingMetrics(
        precision_at_k=precision_at_k(recommended, relevant, k),
        recall_at_k=recall_at_k(recommended, relevant, k),
        ndcg_at_k=ndcg(
            np.array([1.0 if item in relevant else 0.0 for item in recommended[:k]]),
            np.ones(min(k, len(relevant))),
        ),
        map_score=average_precision(recommended, relevant),
        mrr=reciprocal_rank(recommended, relevant),
        hit_rate=hit_rate(recommended, relevant, k),
        k=k,
    )


def intra_list_distance(
    item_features: np.ndarray, recommended_indices: List[int],
) -> float:
    if len(recommended_indices) < 2:
        return 0.0
    feats = item_features[recommended_indices]
    n = len(feats)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            norm_i = np.linalg.norm(feats[i])
            norm_j = np.linalg.norm(feats[j])
            if norm_i > 1e-12 and norm_j > 1e-12:
                cos = np.dot(feats[i], feats[j]) / (norm_i * norm_j)
                total += 1.0 - float(np.clip(cos, -1, 1))
            else:
                total += 1.0
            pairs += 1
    return total / max(pairs, 1)


def catalog_coverage(
    all_recommendations: List[List[int]], n_items: int,
) -> float:
    recommended_set: set = set()
    for rec_list in all_recommendations:
        recommended_set.update(rec_list)
    return len(recommended_set) / max(n_items, 1)


def gini_index(item_counts: np.ndarray) -> float:
    """Gini coefficient: 0 = perfectly equal, 1 = maximally concentrated."""
    sorted_counts = np.sort(item_counts)
    n = len(sorted_counts)
    if n == 0 or sorted_counts.sum() < 1e-12:
        return 0.0
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_counts) - (n + 1) * sorted_counts.sum())
        / (n * sorted_counts.sum())
    )


def popularity_bias_score(
    recommended: List[int], item_popularity: np.ndarray,
) -> float:
    """Average log-popularity of recommended items (lower = less biased)."""
    if not recommended:
        return 0.0
    pops = item_popularity[recommended]
    return float(np.mean(np.log1p(pops)))


def novelty_score(
    recommended: List[int], item_popularity: np.ndarray, n_users: int,
) -> float:
    """Self-information novelty: -log2(popularity/N_users), averaged."""
    if not recommended or n_users == 0:
        return 0.0
    total = 0.0
    for item in recommended:
        p = item_popularity[item] / n_users
        if p > 0:
            total += -np.log2(p)
        else:
            total += 20.0
    return total / len(recommended)


def serendipity(
    recommended: List[int], relevant: set,
    baseline_recommended: List[int],
) -> float:
    """Fraction of relevant recommendations NOT in the baseline."""
    baseline_set = set(baseline_recommended)
    surprising_hits = sum(
        1 for item in recommended
        if item in relevant and item not in baseline_set
    )
    return surprising_hits / max(len(recommended), 1)


def compute_diversity_metrics(
    all_recommendations: List[List[int]],
    item_features: np.ndarray,
    item_popularity: np.ndarray,
    n_users: int,
    relevant_sets: Optional[List[set]] = None,
    baseline_recs: Optional[List[List[int]]] = None,
) -> DiversityMetrics:
    n_items = item_features.shape[0]
    ild_scores = [
        intra_list_distance(item_features, rec) for rec in all_recommendations
    ]
    cov = catalog_coverage(all_recommendations, n_items)
    rec_counts = np.zeros(n_items)
    for rec in all_recommendations:
        for item in rec:
            if 0 <= item < n_items:
                rec_counts[item] += 1
    gi = gini_index(rec_counts)
    pop_bias = float(np.mean([
        popularity_bias_score(rec, item_popularity) for rec in all_recommendations
    ]))
    nov = float(np.mean([
        novelty_score(rec, item_popularity, n_users) for rec in all_recommendations
    ]))
    ser = 0.0
    if relevant_sets and baseline_recs:
        ser_scores = [
            serendipity(rec, rel, base)
            for rec, rel, base in zip(all_recommendations, relevant_sets, baseline_recs)
        ]
        ser = float(np.mean(ser_scores))
    return DiversityMetrics(
        intra_list_diversity=float(np.mean(ild_scores)),
        coverage=cov,
        gini_index=gi,
        popularity_bias=pop_bias,
        novelty=nov,
        serendipity=ser,
    )


def calibration_error(
    user_profile: Dict[str, float], recommended_distribution: Dict[str, float],
) -> float:
    """KL-divergence-inspired calibration: how well recs match user taste distribution."""
    all_keys = set(user_profile) | set(recommended_distribution)
    error = 0.0
    for key in all_keys:
        p = user_profile.get(key, 0.0)
        q = recommended_distribution.get(key, 1e-6)
        if p > 0:
            error += p * np.log(p / max(q, 1e-12))
    return float(error)


def leave_one_out_evaluate(
    user_interactions: Dict[int, List[int]],
    recommend_fn: Callable[[int, List[int]], List[int]],
    k: int = 10,
) -> List[RankingMetrics]:
    """Leave-one-out evaluation: hold out last interaction per user."""
    results: List[RankingMetrics] = []
    for user_id, items in user_interactions.items():
        if len(items) < 2:
            continue
        train_items = items[:-1]
        test_item = {items[-1]}
        recommended = recommend_fn(user_id, train_items)
        metrics = compute_ranking_metrics(recommended, test_item, k)
        results.append(metrics)
    return results


def temporal_split_evaluate(
    user_interactions: Dict[int, List[Tuple[int, float]]],
    recommend_fn: Callable[[int, List[int]], List[int]],
    split_time: float,
    k: int = 10,
) -> List[RankingMetrics]:
    """Temporal split: items before split_time train, after test."""
    results: List[RankingMetrics] = []
    for user_id, interactions in user_interactions.items():
        train = [item for item, t in interactions if t < split_time]
        test = {item for item, t in interactions if t >= split_time}
        if not train or not test:
            continue
        recommended = recommend_fn(user_id, train)
        metrics = compute_ranking_metrics(recommended, test, k)
        results.append(metrics)
    return results


def aggregate_metrics(metrics_list: List[RankingMetrics]) -> RankingMetrics:
    if not metrics_list:
        return RankingMetrics(0, 0, 0, 0, 0, 0, 0)
    k = metrics_list[0].k
    return RankingMetrics(
        precision_at_k=float(np.mean([m.precision_at_k for m in metrics_list])),
        recall_at_k=float(np.mean([m.recall_at_k for m in metrics_list])),
        ndcg_at_k=float(np.mean([m.ndcg_at_k for m in metrics_list])),
        map_score=float(np.mean([m.map_score for m in metrics_list])),
        mrr=float(np.mean([m.mrr for m in metrics_list])),
        hit_rate=float(np.mean([m.hit_rate for m in metrics_list])),
        k=k,
    )


def ips_corrected_dcg(
    relevances: np.ndarray, propensities: np.ndarray,
) -> float:
    """Inverse Propensity Scoring correction for position bias in implicit feedback."""
    positions = np.arange(1, len(relevances) + 1)
    weights = relevances / np.clip(propensities, 0.01, 1.0)
    return float(np.sum(weights / np.log2(positions + 1)))


def statistical_significance(
    metric_a: List[float], metric_b: List[float],
    test: str = "mannwhitney",
) -> Tuple[float, float]:
    """Test whether two metric distributions differ significantly.

    Returns (statistic, p_value).
    """
    arr_a = np.array(metric_a)
    arr_b = np.array(metric_b)
    if test == "mannwhitney":
        stat, p = mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    else:
        stat, p = ttest_ind(arr_a, arr_b, equal_var=False)
    return float(stat), float(p)
