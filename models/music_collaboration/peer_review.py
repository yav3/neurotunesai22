"""Peer review and feedback system for collaborative music composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ReviewCriterion", "ReviewScore", "PeerReview", "ReviewSummary",
    "create_review", "aggregate_reviews", "reviewer_reliability",
    "inter_rater_agreement", "weighted_consensus",
    "feedback_quality_score", "review_bias_detection",
    "bayesian_score_estimate", "rank_submissions",
    "calibration_factor", "detect_outlier_reviews",
]


@dataclass
class ReviewCriterion:
    name: str
    weight: float = 1.0
    min_score: float = 0.0
    max_score: float = 10.0


@dataclass
class ReviewScore:
    criterion: str
    score: float
    confidence: float = 1.0
    comment: str = ""


@dataclass
class PeerReview:
    reviewer_id: str
    submission_id: str
    scores: List[ReviewScore] = field(default_factory=list)
    overall: float = 0.0
    timestamp: float = 0.0


@dataclass
class ReviewSummary:
    submission_id: str
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)
    weighted_overall: float = 0.0
    n_reviews: int = 0
    agreement: float = 0.0


DEFAULT_CRITERIA = [
    ReviewCriterion("melody", 1.5),
    ReviewCriterion("harmony", 1.2),
    ReviewCriterion("rhythm", 1.0),
    ReviewCriterion("originality", 1.3),
    ReviewCriterion("production", 0.8),
    ReviewCriterion("emotional_impact", 1.2),
]


def create_review(
    reviewer_id: str, submission_id: str,
    scores: Dict[str, float], confidences: Optional[Dict[str, float]] = None,
    criteria: Optional[List[ReviewCriterion]] = None,
) -> PeerReview:
    crit_list = criteria or DEFAULT_CRITERIA
    crit_map = {c.name: c for c in crit_list}
    review_scores = []
    weighted_sum = 0.0
    weight_total = 0.0
    for name, score in scores.items():
        conf = (confidences or {}).get(name, 1.0)
        crit = crit_map.get(name)
        if crit:
            clamped = max(crit.min_score, min(crit.max_score, score))
            review_scores.append(ReviewScore(name, clamped, conf))
            weighted_sum += clamped * crit.weight
            weight_total += crit.weight
        else:
            review_scores.append(ReviewScore(name, score, conf))
            weighted_sum += score
            weight_total += 1.0
    overall = weighted_sum / weight_total if weight_total > 0 else 0.0
    return PeerReview(reviewer_id, submission_id, review_scores, overall)


def aggregate_reviews(
    reviews: List[PeerReview],
    criteria: Optional[List[ReviewCriterion]] = None,
) -> ReviewSummary:
    if not reviews:
        return ReviewSummary("", n_reviews=0)
    sub_id = reviews[0].submission_id
    crit_list = criteria or DEFAULT_CRITERIA
    crit_map = {c.name: c for c in crit_list}
    all_scores: Dict[str, List[float]] = {}
    for r in reviews:
        for s in r.scores:
            all_scores.setdefault(s.criterion, []).append(s.score)
    means = {k: float(np.mean(v)) for k, v in all_scores.items()}
    stds = {k: float(np.std(v)) for k, v in all_scores.items()}
    weighted_sum = 0.0
    weight_total = 0.0
    for name, mean in means.items():
        w = crit_map[name].weight if name in crit_map else 1.0
        weighted_sum += mean * w
        weight_total += w
    overall = weighted_sum / weight_total if weight_total > 0 else 0.0
    overalls = [r.overall for r in reviews]
    agreement = 1.0 - float(np.std(overalls) / max(np.mean(overalls), 1e-9))
    return ReviewSummary(
        sub_id, means, stds, overall, len(reviews), max(0.0, agreement),
    )


def reviewer_reliability(
    reviewer_id: str, all_reviews: List[PeerReview],
) -> float:
    reviewer_reviews = [r for r in all_reviews if r.reviewer_id == reviewer_id]
    if len(reviewer_reviews) < 2:
        return 0.5
    by_sub: Dict[str, List[PeerReview]] = {}
    for r in all_reviews:
        by_sub.setdefault(r.submission_id, []).append(r)
    deviations = []
    for r in reviewer_reviews:
        peers = [p for p in by_sub.get(r.submission_id, []) if p.reviewer_id != reviewer_id]
        if peers:
            peer_mean = np.mean([p.overall for p in peers])
            deviations.append(abs(r.overall - peer_mean))
    if not deviations:
        return 0.5
    mean_dev = np.mean(deviations)
    return float(np.exp(-mean_dev / 2.0))


def inter_rater_agreement(reviews_by_item: Dict[str, List[float]]) -> float:
    items = list(reviews_by_item.values())
    if len(items) < 2:
        return 1.0
    all_ratings = np.array([r for item in items for r in item])
    grand_mean = all_ratings.mean()
    ss_between = sum(len(r) * (np.mean(r) - grand_mean) ** 2 for r in items)
    ss_within = sum(np.sum((np.array(r) - np.mean(r)) ** 2) for r in items)
    k = len(items)
    n_total = len(all_ratings)
    if k <= 1:
        return 1.0
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k) if n_total > k else 1e-9
    icc = (ms_between - ms_within) / (ms_between + ms_within) if (ms_between + ms_within) > 1e-9 else 0.0
    return float(max(0.0, min(1.0, icc)))


def weighted_consensus(
    reviews: List[PeerReview], reliability_scores: Dict[str, float],
) -> float:
    if not reviews:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for r in reviews:
        w = reliability_scores.get(r.reviewer_id, 0.5)
        weighted_sum += r.overall * w
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def feedback_quality_score(review: PeerReview) -> float:
    if not review.scores:
        return 0.0
    n_criteria = len(review.scores)
    completeness = min(1.0, n_criteria / 6.0)
    has_comments = sum(1 for s in review.scores if s.comment) / max(n_criteria, 1)
    score_variance = float(np.std([s.score for s in review.scores]))
    discrimination = min(1.0, score_variance / 2.0)
    return 0.4 * completeness + 0.3 * has_comments + 0.3 * discrimination


def review_bias_detection(
    reviewer_id: str, all_reviews: List[PeerReview],
) -> Dict[str, float]:
    reviewer_reviews = [r for r in all_reviews if r.reviewer_id == reviewer_id]
    other_reviews = [r for r in all_reviews if r.reviewer_id != reviewer_id]
    if not reviewer_reviews or not other_reviews:
        return {"leniency": 0.0, "central_tendency": 0.0, "halo": 0.0}
    rev_mean = np.mean([r.overall for r in reviewer_reviews])
    other_mean = np.mean([r.overall for r in other_reviews])
    leniency = float(rev_mean - other_mean)
    score_stds = [float(np.std([s.score for s in r.scores])) for r in reviewer_reviews if r.scores]
    central_tendency = -float(np.mean(score_stds)) if score_stds else 0.0
    corrs = []
    for r in reviewer_reviews:
        if len(r.scores) >= 2:
            vals = [s.score for s in r.scores]
            corrs.append(float(np.std(vals)))
    halo = 1.0 - float(np.mean(corrs) / 3.0) if corrs else 0.0
    return {"leniency": leniency, "central_tendency": central_tendency, "halo": max(0.0, halo)}


def bayesian_score_estimate(
    scores: List[float], prior_mean: float = 5.0, prior_strength: float = 2.0,
) -> float:
    if not scores:
        return prior_mean
    n = len(scores)
    sample_mean = np.mean(scores)
    return float((prior_strength * prior_mean + n * sample_mean) / (prior_strength + n))


def rank_submissions(summaries: List[ReviewSummary]) -> List[Tuple[str, float, int]]:
    ranked = sorted(summaries, key=lambda s: s.weighted_overall, reverse=True)
    return [(s.submission_id, s.weighted_overall, i + 1) for i, s in enumerate(ranked)]


def calibration_factor(
    reviewer_id: str, all_reviews: List[PeerReview],
    ground_truth: Optional[Dict[str, float]] = None,
) -> float:
    if ground_truth is None:
        return 1.0
    reviewer_reviews = [r for r in all_reviews if r.reviewer_id == reviewer_id]
    diffs = []
    for r in reviewer_reviews:
        if r.submission_id in ground_truth:
            diffs.append(ground_truth[r.submission_id] - r.overall)
    if not diffs:
        return 1.0
    mean_diff = np.mean(diffs)
    return float(1.0 + mean_diff / 10.0)


def detect_outlier_reviews(
    reviews: List[PeerReview], threshold: float = 2.0,
) -> List[str]:
    if len(reviews) < 3:
        return []
    overalls = np.array([r.overall for r in reviews])
    mean = overalls.mean()
    std = overalls.std()
    if std < 1e-9:
        return []
    outliers = []
    for r, o in zip(reviews, overalls):
        if abs(o - mean) > threshold * std:
            outliers.append(r.reviewer_id)
    return outliers
