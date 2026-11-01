"""Quantitative outcome measurement for music therapy sessions."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class OutcomeDomain(str, Enum):
    MOOD = "mood"
    ANXIETY = "anxiety"
    PAIN = "pain"
    MOTOR = "motor"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    COMMUNICATION = "communication"


@dataclass
class AssessmentScore:
    domain: OutcomeDomain
    pre_score: float
    post_score: float
    scale_min: float = 0.0
    scale_max: float = 100.0
    higher_is_better: bool = True


@dataclass
class OutcomeResult:
    domain: OutcomeDomain
    raw_change: float
    percent_change: float
    effect_size: float
    reliable_change: bool
    clinically_significant: bool


@dataclass
class SessionTrajectory:
    session_ids: List[int]
    scores: np.ndarray
    slope: float
    intercept: float
    r_squared: float
    trend_label: str


def raw_change(assessment: AssessmentScore) -> float:
    return assessment.post_score - assessment.pre_score


def percent_change(assessment: AssessmentScore) -> float:
    denom = abs(assessment.pre_score) if assessment.pre_score != 0.0 else 1.0
    return (assessment.post_score - assessment.pre_score) / denom * 100.0


def cohens_d(pre_scores: np.ndarray, post_scores: np.ndarray) -> float:
    diff = post_scores - pre_scores
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 1.0
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff)) / sd


def pooled_cohens_d(
    group_pre: np.ndarray, group_post: np.ndarray,
    control_pre: np.ndarray, control_post: np.ndarray,
) -> float:
    d_treat = np.mean(group_post - group_pre)
    d_ctrl = np.mean(control_post - control_pre)
    n1, n2 = len(group_pre), len(control_pre)
    s1 = float(np.std(group_post - group_pre, ddof=1)) if n1 > 1 else 1.0
    s2 = float(np.std(control_post - control_pre, ddof=1)) if n2 > 1 else 1.0
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1))
    if sp < 1e-12:
        return 0.0
    return float(d_treat - d_ctrl) / sp


def reliable_change_index(
    pre: float, post: float, test_retest_r: float, sd_pre: float,
) -> Tuple[float, bool]:
    se = sd_pre * np.sqrt(1.0 - test_retest_r)
    s_diff = np.sqrt(2.0) * se
    if s_diff < 1e-12:
        return 0.0, False
    rci = (post - pre) / s_diff
    return float(rci), abs(rci) >= 1.96


def clinically_significant_change(
    post: float, norm_mean: float, norm_sd: float,
    clinical_mean: float, higher_is_better: bool = True,
) -> bool:
    cutoff_c = (norm_sd * clinical_mean + norm_sd * norm_mean) / (2.0 * norm_sd) \
        if norm_sd > 1e-12 else norm_mean
    if higher_is_better:
        return post >= cutoff_c
    return post <= cutoff_c


def compute_outcome(
    assessment: AssessmentScore,
    sd_pre: float = 15.0,
    test_retest_r: float = 0.80,
    norm_mean: float = 50.0,
    norm_sd: float = 10.0,
    clinical_mean: float = 30.0,
) -> OutcomeResult:
    rc = raw_change(assessment)
    pc = percent_change(assessment)
    pre_arr = np.array([assessment.pre_score])
    post_arr = np.array([assessment.post_score])
    es = cohens_d(pre_arr, post_arr)
    _, rci_sig = reliable_change_index(
        assessment.pre_score, assessment.post_score, test_retest_r, sd_pre,
    )
    csc = clinically_significant_change(
        assessment.post_score, norm_mean, norm_sd,
        clinical_mean, assessment.higher_is_better,
    )
    return OutcomeResult(
        domain=assessment.domain, raw_change=rc, percent_change=pc,
        effect_size=es, reliable_change=rci_sig,
        clinically_significant=csc,
    )


def session_trend(scores: np.ndarray) -> SessionTrajectory:
    n = len(scores)
    x = np.arange(n, dtype=float)
    if n < 2:
        return SessionTrajectory(
            session_ids=list(range(n)), scores=scores,
            slope=0.0, intercept=float(scores[0]) if n else 0.0,
            r_squared=0.0, trend_label="insufficient_data",
        )
    slope, intercept = np.polyfit(x, scores, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((scores - y_hat) ** 2))
    ss_tot = float(np.sum((scores - np.mean(scores)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    if abs(slope) < 0.5:
        label = "stable"
    elif slope > 0:
        label = "improving"
    else:
        label = "declining"
    return SessionTrajectory(
        session_ids=list(range(n)), scores=scores,
        slope=float(slope), intercept=float(intercept),
        r_squared=float(r2), trend_label=label,
    )


def outcome_trajectory_cluster(
    trajectories: List[np.ndarray], k: int = 3,
    max_iter: int = 50, seed: int = 42,
) -> np.ndarray:
    max_len = max(len(t) for t in trajectories)
    padded = np.zeros((len(trajectories), max_len))
    for i, t in enumerate(trajectories):
        padded[i, :len(t)] = t
        if len(t) < max_len:
            padded[i, len(t):] = t[-1]
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(padded), size=min(k, len(padded)), replace=False)
    centers = padded[indices].copy()
    labels = np.zeros(len(padded), dtype=int)
    for _ in range(max_iter):
        dists = np.array([
            np.sum((padded - c) ** 2, axis=1) for c in centers
        ])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = padded[mask].mean(axis=0)
    return labels


def responder_rate(outcomes: List[OutcomeResult]) -> float:
    if not outcomes:
        return 0.0
    n_respond = sum(1 for o in outcomes if o.reliable_change)
    return n_respond / len(outcomes)


def number_needed_to_treat(
    treatment_responder_rate: float,
    control_responder_rate: float,
) -> float:
    diff = treatment_responder_rate - control_responder_rate
    if abs(diff) < 1e-12:
        return float("inf")
    return 1.0 / diff


def minimal_detectable_change(sd_pre: float, test_retest_r: float) -> float:
    se = sd_pre * np.sqrt(1.0 - test_retest_r)
    return 1.96 * np.sqrt(2.0) * se


def session_dose_correlation(
    doses: np.ndarray, outcomes: np.ndarray,
) -> Tuple[float, float]:
    if len(doses) < 3:
        return 0.0, 1.0
    r = float(np.corrcoef(doses, outcomes)[0, 1])
    n = len(doses)
    t_stat = r * np.sqrt((n - 2) / max(1.0 - r**2, 1e-12))
    from scipy.stats import t as t_dist
    p_val = float(2.0 * t_dist.sf(abs(t_stat), df=n - 2))
    return r, p_val
