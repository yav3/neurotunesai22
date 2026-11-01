"""Therapeutic assessment: outcome measures, session rating scales,
progress tracking, clinical goal evaluation for music therapy.

Pure numpy — no external clinical frameworks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


class GoalDomain(str, Enum):
    COGNITIVE = "cognitive"
    MOTOR = "motor"
    COMMUNICATION = "communication"
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    BEHAVIORAL = "behavioral"
    SENSORY = "sensory"


class ProgressLevel(IntEnum):
    NOT_ADDRESSED = 0
    MINIMAL = 1
    EMERGING = 2
    DEVELOPING = 3
    ACHIEVED = 4
    MAINTAINED = 5


@dataclass
class TherapeuticGoal:
    goal_id: str
    domain: GoalDomain
    description: str
    target_level: ProgressLevel = ProgressLevel.ACHIEVED
    baseline_level: ProgressLevel = ProgressLevel.NOT_ADDRESSED


@dataclass
class SessionObservation:
    session_number: int
    goal_id: str
    level: ProgressLevel
    engagement_rating: float = 0.0
    notes: str = ""


@dataclass
class OutcomeMeasure:
    goal_id: str
    domain: GoalDomain
    baseline: float
    current: float
    target: float
    effect_size: float
    progress_pct: float
    trend_slope: float
    sessions_to_target: Optional[int] = None


def engagement_score(attention_pct: float, participation_pct: float,
                     affect_rating: float, cooperation_rating: float,
                     weights: Tuple[float, ...] = (0.3, 0.3, 0.2, 0.2)
                     ) -> float:
    values = np.array([attention_pct, participation_pct,
                       affect_rating, cooperation_rating])
    values = np.clip(values, 0.0, 1.0)
    w = np.array(weights[:4])
    w = w / (w.sum() + 1e-12)
    return float(np.dot(values, w))


def session_quality_index(engagement: float, goal_progress: float,
                          therapeutic_relationship: float,
                          musical_interaction: float) -> float:
    scores = np.clip([engagement, goal_progress,
                      therapeutic_relationship, musical_interaction], 0.0, 1.0)
    return float(np.mean(scores))


def compute_effect_size(baseline_scores: np.ndarray,
                        post_scores: np.ndarray) -> float:
    if len(baseline_scores) < 2 or len(post_scores) < 2:
        diff = np.mean(post_scores) - np.mean(baseline_scores)
        return float(np.sign(diff) * min(abs(diff), 5.0))
    pooled_std = math.sqrt(
        ((len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1) +
         (len(post_scores) - 1) * np.var(post_scores, ddof=1)) /
        (len(baseline_scores) + len(post_scores) - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(post_scores) - np.mean(baseline_scores)) / pooled_std)


def interpret_effect_size(d: float) -> str:
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def progress_trajectory(observations: List[SessionObservation],
                        goal_id: str) -> Tuple[np.ndarray, np.ndarray]:
    filtered = [o for o in observations if o.goal_id == goal_id]
    filtered.sort(key=lambda o: o.session_number)
    sessions = np.array([o.session_number for o in filtered], dtype=np.float64)
    levels = np.array([float(o.level) for o in filtered], dtype=np.float64)
    return sessions, levels


def trend_analysis(sessions: np.ndarray, scores: np.ndarray
                   ) -> Tuple[float, float, float]:
    if len(sessions) < 2:
        return 0.0, float(scores[0]) if len(scores) > 0 else 0.0, 0.0
    x = sessions - sessions.mean()
    y = scores - scores.mean()
    slope = float(np.dot(x, y) / (np.dot(x, x) + 1e-12))
    intercept = float(scores.mean() - slope * sessions.mean())

    y_pred = slope * sessions + intercept
    ss_res = np.sum((scores - y_pred) ** 2)
    ss_tot = np.sum((scores - scores.mean()) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-12 else 0.0
    return slope, intercept, max(0.0, r_squared)


def sessions_to_target(slope: float, current_level: float,
                       target_level: float) -> Optional[int]:
    if slope <= 0 or current_level >= target_level:
        return None if slope <= 0 and current_level < target_level else 0
    remaining = target_level - current_level
    return max(1, int(math.ceil(remaining / slope)))


def compute_outcome(goal: TherapeuticGoal,
                    observations: List[SessionObservation]) -> OutcomeMeasure:
    sessions, levels = progress_trajectory(observations, goal.goal_id)
    if len(levels) == 0:
        return OutcomeMeasure(
            goal_id=goal.goal_id, domain=goal.domain,
            baseline=float(goal.baseline_level),
            current=float(goal.baseline_level),
            target=float(goal.target_level),
            effect_size=0.0, progress_pct=0.0, trend_slope=0.0,
        )

    baseline_val = float(goal.baseline_level)
    current_val = float(levels[-1])
    target_val = float(goal.target_level)

    n_baseline = max(1, len(levels) // 4)
    baseline_scores = levels[:n_baseline]
    post_scores = levels[-n_baseline:]
    es = compute_effect_size(baseline_scores, post_scores)

    range_val = target_val - baseline_val
    if abs(range_val) < 1e-12:
        progress = 1.0 if current_val >= target_val else 0.0
    else:
        progress = (current_val - baseline_val) / range_val

    slope, _, _ = trend_analysis(sessions, levels)
    est_sessions = sessions_to_target(slope, current_val, target_val)

    return OutcomeMeasure(
        goal_id=goal.goal_id, domain=goal.domain,
        baseline=baseline_val, current=current_val, target=target_val,
        effect_size=es, progress_pct=float(np.clip(progress, 0.0, 1.0)),
        trend_slope=slope, sessions_to_target=est_sessions,
    )


def aggregate_progress(outcomes: List[OutcomeMeasure],
                       domain: Optional[GoalDomain] = None) -> Dict[str, float]:
    filtered = outcomes
    if domain is not None:
        filtered = [o for o in outcomes if o.domain == domain]
    if not filtered:
        return {"mean_progress": 0.0, "mean_effect_size": 0.0,
                "goals_met_pct": 0.0, "mean_trend": 0.0}

    progresses = [o.progress_pct for o in filtered]
    effects = [o.effect_size for o in filtered]
    met = sum(1 for o in filtered if o.current >= o.target)
    trends = [o.trend_slope for o in filtered]

    return {
        "mean_progress": float(np.mean(progresses)),
        "mean_effect_size": float(np.mean(effects)),
        "goals_met_pct": met / len(filtered),
        "mean_trend": float(np.mean(trends)),
    }


def reliable_change_index(pre_score: float, post_score: float,
                          test_retest_r: float, sd: float) -> float:
    se = sd * math.sqrt(2.0 * (1.0 - test_retest_r))
    if se < 1e-12:
        return 0.0
    return (post_score - pre_score) / se


def clinically_significant(rci: float, threshold: float = 1.96) -> bool:
    return abs(rci) >= threshold


@dataclass
class SessionReport:
    session_number: int
    engagement: float
    quality_index: float
    goals_addressed: List[str]
    domain_progress: Dict[str, float]
    overall_progress: float
    recommendations: List[str]


def generate_session_report(session_num: int,
                            observations: List[SessionObservation],
                            goals: List[TherapeuticGoal],
                            all_observations: List[SessionObservation]
                            ) -> SessionReport:
    current_obs = [o for o in observations if o.session_number == session_num]
    engagements = [o.engagement_rating for o in current_obs if o.engagement_rating > 0]
    avg_engagement = float(np.mean(engagements)) if engagements else 0.0

    goals_addressed = list({o.goal_id for o in current_obs})

    outcomes = [compute_outcome(g, all_observations) for g in goals]
    domain_prog = {}
    for domain in GoalDomain:
        agg = aggregate_progress(outcomes, domain)
        if agg["mean_progress"] > 0:
            domain_prog[domain.value] = agg["mean_progress"]

    overall = aggregate_progress(outcomes)

    recommendations = []
    for o in outcomes:
        if o.trend_slope <= 0 and o.progress_pct < 0.5:
            recommendations.append(
                f"Goal {o.goal_id}: plateau detected, consider approach modification")
        if o.progress_pct >= 0.9:
            recommendations.append(
                f"Goal {o.goal_id}: near target, prepare maintenance phase")

    qi = session_quality_index(avg_engagement, overall["mean_progress"], 0.5, 0.5)

    return SessionReport(
        session_number=session_num,
        engagement=avg_engagement,
        quality_index=qi,
        goals_addressed=goals_addressed,
        domain_progress=domain_prog,
        overall_progress=overall["mean_progress"],
        recommendations=recommendations,
    )
