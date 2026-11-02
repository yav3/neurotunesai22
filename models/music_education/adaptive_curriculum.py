"""Adaptive music curriculum: IRT parameter estimation, SPRT mastery, learning curves, scheduling.

Extends the basic IRT/Elo framework in adaptive_difficulty.py with:
- Music-specific skill taxonomy with prerequisite graph
- Maximum-likelihood 2PL parameter estimation from response data
- Sequential probability ratio test (SPRT) for mastery decisions
- Power-law learning curve fitting (Newell & Rosenbloom 1981)
- Enhanced SM-2 scheduling with music-domain stability priors
- Comprehensive progress analytics (accuracy trajectory, time-on-task, skill gaps)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Music-specific skill taxonomy
# ---------------------------------------------------------------------------

class MusicSkill(Enum):
    RHYTHM_READING = "rhythm_reading"
    RHYTHM_PERFORMANCE = "rhythm_performance"
    INTERVAL_RECOGNITION = "interval_recognition"
    INTERVAL_SINGING = "interval_singing"
    CHORD_IDENTIFICATION = "chord_identification"
    CHORD_VOICING = "chord_voicing"
    SIGHT_SINGING = "sight_singing"
    SIGHT_READING_TREBLE = "sight_reading_treble"
    SIGHT_READING_BASS = "sight_reading_bass"
    KEY_IDENTIFICATION = "key_identification"
    SCALE_CONSTRUCTION = "scale_construction"
    MELODIC_DICTATION = "melodic_dictation"
    HARMONIC_DICTATION = "harmonic_dictation"
    FORM_ANALYSIS = "form_analysis"
    TEMPO_DISCRIMINATION = "tempo_discrimination"
    DYNAMICS_EXPRESSION = "dynamics_expression"
    TIMBRE_RECOGNITION = "timbre_recognition"


SKILL_PREREQUISITES: Dict[MusicSkill, List[MusicSkill]] = {
    MusicSkill.RHYTHM_PERFORMANCE: [MusicSkill.RHYTHM_READING],
    MusicSkill.INTERVAL_SINGING: [MusicSkill.INTERVAL_RECOGNITION],
    MusicSkill.CHORD_VOICING: [MusicSkill.CHORD_IDENTIFICATION],
    MusicSkill.SIGHT_SINGING: [MusicSkill.INTERVAL_SINGING, MusicSkill.RHYTHM_READING],
    MusicSkill.SIGHT_READING_BASS: [MusicSkill.SIGHT_READING_TREBLE],
    MusicSkill.MELODIC_DICTATION: [MusicSkill.INTERVAL_RECOGNITION, MusicSkill.RHYTHM_READING],
    MusicSkill.HARMONIC_DICTATION: [MusicSkill.CHORD_IDENTIFICATION, MusicSkill.MELODIC_DICTATION],
    MusicSkill.FORM_ANALYSIS: [MusicSkill.KEY_IDENTIFICATION],
    MusicSkill.SCALE_CONSTRUCTION: [MusicSkill.KEY_IDENTIFICATION],
}

SKILL_LEVELS = {
    "novice": (-2.0, -0.5),
    "beginner": (-0.5, 0.5),
    "intermediate": (0.5, 1.5),
    "advanced": (1.5, 2.5),
    "expert": (2.5, 4.0),
}


def skill_level_label(ability: float) -> str:
    for label, (lo, hi) in SKILL_LEVELS.items():
        if lo <= ability < hi:
            return label
    return "expert" if ability >= 4.0 else "novice"


def prerequisites_satisfied(skill: MusicSkill,
                            abilities: Dict[MusicSkill, float],
                            threshold: float = 0.0) -> bool:
    prereqs = SKILL_PREREQUISITES.get(skill, [])
    return all(abilities.get(p, -3.0) >= threshold for p in prereqs)


def recommended_skills(abilities: Dict[MusicSkill, float],
                       threshold: float = 0.0) -> List[MusicSkill]:
    """Skills whose prerequisites are met but that haven't reached threshold."""
    ready = []
    for skill in MusicSkill:
        current = abilities.get(skill, -3.0)
        if current < threshold and prerequisites_satisfied(skill, abilities, threshold):
            ready.append(skill)
    return ready


# ---------------------------------------------------------------------------
# 2. IRT 2PL maximum-likelihood parameter estimation
# ---------------------------------------------------------------------------

@dataclass
class ResponseRecord:
    item_id: str
    ability_est: float
    correct: bool


def _irt_2pl_prob(theta: float, a: float, b: float) -> float:
    z = a * (theta - b)
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def estimate_difficulty_mle(responses: List[ResponseRecord],
                            discrimination: float = 1.0,
                            lr: float = 0.5, tol: float = 1e-4,
                            max_iter: int = 100) -> float:
    """MLE estimate of item difficulty from learner responses (Newton-Raphson)."""
    b = 0.0
    for _ in range(max_iter):
        grad = 0.0
        hess = 0.0
        for r in responses:
            p = _irt_2pl_prob(r.ability_est, discrimination, b)
            q = 1.0 - p
            u = 1.0 if r.correct else 0.0
            grad += -discrimination * (u - p)
            hess += discrimination ** 2 * p * q
        if abs(hess) < 1e-15:
            break
        step = grad / (hess + 1e-10)
        b_new = b - lr * step
        if abs(b_new - b) < tol:
            b = b_new
            break
        b = b_new
    return b


def estimate_discrimination_mle(responses: List[ResponseRecord],
                                difficulty: float,
                                lr: float = 0.3, tol: float = 1e-4,
                                max_iter: int = 100) -> float:
    """MLE estimate of item discrimination (clamped to [0.2, 5.0])."""
    a = 1.0
    for _ in range(max_iter):
        grad = 0.0
        hess = 0.0
        for r in responses:
            p = _irt_2pl_prob(r.ability_est, a, difficulty)
            q = 1.0 - p
            u = 1.0 if r.correct else 0.0
            d = r.ability_est - difficulty
            grad += d * (u - p)
            hess += -(d ** 2) * p * q
        if abs(hess) < 1e-15:
            break
        step = grad / (hess - 1e-10)
        a_new = a - lr * step
        a_new = max(0.2, min(5.0, a_new))
        if abs(a_new - a) < tol:
            a = a_new
            break
        a = a_new
    return a


def estimate_2pl_parameters(responses: List[ResponseRecord],
                            max_outer: int = 20,
                            tol: float = 1e-3) -> Tuple[float, float]:
    """Joint MLE of (discrimination, difficulty) via alternating optimisation."""
    a, b = 1.0, 0.0
    for _ in range(max_outer):
        b_new = estimate_difficulty_mle(responses, discrimination=a)
        a_new = estimate_discrimination_mle(responses, difficulty=b_new)
        if abs(a_new - a) + abs(b_new - b) < tol:
            return a_new, b_new
        a, b = a_new, b_new
    return a, b


# ---------------------------------------------------------------------------
# 3. Sequential Probability Ratio Test (SPRT) for mastery detection
# ---------------------------------------------------------------------------

@dataclass
class SPRTState:
    """Wald's SPRT for classifying mastery vs. non-mastery."""
    mastery_prob: float = 0.85
    non_mastery_prob: float = 0.55
    alpha: float = 0.05   # false-mastery rate
    beta: float = 0.10    # false-non-mastery rate
    log_lr: float = 0.0
    n_obs: int = 0

    @property
    def upper_boundary(self) -> float:
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower_boundary(self) -> float:
        return math.log(self.beta / (1 - self.alpha))

    def update(self, correct: bool) -> str:
        """Update log-likelihood ratio. Returns 'mastery', 'non_mastery', or 'continue'."""
        p1 = self.mastery_prob
        p0 = self.non_mastery_prob
        if correct:
            self.log_lr += math.log(p1 / max(p0, 1e-15))
        else:
            self.log_lr += math.log((1 - p1) / max(1 - p0, 1e-15))
        self.n_obs += 1

        if self.log_lr >= self.upper_boundary:
            return "mastery"
        elif self.log_lr <= self.lower_boundary:
            return "non_mastery"
        return "continue"

    @property
    def mastery_probability_estimate(self) -> float:
        """Posterior odds → probability approximation."""
        odds = math.exp(min(self.log_lr, 20.0))
        return odds / (1 + odds)


def run_sprt(responses: List[bool],
             mastery_prob: float = 0.85,
             non_mastery_prob: float = 0.55,
             alpha: float = 0.05,
             beta: float = 0.10) -> Tuple[str, int]:
    """Run SPRT on a response sequence. Returns (decision, n_items_used)."""
    state = SPRTState(mastery_prob, non_mastery_prob, alpha, beta)
    for i, correct in enumerate(responses):
        decision = state.update(correct)
        if decision != "continue":
            return decision, i + 1
    return "continue", len(responses)


# ---------------------------------------------------------------------------
# 4. Power-law learning curve
# ---------------------------------------------------------------------------

def power_law_predict(trial: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Power law of practice: T(n) = a * n^(-b) + c.

    a: initial performance penalty
    b: learning rate exponent (typically 0.2–0.6)
    c: asymptotic performance floor
    """
    return a * np.power(np.maximum(trial, 1), -b) + c


def fit_power_law(trials: np.ndarray, times: np.ndarray,
                  max_iter: int = 200, lr: float = 0.01) -> Tuple[float, float, float]:
    """Fit power-law learning curve via gradient descent on MSE.

    Returns (a, b, c) parameters.
    """
    a = float(times[0] - times[-1]) if len(times) > 1 else 1.0
    b = 0.3
    c = float(np.min(times))
    a = max(a, 0.01)

    for _ in range(max_iter):
        pred = power_law_predict(trials, a, b, c)
        residual = pred - times
        mse_grad_a = 2.0 * np.mean(residual * np.power(np.maximum(trials, 1), -b))
        log_n = np.log(np.maximum(trials, 1))
        mse_grad_b = 2.0 * np.mean(residual * (-a * log_n * np.power(np.maximum(trials, 1), -b)))
        mse_grad_c = 2.0 * np.mean(residual)

        a -= lr * mse_grad_a
        b -= lr * mse_grad_b
        c -= lr * mse_grad_c

        a = max(0.01, a)
        b = max(0.05, min(2.0, b))
        c = max(0.0, c)

    return a, b, c


def predict_mastery_trial(a: float, b: float, c: float,
                          target_time: float) -> int:
    """Estimate trial number at which performance reaches target_time."""
    if target_time <= c:
        return 1
    if a <= 0:
        return 1
    ratio = (target_time - c) / a
    if ratio <= 0:
        return 1
    return max(1, int(math.ceil(ratio ** (-1.0 / b))))


# ---------------------------------------------------------------------------
# 5. Enhanced SM-2 with music-domain stability priors
# ---------------------------------------------------------------------------

DOMAIN_STABILITY: Dict[MusicSkill, float] = {
    MusicSkill.RHYTHM_READING: 3.0,
    MusicSkill.RHYTHM_PERFORMANCE: 2.5,
    MusicSkill.INTERVAL_RECOGNITION: 2.0,
    MusicSkill.INTERVAL_SINGING: 1.8,
    MusicSkill.CHORD_IDENTIFICATION: 2.2,
    MusicSkill.CHORD_VOICING: 1.5,
    MusicSkill.SIGHT_SINGING: 1.5,
    MusicSkill.SIGHT_READING_TREBLE: 2.0,
    MusicSkill.SIGHT_READING_BASS: 1.8,
    MusicSkill.KEY_IDENTIFICATION: 3.5,
    MusicSkill.SCALE_CONSTRUCTION: 3.0,
    MusicSkill.MELODIC_DICTATION: 1.5,
    MusicSkill.HARMONIC_DICTATION: 1.3,
    MusicSkill.FORM_ANALYSIS: 3.5,
    MusicSkill.TEMPO_DISCRIMINATION: 2.8,
    MusicSkill.DYNAMICS_EXPRESSION: 2.5,
    MusicSkill.TIMBRE_RECOGNITION: 2.0,
}


@dataclass
class CurriculumItem:
    item_id: str
    skill: MusicSkill
    difficulty: float
    discrimination: float = 1.0
    ease_factor: float = 2.5
    interval_days: float = 1.0
    repetitions: int = 0
    last_review_day: float = 0.0
    lapse_count: int = 0


def sm2_music_update(item: CurriculumItem, quality: int,
                     current_day: float) -> float:
    """SM-2 update with per-skill stability priors. Returns next review day."""
    q = max(0, min(5, quality))
    ef = item.ease_factor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    ef = max(1.3, ef)

    stability = DOMAIN_STABILITY.get(item.skill, 2.5)

    if q < 3:
        interval = max(0.5, stability * 0.3)
        item.repetitions = 0
        item.lapse_count += 1
    else:
        item.repetitions += 1
        if item.repetitions == 1:
            interval = max(1.0, stability * 0.5)
        elif item.repetitions == 2:
            interval = max(3.0, stability * 2.0)
        else:
            interval = item.interval_days * ef

    item.ease_factor = ef
    item.interval_days = interval
    item.last_review_day = current_day
    return current_day + interval


def schedule_curriculum_session(items: List[CurriculumItem],
                                current_day: float,
                                max_reviews: int = 25,
                                target_retention: float = 0.85) -> List[str]:
    """Select overdue items prioritised by retention deficit."""
    overdue = []
    for item in items:
        days_elapsed = current_day - item.last_review_day
        stability = DOMAIN_STABILITY.get(item.skill, 2.5)
        retention = math.exp(-0.2 * days_elapsed / max(stability * item.ease_factor, 0.01))
        if retention < target_retention:
            overdue.append((retention, item.item_id))
    overdue.sort()
    return [iid for _, iid in overdue[:max_reviews]]


# ---------------------------------------------------------------------------
# 6. Progress analytics
# ---------------------------------------------------------------------------

@dataclass
class ProgressSnapshot:
    skill: MusicSkill
    ability: float
    level: str
    accuracy: float
    n_attempts: int
    mastery_status: str  # "mastery", "non_mastery", "in_progress"
    learning_rate: float  # power-law b exponent, or 0 if < 5 attempts


def accuracy_trajectory(responses: List[bool], window: int = 10) -> np.ndarray:
    """Rolling accuracy over a sliding window."""
    if not responses:
        return np.array([])
    arr = np.array(responses, dtype=float)
    if len(arr) < window:
        return np.array([np.mean(arr)])
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0)
    return (cumsum[window:] - cumsum[:-window]) / window


def time_on_task_summary(response_times_s: np.ndarray) -> Dict[str, float]:
    """Summary statistics for per-item response times."""
    if len(response_times_s) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "total": 0.0, "n": 0}
    return {
        "mean": float(np.mean(response_times_s)),
        "median": float(np.median(response_times_s)),
        "std": float(np.std(response_times_s)),
        "total": float(np.sum(response_times_s)),
        "n": len(response_times_s),
    }


def skill_gap_analysis(abilities: Dict[MusicSkill, float]) -> List[Tuple[MusicSkill, float]]:
    """Identify skills with the largest gap below their prerequisites' mean ability."""
    gaps = []
    for skill in MusicSkill:
        prereqs = SKILL_PREREQUISITES.get(skill, [])
        if not prereqs:
            continue
        prereq_mean = np.mean([abilities.get(p, -3.0) for p in prereqs])
        current = abilities.get(skill, -3.0)
        gap = prereq_mean - current
        if gap > 0.3:
            gaps.append((skill, float(gap)))
    gaps.sort(key=lambda x: -x[1])
    return gaps


def compute_progress_snapshot(skill: MusicSkill,
                              ability: float,
                              responses: List[bool],
                              response_times: Optional[np.ndarray] = None) -> ProgressSnapshot:
    """Build a progress snapshot for one skill."""
    n = len(responses)
    acc = sum(responses) / n if n > 0 else 0.0

    if n >= 5:
        decision, _ = run_sprt(responses[-min(n, 30):])
    else:
        decision = "in_progress"

    lr_est = 0.0
    if response_times is not None and len(response_times) >= 5:
        trials = np.arange(1, len(response_times) + 1, dtype=float)
        _, b, _ = fit_power_law(trials, response_times)
        lr_est = b

    return ProgressSnapshot(
        skill=skill,
        ability=ability,
        level=skill_level_label(ability),
        accuracy=acc,
        n_attempts=n,
        mastery_status=decision,
        learning_rate=lr_est,
    )
