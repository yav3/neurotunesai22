"""Dose-response modeling for music therapy interventions.

Models the relationship between therapy parameters (duration, frequency,
intensity) and therapeutic outcomes using sigmoid dose-response curves,
threshold detection, and optimal dosing estimation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TherapyDose:
    duration_min: float
    frequency_hz: float
    intensity_db: float
    session_count: int = 1

    @property
    def total_exposure_min(self) -> float:
        return self.duration_min * self.session_count


@dataclass
class DoseResponseCurve:
    ec50: float
    hill_coefficient: float
    e_max: float
    baseline: float = 0.0

    def response(self, dose: float) -> float:
        if dose <= 0:
            return self.baseline
        return self.baseline + self.e_max * (
            dose ** self.hill_coefficient /
            (self.ec50 ** self.hill_coefficient + dose ** self.hill_coefficient)
        )

    def inverse(self, target_response: float) -> float:
        if target_response <= self.baseline:
            return 0.0
        if target_response >= self.baseline + self.e_max:
            return float('inf')
        frac = (target_response - self.baseline) / self.e_max
        return self.ec50 * (frac / (1.0 - frac)) ** (1.0 / self.hill_coefficient)


def fit_dose_response(doses: np.ndarray, responses: np.ndarray,
                      max_iter: int = 200) -> DoseResponseCurve:
    """Fit a Hill-equation dose-response curve via iterative least squares."""
    doses = np.asarray(doses, dtype=float)
    responses = np.asarray(responses, dtype=float)

    baseline = float(np.min(responses))
    e_max = float(np.max(responses) - baseline)
    if e_max < 1e-12:
        return DoseResponseCurve(ec50=1.0, hill_coefficient=1.0,
                                 e_max=0.0, baseline=baseline)

    ec50 = float(np.median(doses[doses > 0])) if np.any(doses > 0) else 1.0
    n = 1.0

    for _ in range(max_iter):
        pred = np.array([baseline + e_max * (d**n / (ec50**n + d**n))
                         if d > 0 else baseline for d in doses])
        residuals = responses - pred
        ssr = float(np.sum(residuals ** 2))
        if ssr < 1e-12:
            break

        for delta in [0.01, -0.01]:
            ec50_t = ec50 * (1 + delta)
            pred_t = np.array([baseline + e_max * (d**n / (ec50_t**n + d**n))
                               if d > 0 else baseline for d in doses])
            if np.sum((responses - pred_t)**2) < ssr:
                ec50 = ec50_t
                break

        for delta in [0.05, -0.05]:
            n_t = max(n + delta, 0.1)
            pred_t = np.array([baseline + e_max * (d**n_t / (ec50**n_t + d**n_t))
                               if d > 0 else baseline for d in doses])
            if np.sum((responses - pred_t)**2) < ssr:
                n = n_t
                break

    return DoseResponseCurve(ec50=ec50, hill_coefficient=n,
                             e_max=e_max, baseline=baseline)


def therapeutic_threshold(curve: DoseResponseCurve,
                          effect_fraction: float = 0.2) -> float:
    """Minimum dose for a clinically meaningful effect."""
    target = curve.baseline + effect_fraction * curve.e_max
    return curve.inverse(target)


def optimal_dose(curve: DoseResponseCurve,
                 target_fraction: float = 0.8) -> float:
    """Dose achieving target fraction of maximum effect."""
    target = curve.baseline + target_fraction * curve.e_max
    return curve.inverse(target)


def dose_efficiency(curve: DoseResponseCurve, dose: float) -> float:
    """Marginal efficiency: response per unit dose at given dose."""
    if dose <= 0:
        return 0.0
    resp = curve.response(dose)
    return (resp - curve.baseline) / dose


def diminishing_returns_point(curve: DoseResponseCurve,
                              threshold: float = 0.01) -> float:
    """Dose beyond which marginal gain per unit dose drops below threshold."""
    test_doses = np.linspace(0.01, curve.ec50 * 10, 500)
    for i in range(1, len(test_doses)):
        d = test_doses[i]
        dd = test_doses[i] - test_doses[i - 1]
        marginal = (curve.response(d) - curve.response(d - dd)) / dd
        if marginal < threshold:
            return float(d)
    return float(test_doses[-1])


def session_schedule(curve: DoseResponseCurve,
                     total_sessions: int,
                     decay_rate: float = 0.05) -> List[TherapyDose]:
    """Generate a tapering session schedule based on dose-response."""
    opt = optimal_dose(curve)
    schedule = []
    for i in range(total_sessions):
        decay = np.exp(-decay_rate * i)
        dur = opt * decay
        schedule.append(TherapyDose(
            duration_min=max(dur, 5.0),
            frequency_hz=1.0,
            intensity_db=65.0,
            session_count=1,
        ))
    return schedule


def cumulative_response(curve: DoseResponseCurve,
                        doses: List[float],
                        carry_over: float = 0.3) -> np.ndarray:
    """Compute cumulative therapeutic response across sessions with carry-over."""
    responses = np.zeros(len(doses))
    accumulated = 0.0
    for i, d in enumerate(doses):
        effective = d + carry_over * accumulated
        responses[i] = curve.response(effective)
        accumulated = responses[i] - curve.baseline
    return responses
