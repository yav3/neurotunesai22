"""Disease progression solvers.

Three models from progression_parameters.json:

1. Linear decline:
    Y(t) = Y(0) + β·t + σ·W(t)
    where W(t) is AR(1) noise: ε(t) = ρ·ε(t-1) + √(1-ρ²)·z(t)

2. Episodic (MDD, BD, MS):
    Relapse/remission cycles with specified transition probabilities
    State: {remission, episode}
    During episode: biomarker shifts by Δ_episode
    Baseline drifts linearly between episodes

3. Stable with events (EP):
    Baseline stable, punctuated by stochastic events
    Events: Poisson process with rate λ, transient perturbation
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LinearProgressionParams:
    baseline: float
    annual_rate: float
    rate_std: float
    ar1_rho: float = 0.6
    noise_std: float = 0.0


@dataclass
class EpisodicParams:
    relapse_prob_per_year: float
    mean_episode_months: float
    std_episode_months: float
    remission_target: float
    episode_target: float
    baseline_annual_rate: float = 0.0


@dataclass
class EventParams:
    event_rate_per_year: float
    event_magnitude: float
    event_std: float
    recovery_hours: float
    baseline_annual_rate: float = 0.0


class ProgressionSolver:

    @staticmethod
    def linear_trajectory(
        params: LinearProgressionParams,
        time_months: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Y(t) = Y(0) + β·(t/12) + AR(1) noise."""
        rng = rng or np.random.default_rng()
        individual_rate = rng.normal(params.annual_rate, params.rate_std)
        t_years = time_months / 12.0
        trend = params.baseline + individual_rate * t_years

        # AR(1) residuals
        n = len(time_months)
        eps = np.zeros(n)
        for i in range(1, n):
            z = rng.normal(0, params.noise_std) if params.noise_std > 0 else 0
            eps[i] = params.ar1_rho * eps[i - 1] + np.sqrt(1 - params.ar1_rho**2) * z

        return trend + eps

    @staticmethod
    def episodic_trajectory(
        params: EpisodicParams,
        time_months: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate relapse/remission cycles."""
        rng = rng or np.random.default_rng()
        n = len(time_months)
        values = np.zeros(n)
        in_episode = False
        episode_end = -1.0
        baseline = params.remission_target

        monthly_relapse_prob = 1 - (1 - params.relapse_prob_per_year) ** (1.0 / 12.0)

        for i, t in enumerate(time_months):
            baseline += params.baseline_annual_rate / 12.0

            if not in_episode:
                if rng.random() < monthly_relapse_prob:
                    in_episode = True
                    dur = max(1, rng.normal(params.mean_episode_months, params.std_episode_months))
                    episode_end = t + dur
                values[i] = baseline + rng.normal(0, 1.0)
            else:
                values[i] = params.episode_target + rng.normal(0, 2.0)
                if t >= episode_end:
                    in_episode = False

        return values

    @staticmethod
    def event_trajectory(
        params: EventParams,
        time_months: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Stable baseline punctuated by Poisson events."""
        rng = rng or np.random.default_rng()
        n = len(time_months)
        baseline = np.zeros(n)
        events = np.zeros(n)

        monthly_rate = params.event_rate_per_year / 12.0
        recovery_months = params.recovery_hours / (24 * 30)

        for i, t in enumerate(time_months):
            baseline[i] = params.baseline_annual_rate * t / 12.0
            if rng.random() < monthly_rate:
                mag = rng.normal(params.event_magnitude, params.event_std)
                for j in range(i, n):
                    dt = time_months[j] - t
                    if dt < recovery_months:
                        events[j] += mag * (1 - dt / recovery_months)

        return baseline + events

    @staticmethod
    def mci_to_ad_conversion(
        time_years: float,
        annual_prob: float = 0.12,
        rng: np.random.Generator | None = None,
    ) -> tuple[bool, float]:
        """Simulate whether MCI → AD conversion occurs, and when.
        Returns (converted, conversion_time_years).
        """
        rng = rng or np.random.default_rng()
        for year in range(int(np.ceil(time_years))):
            if rng.random() < annual_prob:
                return True, float(year) + rng.random()
        return False, float("inf")
