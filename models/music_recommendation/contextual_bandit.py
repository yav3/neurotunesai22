"""Contextual bandit recommendation: LinUCB, Thompson sampling, epsilon-greedy.

Exploration-exploitation framework for music recommendation where each
track is an arm and user/context features determine reward estimates.
Pure numpy — no external RL libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class BanditStrategy(str, Enum):
    LINUCB = "linucb"
    THOMPSON = "thompson"
    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"
    EXP3 = "exp3"


@dataclass
class ArmStats:
    arm_id: int
    pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0


@dataclass
class LinUCBArm:
    d: int
    A: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)
    alpha: float = 1.0

    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(self.d)
        if self.b is None:
            self.b = np.zeros(self.d)

    def predict(self, x: np.ndarray) -> float:
        A_inv = np.linalg.solve(self.A, np.eye(self.d))
        theta = A_inv @ self.b
        p = float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
        return p

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x


class LinUCBRecommender:
    """Linear Upper Confidence Bound contextual bandit."""

    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.arms: dict[int, LinUCBArm] = {}

    def add_arm(self, arm_id: int) -> None:
        self.arms[arm_id] = LinUCBArm(d=self.n_features, alpha=self.alpha)

    def recommend(self, context: np.ndarray, candidate_ids: list[int],
                  k: int = 5) -> list[int]:
        scores = {}
        for arm_id in candidate_ids:
            if arm_id not in self.arms:
                self.add_arm(arm_id)
            scores[arm_id] = self.arms[arm_id].predict(context)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:k]

    def update(self, arm_id: int, context: np.ndarray, reward: float) -> None:
        if arm_id not in self.arms:
            self.add_arm(arm_id)
        self.arms[arm_id].update(context, reward)

    def cumulative_regret(self, rewards: np.ndarray,
                          optimal_rewards: np.ndarray) -> np.ndarray:
        return np.cumsum(optimal_rewards - rewards)


class ThompsonSamplingRecommender:
    """Thompson sampling with Beta(alpha, beta) posteriors per arm."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.alphas: dict[int, float] = {}
        self.betas: dict[int, float] = {}

    def add_arm(self, arm_id: int, prior_alpha: float = 1.0,
                prior_beta: float = 1.0) -> None:
        self.alphas[arm_id] = prior_alpha
        self.betas[arm_id] = prior_beta

    def recommend(self, candidate_ids: list[int], k: int = 5) -> list[int]:
        samples = {}
        for arm_id in candidate_ids:
            if arm_id not in self.alphas:
                self.add_arm(arm_id)
            samples[arm_id] = self.rng.beta(self.alphas[arm_id],
                                             self.betas[arm_id])
        ranked = sorted(samples, key=samples.get, reverse=True)
        return ranked[:k]

    def update(self, arm_id: int, reward: float) -> None:
        if arm_id not in self.alphas:
            self.add_arm(arm_id)
        if reward > 0.5:
            self.alphas[arm_id] += 1.0
        else:
            self.betas[arm_id] += 1.0

    def expected_reward(self, arm_id: int) -> float:
        a = self.alphas.get(arm_id, 1.0)
        b = self.betas.get(arm_id, 1.0)
        return a / (a + b)


class EpsilonGreedyRecommender:
    """Epsilon-greedy with optional decay schedule."""

    def __init__(self, epsilon: float = 0.1, decay: float = 1.0,
                 seed: int = 42):
        self.epsilon = epsilon
        self.decay = decay
        self.rng = np.random.default_rng(seed)
        self.stats: dict[int, ArmStats] = {}
        self.t = 0

    def _current_epsilon(self) -> float:
        return self.epsilon * (self.decay ** self.t)

    def recommend(self, candidate_ids: list[int], k: int = 5) -> list[int]:
        self.t += 1
        eps = self._current_epsilon()
        result = []
        remaining = list(candidate_ids)

        for _ in range(min(k, len(remaining))):
            if self.rng.random() < eps or not any(
                    cid in self.stats for cid in remaining):
                idx = self.rng.integers(0, len(remaining))
                chosen = remaining[idx]
            else:
                best_id = max(
                    (cid for cid in remaining if cid in self.stats),
                    key=lambda c: self.stats[c].mean_reward,
                    default=remaining[0],
                )
                chosen = best_id
            result.append(chosen)
            remaining.remove(chosen)
        return result

    def update(self, arm_id: int, reward: float) -> None:
        if arm_id not in self.stats:
            self.stats[arm_id] = ArmStats(arm_id=arm_id)
        s = self.stats[arm_id]
        s.pulls += 1
        s.total_reward += reward
        s.mean_reward = s.total_reward / s.pulls


def ucb1_scores(stats: list[ArmStats], t: int,
                c: float = 2.0) -> np.ndarray:
    scores = np.zeros(len(stats))
    for i, s in enumerate(stats):
        if s.pulls == 0:
            scores[i] = float("inf")
        else:
            exploitation = s.mean_reward
            exploration = c * np.sqrt(np.log(max(t, 1)) / s.pulls)
            scores[i] = exploitation + exploration
    return scores


def exp3_weights(losses: np.ndarray, eta: float = 0.1) -> np.ndarray:
    w = np.exp(-eta * losses)
    return w / (w.sum() + 1e-12)


@dataclass
class RewardSignal:
    skip: float = 0.0
    partial_listen: float = 0.3
    full_listen: float = 0.7
    repeat: float = 1.0
    save: float = 1.0
    thumbs_down: float = -0.5

    def compute(self, listen_fraction: float, saved: bool = False,
                repeated: bool = False,
                thumbs_down: bool = False) -> float:
        if thumbs_down:
            return self.thumbs_down
        if listen_fraction < 0.1:
            r = self.skip
        elif listen_fraction < 0.8:
            r = self.partial_listen + (self.full_listen - self.partial_listen) * (
                (listen_fraction - 0.1) / 0.7
            )
        else:
            r = self.full_listen
        if repeated:
            r = max(r, self.repeat)
        if saved:
            r = max(r, self.save)
        return float(np.clip(r, -1.0, 1.0))


def evaluate_bandit(rewards_per_step: list[list[float]],
                    optimal_per_step: list[float]) -> dict:
    rewards_flat = np.array([max(r) if r else 0.0 for r in rewards_per_step])
    optimal = np.array(optimal_per_step)
    regret = np.cumsum(optimal - rewards_flat)
    return {
        "total_reward": float(rewards_flat.sum()),
        "mean_reward": float(rewards_flat.mean()),
        "total_regret": float(regret[-1]) if len(regret) > 0 else 0.0,
        "regret_curve": regret,
    }
