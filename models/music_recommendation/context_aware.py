"""Context-aware music recommendation: temporal, mood, activity contexts and bandits."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class TimeOfDay(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    LATE_NIGHT = "late_night"


class Activity(str, Enum):
    WORKOUT = "workout"
    STUDY = "study"
    COMMUTE = "commute"
    RELAX = "relax"
    PARTY = "party"
    SLEEP = "sleep"
    WORK = "work"


@dataclass
class ListeningContext:
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON
    activity: Activity = Activity.RELAX
    mood_valence: float = 0.5
    mood_arousal: float = 0.5
    weather_temp_c: Optional[float] = None
    is_weekend: bool = False

    def to_vector(self) -> np.ndarray:
        time_map = {TimeOfDay.MORNING: 0, TimeOfDay.AFTERNOON: 1,
                    TimeOfDay.EVENING: 2, TimeOfDay.NIGHT: 3, TimeOfDay.LATE_NIGHT: 4}
        act_map = {Activity.WORKOUT: 0, Activity.STUDY: 1, Activity.COMMUTE: 2,
                   Activity.RELAX: 3, Activity.PARTY: 4, Activity.SLEEP: 5, Activity.WORK: 6}
        time_oh = np.zeros(5)
        time_oh[time_map[self.time_of_day]] = 1.0
        act_oh = np.zeros(7)
        act_oh[act_map[self.activity]] = 1.0
        return np.concatenate([
            time_oh, act_oh,
            [self.mood_valence, self.mood_arousal,
             (self.weather_temp_c or 20.0) / 40.0,
             float(self.is_weekend)],
        ])


@dataclass
class ContextualPreference:
    context_weights: np.ndarray
    item_features: np.ndarray
    bias: float = 0.0

    def score(self, context: ListeningContext, item_idx: int) -> float:
        ctx_vec = context.to_vector()
        item_vec = self.item_features[item_idx]
        ctx_proj = ctx_vec @ self.context_weights[:ctx_vec.shape[0], :item_vec.shape[0]]
        return float(np.dot(ctx_proj, item_vec) + self.bias)


@dataclass
class BanditArm:
    item_idx: int
    n_pulls: int = 0
    total_reward: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    def update(self, reward: float) -> None:
        self.n_pulls += 1
        self.total_reward += reward
        if reward > 0.5:
            self.alpha += 1
        else:
            self.beta += 1


def epsilon_greedy(arms: List[BanditArm], epsilon: float = 0.1,
                   rng: Optional[np.random.Generator] = None) -> int:
    rng = rng or np.random.default_rng()
    if rng.random() < epsilon:
        return rng.integers(0, len(arms))
    return max(range(len(arms)), key=lambda i: arms[i].mean_reward)


def ucb1(arms: List[BanditArm], total_pulls: Optional[int] = None) -> int:
    if total_pulls is None:
        total_pulls = sum(a.n_pulls for a in arms)
    if total_pulls == 0:
        return 0
    scores = []
    for arm in arms:
        if arm.n_pulls == 0:
            scores.append(float("inf"))
        else:
            exploit = arm.mean_reward
            explore = math.sqrt(2.0 * math.log(total_pulls) / arm.n_pulls)
            scores.append(exploit + explore)
    return int(np.argmax(scores))


def thompson_sampling(arms: List[BanditArm],
                      rng: Optional[np.random.Generator] = None) -> int:
    rng = rng or np.random.default_rng()
    samples = [rng.beta(arm.alpha, arm.beta) for arm in arms]
    return int(np.argmax(samples))


@dataclass
class SessionState:
    listened: List[int] = field(default_factory=list)
    skipped: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    session_start: float = 0.0

    def add_listen(self, item_idx: int, timestamp: float) -> None:
        self.listened.append(item_idx)
        self.timestamps.append(timestamp)

    def add_skip(self, item_idx: int) -> None:
        self.skipped.append(item_idx)

    @property
    def duration(self) -> float:
        if not self.timestamps:
            return 0.0
        return self.timestamps[-1] - self.session_start

    @property
    def skip_rate(self) -> float:
        total = len(self.listened) + len(self.skipped)
        return len(self.skipped) / total if total > 0 else 0.0


def session_decay_scores(item_scores: np.ndarray, session: SessionState,
                         decay: float = 0.9) -> np.ndarray:
    adjusted = item_scores.copy()
    for i, idx in enumerate(reversed(session.listened)):
        weight = decay ** (i + 1)
        adjusted[idx] *= (1.0 - weight)
    for idx in session.skipped:
        adjusted[idx] *= 0.3
    return adjusted


@dataclass
class ListeningHistory:
    play_counts: Dict[int, int] = field(default_factory=dict)
    last_played: Dict[int, float] = field(default_factory=dict)
    total_time: Dict[int, float] = field(default_factory=dict)

    def record(self, item_idx: int, timestamp: float, duration: float) -> None:
        self.play_counts[item_idx] = self.play_counts.get(item_idx, 0) + 1
        self.last_played[item_idx] = max(self.last_played.get(item_idx, 0.0), timestamp)
        self.total_time[item_idx] = self.total_time.get(item_idx, 0.0) + duration

    def frequency(self, item_idx: int) -> int:
        return self.play_counts.get(item_idx, 0)

    def recency(self, item_idx: int, now: float) -> float:
        lp = self.last_played.get(item_idx)
        if lp is None:
            return 0.0
        delta = now - lp
        return math.exp(-delta / 86400.0)

    def engagement(self, item_idx: int, track_duration: float) -> float:
        if track_duration <= 0:
            return 0.0
        total = self.total_time.get(item_idx, 0.0)
        count = self.play_counts.get(item_idx, 0)
        if count == 0:
            return 0.0
        avg_listen = total / count
        return min(avg_listen / track_duration, 1.0)


def freshness_score(item_release_time: float, now: float,
                    half_life_days: float = 30.0) -> float:
    delta_days = (now - item_release_time) / 86400.0
    if delta_days < 0:
        return 1.0
    return math.exp(-0.693 * delta_days / half_life_days)


def recency_boost(base_score: float, item_release_time: float, now: float,
                  boost: float = 0.2, half_life_days: float = 14.0) -> float:
    fresh = freshness_score(item_release_time, now, half_life_days)
    return base_score + boost * fresh


def context_similarity(ctx_a: ListeningContext, ctx_b: ListeningContext) -> float:
    va = ctx_a.to_vector()
    vb = ctx_b.to_vector()
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class ContextualHistory:
    entries: List[Tuple[ListeningContext, int, float]] = field(default_factory=list)

    def add(self, context: ListeningContext, item_idx: int, reward: float) -> None:
        self.entries.append((context, item_idx, reward))

    def context_knn_score(self, context: ListeningContext, item_idx: int,
                          k: int = 5) -> float:
        relevant = [(ctx, rew) for ctx, idx, rew in self.entries if idx == item_idx]
        if not relevant:
            return 0.0
        sims = [(context_similarity(context, ctx), rew) for ctx, rew in relevant]
        sims.sort(key=lambda x: -x[0])
        top = sims[:k]
        w_sum = sum(s for s, _ in top)
        if w_sum < 1e-12:
            return 0.0
        return sum(s * r for s, r in top) / w_sum


ACTIVITY_AUDIO_PRIORS: Dict[Activity, Dict[str, float]] = {
    Activity.WORKOUT: {"tempo_target": 130.0, "energy_target": 0.9, "valence_target": 0.7},
    Activity.STUDY: {"tempo_target": 80.0, "energy_target": 0.2, "valence_target": 0.5},
    Activity.COMMUTE: {"tempo_target": 110.0, "energy_target": 0.5, "valence_target": 0.6},
    Activity.RELAX: {"tempo_target": 90.0, "energy_target": 0.3, "valence_target": 0.6},
    Activity.PARTY: {"tempo_target": 125.0, "energy_target": 0.85, "valence_target": 0.8},
    Activity.SLEEP: {"tempo_target": 60.0, "energy_target": 0.1, "valence_target": 0.4},
    Activity.WORK: {"tempo_target": 100.0, "energy_target": 0.4, "valence_target": 0.5},
}


def activity_match_score(activity: Activity, item_tempo: float,
                         item_energy: float, item_valence: float) -> float:
    priors = ACTIVITY_AUDIO_PRIORS.get(activity, ACTIVITY_AUDIO_PRIORS[Activity.RELAX])
    tempo_diff = abs(item_tempo - priors["tempo_target"]) / 60.0
    energy_diff = abs(item_energy - priors["energy_target"])
    valence_diff = abs(item_valence - priors["valence_target"])
    distance = math.sqrt(tempo_diff**2 + energy_diff**2 + valence_diff**2)
    return math.exp(-distance)
