"""Audience analytics: listener demographics, engagement metrics, fan segmentation.

Models listener behavior, engagement patterns, fan segmentation by listening
profile, and retention/churn prediction for music catalog analytics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ListenerSegment(Enum):
    CASUAL = "casual"
    REGULAR = "regular"
    SUPERFAN = "superfan"
    DORMANT = "dormant"
    LAPSED = "lapsed"


@dataclass(frozen=True)
class ListenerProfile:
    listener_id: str
    total_streams: int
    unique_tracks: int
    avg_completion_rate: float
    days_active_last_90: int
    top_genres: list[str] = field(default_factory=list)
    playlist_adds: int = 0
    shares: int = 0
    saves: int = 0

    @property
    def engagement_score(self) -> float:
        stream_score = min(1.0, self.total_streams / 500.0)
        diversity = min(1.0, self.unique_tracks / max(self.total_streams, 1))
        completion = self.avg_completion_rate
        activity = min(1.0, self.days_active_last_90 / 60.0)
        social = min(1.0, (self.playlist_adds + self.shares + self.saves) / 20.0)
        return 0.30 * stream_score + 0.15 * diversity + 0.25 * completion + 0.20 * activity + 0.10 * social


@dataclass(frozen=True)
class AudienceMetrics:
    total_listeners: int
    monthly_active: int
    avg_streams_per_listener: float
    avg_completion: float
    save_rate: float
    segment_distribution: dict[str, float]
    retention_30d: float
    genre_affinity: dict[str, float]


def classify_listener(profile: ListenerProfile) -> ListenerSegment:
    if profile.days_active_last_90 == 0:
        return ListenerSegment.LAPSED
    if profile.days_active_last_90 < 5 and profile.total_streams < 10:
        return ListenerSegment.DORMANT
    score = profile.engagement_score
    if score > 0.7:
        return ListenerSegment.SUPERFAN
    if score > 0.3:
        return ListenerSegment.REGULAR
    return ListenerSegment.CASUAL


def segment_audience(
    profiles: list[ListenerProfile],
) -> dict[ListenerSegment, list[ListenerProfile]]:
    result: dict[ListenerSegment, list[ListenerProfile]] = {s: [] for s in ListenerSegment}
    for p in profiles:
        seg = classify_listener(p)
        result[seg].append(p)
    return result


def compute_audience_metrics(
    profiles: list[ListenerProfile],
) -> AudienceMetrics:
    if not profiles:
        return AudienceMetrics(0, 0, 0.0, 0.0, 0.0, {}, 0.0, {})
    total = len(profiles)
    active = sum(1 for p in profiles if p.days_active_last_90 > 0)
    avg_streams = float(np.mean([p.total_streams for p in profiles]))
    avg_comp = float(np.mean([p.avg_completion_rate for p in profiles]))
    total_saves = sum(p.saves for p in profiles)
    total_streams = sum(p.total_streams for p in profiles)
    save_rate = total_saves / max(total_streams, 1)
    segments = segment_audience(profiles)
    seg_dist = {s.value: len(v) / total for s, v in segments.items()}
    retained = sum(1 for p in profiles if p.days_active_last_90 >= 30)
    retention = retained / max(total, 1)
    genre_counts: dict[str, int] = {}
    for p in profiles:
        for g in p.top_genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    genre_aff = {g: c / total for g, c in genre_counts.items()}
    return AudienceMetrics(
        total_listeners=total,
        monthly_active=active,
        avg_streams_per_listener=avg_streams,
        avg_completion=avg_comp,
        save_rate=save_rate,
        segment_distribution=seg_dist,
        retention_30d=retention,
        genre_affinity=genre_aff,
    )


def churn_risk(profile: ListenerProfile, lookback_days: int = 90) -> float:
    if profile.days_active_last_90 == 0:
        return 1.0
    recency_risk = 1.0 - min(1.0, profile.days_active_last_90 / lookback_days)
    engagement_risk = 1.0 - profile.engagement_score
    completion_risk = 1.0 - profile.avg_completion_rate
    return 0.4 * recency_risk + 0.35 * engagement_risk + 0.25 * completion_risk


def lifetime_value_estimate(
    profile: ListenerProfile,
    revenue_per_stream: float = 0.004,
    months_horizon: int = 12,
) -> float:
    monthly_rate = profile.total_streams * (profile.days_active_last_90 / 90.0) * (30.0 / max(profile.days_active_last_90, 1))
    churn = churn_risk(profile)
    monthly_retention = 1.0 - churn * 0.1
    total_rev = 0.0
    for m in range(months_horizon):
        total_rev += monthly_rate * revenue_per_stream * (monthly_retention ** m)
    return round(total_rev, 2)


def genre_overlap_matrix(
    profiles: list[ListenerProfile],
    genres: list[str],
) -> np.ndarray:
    n = len(genres)
    co_occur = np.zeros((n, n))
    genre_idx = {g: i for i, g in enumerate(genres)}
    for p in profiles:
        present = [genre_idx[g] for g in p.top_genres if g in genre_idx]
        for i in present:
            for j in present:
                co_occur[i, j] += 1
    diag = np.sqrt(np.diag(co_occur).clip(1))
    return co_occur / np.outer(diag, diag)
