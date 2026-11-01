"""Audience segmentation: listener clustering, demographic modeling, engagement scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class EngagementLevel(str, Enum):
    SUPERFAN = "superfan"
    ACTIVE = "active"
    CASUAL = "casual"
    DORMANT = "dormant"
    CHURNED = "churned"


@dataclass
class ListenerProfile:
    listener_id: str
    features: np.ndarray
    streams_per_month: float = 0.0
    save_rate: float = 0.0
    skip_rate: float = 0.0
    playlist_adds: int = 0
    last_active_days: int = 0


@dataclass
class Segment:
    segment_id: int
    centroid: np.ndarray
    size: int
    engagement: EngagementLevel
    avg_streams: float = 0.0
    genre_affinity: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    segments: List[Segment]
    labels: np.ndarray
    inertia: float = 0.0


def engagement_score(profile: ListenerProfile) -> float:
    stream_score = min(1.0, profile.streams_per_month / 50.0)
    save_score = profile.save_rate
    skip_penalty = profile.skip_rate * 0.5
    recency = max(0.0, 1.0 - profile.last_active_days / 90.0)
    playlist_bonus = min(0.2, profile.playlist_adds * 0.02)
    return max(0.0, min(1.0, stream_score * 0.35 + save_score * 0.25 - skip_penalty + recency * 0.25 + playlist_bonus))


def classify_engagement(score: float, last_active_days: int) -> EngagementLevel:
    if last_active_days > 180:
        return EngagementLevel.CHURNED
    if last_active_days > 60:
        return EngagementLevel.DORMANT
    if score >= 0.8:
        return EngagementLevel.SUPERFAN
    if score >= 0.4:
        return EngagementLevel.ACTIVE
    return EngagementLevel.CASUAL


def kmeans_segment(
    features: np.ndarray, k: int = 4, max_iter: int = 50, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    if n == 0 or k <= 0:
        return np.array([]), np.array([]).reshape(0, 0), 0.0
    k = min(k, n)
    indices = rng.choice(n, size=k, replace=False)
    centroids = features[indices].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.array([np.linalg.norm(features - c, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = features[mask].mean(axis=0)
    inertia = sum(
        float(np.sum((features[labels == j] - centroids[j]) ** 2))
        for j in range(k)
    )
    return labels, centroids, inertia


def segment_audience(
    profiles: List[ListenerProfile], k: int = 4, seed: int = 42,
) -> SegmentationResult:
    if not profiles:
        return SegmentationResult(segments=[], labels=np.array([]), inertia=0.0)
    features = np.array([p.features for p in profiles])
    labels, centroids, inertia = kmeans_segment(features, k=k, seed=seed)
    segments = []
    for j in range(len(centroids)):
        mask = labels == j
        seg_profiles = [p for p, m in zip(profiles, mask) if m]
        avg_streams = np.mean([p.streams_per_month for p in seg_profiles]) if seg_profiles else 0.0
        avg_eng = np.mean([engagement_score(p) for p in seg_profiles]) if seg_profiles else 0.0
        avg_days = int(np.mean([p.last_active_days for p in seg_profiles])) if seg_profiles else 0
        eng_level = classify_engagement(avg_eng, avg_days)
        segments.append(Segment(
            segment_id=j, centroid=centroids[j], size=int(np.sum(mask)),
            engagement=eng_level, avg_streams=float(avg_streams),
        ))
    return SegmentationResult(segments=segments, labels=labels, inertia=inertia)


def elbow_analysis(
    features: np.ndarray, k_range: range = range(2, 8), seed: int = 42,
) -> List[Tuple[int, float]]:
    results = []
    for k in k_range:
        _, _, inertia = kmeans_segment(features, k=k, seed=seed)
        results.append((k, inertia))
    return results


def segment_overlap(seg_a: Segment, seg_b: Segment) -> float:
    dist = float(np.linalg.norm(seg_a.centroid - seg_b.centroid))
    return max(0.0, 1.0 - dist / max(dist + 1.0, 1e-12))


def churn_risk(profile: ListenerProfile) -> float:
    eng = engagement_score(profile)
    recency_factor = min(1.0, profile.last_active_days / 90.0)
    skip_factor = profile.skip_rate
    return max(0.0, min(1.0, recency_factor * 0.5 + skip_factor * 0.3 + (1.0 - eng) * 0.2))


def lifetime_value_estimate(
    profile: ListenerProfile,
    revenue_per_stream: float = 0.004,
    retention_months: float = 12.0,
) -> float:
    monthly_rev = profile.streams_per_month * revenue_per_stream
    risk = churn_risk(profile)
    effective_months = retention_months * (1.0 - risk * 0.5)
    return monthly_rev * effective_months


def segment_summary(result: SegmentationResult) -> Dict[str, any]:
    return {
        "n_segments": len(result.segments),
        "total_listeners": int(np.sum([s.size for s in result.segments])),
        "inertia": result.inertia,
        "segments": [
            {
                "id": s.segment_id,
                "size": s.size,
                "engagement": s.engagement.value,
                "avg_streams": round(s.avg_streams, 1),
            }
            for s in result.segments
        ],
    }
