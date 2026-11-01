"""Streaming analytics: platform metrics, engagement, growth, catalog health."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Platform(Enum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    TIDAL = "tidal"
    AMAZON_MUSIC = "amazon_music"
    DEEZER = "deezer"
    PANDORA = "pandora"
    SOUNDCLOUD = "soundcloud"


PER_STREAM_RATES: dict[Platform, float] = {
    Platform.SPOTIFY: 0.004,
    Platform.APPLE_MUSIC: 0.008,
    Platform.YOUTUBE_MUSIC: 0.002,
    Platform.TIDAL: 0.012,
    Platform.AMAZON_MUSIC: 0.004,
    Platform.DEEZER: 0.004,
    Platform.PANDORA: 0.003,
    Platform.SOUNDCLOUD: 0.001,
}


@dataclass
class StreamingMetrics:
    streams: int = 0
    saves: int = 0
    playlist_adds: int = 0
    skip_rate: float = 0.0
    completion_rate: float = 0.0
    share_rate: float = 0.0
    repeat_listen_rate: float = 0.0
    avg_listen_duration_s: float = 0.0


def engagement_score(metrics: StreamingMetrics) -> float:
    if metrics.streams <= 0:
        return 0.0
    save_ratio = metrics.saves / metrics.streams
    playlist_ratio = metrics.playlist_adds / metrics.streams
    raw = (
        0.25 * metrics.completion_rate
        + 0.20 * (1.0 - metrics.skip_rate)
        + 0.20 * metrics.repeat_listen_rate
        + 0.15 * min(save_ratio, 1.0)
        + 0.10 * min(playlist_ratio, 1.0)
        + 0.10 * metrics.share_rate
    )
    return float(np.clip(raw, 0.0, 1.0))


def estimated_revenue(streams_by_platform: dict[Platform, int]) -> float:
    total = 0.0
    for platform, count in streams_by_platform.items():
        rate = PER_STREAM_RATES.get(platform, 0.003)
        total += count * rate
    return total


def artist_momentum_score(
    current_monthly_streams: int,
    previous_monthly_streams: int,
    two_months_ago_streams: int,
) -> float:
    if previous_monthly_streams <= 0:
        return 0.0
    growth = (current_monthly_streams - previous_monthly_streams) / previous_monthly_streams
    if two_months_ago_streams > 0:
        prev_growth = (previous_monthly_streams - two_months_ago_streams) / two_months_ago_streams
        acceleration = growth - prev_growth
    else:
        acceleration = 0.0
    raw = growth + 0.5 * acceleration
    return float(np.tanh(raw))


def growth_projection(
    monthly_streams: list[int],
    months_ahead: int = 6,
) -> list[float]:
    if len(monthly_streams) < 2 or months_ahead <= 0:
        return []
    log_streams = [np.log1p(s) for s in monthly_streams]
    x = np.arange(len(log_streams))
    coeffs = np.polyfit(x, log_streams, 1)
    projections: list[float] = []
    for i in range(1, months_ahead + 1):
        t = len(log_streams) - 1 + i
        proj = np.expm1(coeffs[0] * t + coeffs[1])
        projections.append(float(max(proj, 0.0)))
    return projections


def viral_coefficient(shares: int, new_listeners_from_shares: int) -> float:
    if shares <= 0:
        return 0.0
    return new_listeners_from_shares / shares


def catalog_health(
    tracks: list[StreamingMetrics],
) -> dict[str, float]:
    if not tracks:
        return {"catalog_size": 0, "avg_engagement": 0.0, "dormant_fraction": 0.0, "concentration": 0.0}
    engagements = [engagement_score(t) for t in tracks]
    total_streams = sum(t.streams for t in tracks)
    dormant = sum(1 for t in tracks if t.streams < 100)
    if total_streams > 0:
        shares = np.array([t.streams / total_streams for t in tracks])
        hhi = float(np.sum(shares ** 2))
    else:
        hhi = 0.0
    return {
        "catalog_size": float(len(tracks)),
        "avg_engagement": float(np.mean(engagements)),
        "dormant_fraction": dormant / len(tracks),
        "concentration": hhi,
    }


def audience_concentration(listener_counts_by_region: dict[str, int]) -> float:
    total = sum(listener_counts_by_region.values())
    if total <= 0:
        return 0.0
    shares = np.array([c / total for c in listener_counts_by_region.values()])
    return float(np.sum(shares ** 2))


def listener_retention_curve(
    weekly_listeners: list[int],
) -> list[float]:
    if not weekly_listeners or weekly_listeners[0] <= 0:
        return []
    base = weekly_listeners[0]
    return [w / base for w in weekly_listeners]


def breakeven_analysis(
    production_cost: float,
    marketing_cost: float,
    streams_by_platform: dict[Platform, int],
    monthly_growth_rate: float = 0.05,
) -> dict[str, float]:
    total_cost = production_cost + marketing_cost
    current_monthly_rev = estimated_revenue(streams_by_platform)
    if current_monthly_rev <= 0:
        return {"months_to_breakeven": float("inf"), "monthly_revenue": 0.0, "total_cost": total_cost}
    cumulative = 0.0
    month = 0
    monthly_rev = current_monthly_rev
    while cumulative < total_cost and month < 120:
        cumulative += monthly_rev
        monthly_rev *= (1.0 + monthly_growth_rate)
        month += 1
    return {
        "months_to_breakeven": float(month) if cumulative >= total_cost else float("inf"),
        "monthly_revenue": current_monthly_rev,
        "total_cost": total_cost,
    }
