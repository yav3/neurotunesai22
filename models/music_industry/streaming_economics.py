"""Streaming platform economics: per-stream rates, revenue modeling, catalog valuation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

__all__ = [
    "Platform", "PER_STREAM_RATES", "MARKET_SHARE",
    "StreamingRevenue", "ArtistTier", "SPLIT_RATIOS",
    "estimate_revenue", "cross_platform_revenue", "breakeven_streams",
    "monthly_listener_value", "catalog_value", "platform_comparison",
]


class Platform(Enum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    AMAZON_MUSIC = "amazon_music"
    TIDAL = "tidal"
    DEEZER = "deezer"


PER_STREAM_RATES: Dict[Platform, float] = {
    Platform.SPOTIFY: 0.004,
    Platform.APPLE_MUSIC: 0.008,
    Platform.YOUTUBE_MUSIC: 0.002,
    Platform.AMAZON_MUSIC: 0.004,
    Platform.TIDAL: 0.012,
    Platform.DEEZER: 0.004,
}

MARKET_SHARE: Dict[Platform, float] = {
    Platform.SPOTIFY: 0.31,
    Platform.APPLE_MUSIC: 0.15,
    Platform.YOUTUBE_MUSIC: 0.10,
    Platform.AMAZON_MUSIC: 0.13,
    Platform.TIDAL: 0.01,
    Platform.DEEZER: 0.02,
}


class ArtistTier(Enum):
    INDIE = "indie"
    MID = "mid"
    MAJOR_LABEL = "major_label"


SPLIT_RATIOS: Dict[ArtistTier, tuple] = {
    ArtistTier.INDIE: (0.80, 0.00, 0.20),
    ArtistTier.MID: (0.50, 0.30, 0.20),
    ArtistTier.MAJOR_LABEL: (0.18, 0.62, 0.20),
}


@dataclass
class StreamingRevenue:
    platform: Platform
    streams: int
    gross_revenue: float
    artist_share: float
    label_share: float
    distributor_share: float


def estimate_revenue(
    platform: Platform,
    streams: int,
    tier: ArtistTier = ArtistTier.INDIE,
) -> StreamingRevenue:
    rate = PER_STREAM_RATES.get(platform, 0.004)
    gross = streams * rate
    artist_pct, label_pct, dist_pct = SPLIT_RATIOS[tier]
    return StreamingRevenue(
        platform=platform,
        streams=streams,
        gross_revenue=round(gross, 2),
        artist_share=round(gross * artist_pct, 2),
        label_share=round(gross * label_pct, 2),
        distributor_share=round(gross * dist_pct, 2),
    )


def cross_platform_revenue(
    stream_counts: Dict[Platform, int],
    tier: ArtistTier = ArtistTier.INDIE,
) -> List[StreamingRevenue]:
    return [estimate_revenue(p, n, tier) for p, n in stream_counts.items()]


def breakeven_streams(
    production_cost: float,
    platform: Platform = Platform.SPOTIFY,
    tier: ArtistTier = ArtistTier.INDIE,
) -> int:
    rate = PER_STREAM_RATES.get(platform, 0.004)
    artist_pct = SPLIT_RATIOS[tier][0]
    per_stream_artist = rate * artist_pct
    if per_stream_artist <= 0:
        return -1
    import math
    return math.ceil(production_cost / per_stream_artist)


def monthly_listener_value(
    streams_per_listener: float,
    platform: Platform = Platform.SPOTIFY,
) -> float:
    rate = PER_STREAM_RATES.get(platform, 0.004)
    return streams_per_listener * rate


def catalog_value(
    total_streams: int,
    growth_rate: float = 0.0,
    discount_rate: float = 0.10,
    years: int = 10,
) -> float:
    if discount_rate <= 0:
        return float("inf")

    avg_rate = sum(PER_STREAM_RATES.values()) / len(PER_STREAM_RATES)
    annual_revenue = total_streams * avg_rate

    npv = 0.0
    for y in range(1, years + 1):
        projected = annual_revenue * ((1.0 + growth_rate) ** y)
        npv += projected / ((1.0 + discount_rate) ** y)
    return round(npv, 2)


def platform_comparison(
    total_streams: int,
    tier: ArtistTier = ArtistTier.INDIE,
) -> Dict[str, StreamingRevenue]:
    result = {}
    for platform in Platform:
        share = MARKET_SHARE.get(platform, 0.05)
        platform_streams = int(total_streams * share)
        result[platform.value] = estimate_revenue(platform, platform_streams, tier)
    return result
