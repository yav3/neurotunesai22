"""Royalty calculation: mechanical, performance, sync, streaming, blanket licence models.

Models music royalty flows: per-stream rates, mechanical royalty computation,
performance royalty splits, sync licence fee estimation, blanket licence
allocation, and revenue waterfall analysis.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RoyaltyType(str, Enum):
    MECHANICAL = "mechanical"
    PERFORMANCE = "performance"
    SYNC = "sync"
    STREAMING = "streaming"
    PRINT = "print"
    GRAND_RIGHTS = "grand_rights"


class LicenceType(str, Enum):
    BLANKET = "blanket"
    PER_PROGRAMME = "per_programme"
    DIRECT = "direct"
    STATUTORY = "statutory"
    VOLUNTARY = "voluntary"


@dataclass
class StreamingRate:
    platform: str
    per_stream_low: float   # USD
    per_stream_high: float
    pro_rata: bool = True

    @property
    def mid_rate(self) -> float:
        return (self.per_stream_low + self.per_stream_high) / 2.0


PLATFORM_RATES_2024 = {
    "spotify":       StreamingRate("Spotify",       0.003, 0.005),
    "apple_music":   StreamingRate("Apple Music",   0.006, 0.010),
    "amazon_music":  StreamingRate("Amazon Music",  0.003, 0.005),
    "youtube_music": StreamingRate("YouTube Music", 0.002, 0.005),
    "tidal":         StreamingRate("Tidal",         0.008, 0.013),
    "deezer":        StreamingRate("Deezer",        0.003, 0.005),
}


@dataclass
class RoyaltySplit:
    songwriter_pct: float = 0.50
    publisher_pct: float = 0.50
    producer_pct: float = 0.0
    label_pct: float = 0.0

    def validate(self) -> bool:
        total = (self.songwriter_pct + self.publisher_pct +
                 self.producer_pct + self.label_pct)
        return abs(total - 1.0) < 0.01


@dataclass
class RevenueBreakdown:
    gross: float
    distributor_fee: float
    label_share: float
    publisher_share: float
    songwriter_share: float
    producer_share: float
    net_artist: float


def streaming_revenue(streams: int, platform: str = "spotify") -> Tuple[float, float]:
    """Estimate revenue range from stream count."""
    rate = PLATFORM_RATES_2024.get(platform)
    if rate is None:
        return (0.0, 0.0)
    return (streams * rate.per_stream_low, streams * rate.per_stream_high)


def mechanical_royalty_us(units: int, statutory_rate: float = 0.0917,
                          song_minutes: float = 5.0) -> float:
    """US statutory mechanical royalty (per copy/download).

    §115 rate: $0.0917 per song ≤5min, or $0.0175/min for songs >5min (2024).
    """
    if song_minutes <= 5.0:
        return units * statutory_rate
    per_minute_rate = 0.0175
    return units * per_minute_rate * song_minutes


def performance_royalty(
    total_licence_pool: float,
    track_plays: int,
    total_plays: int,
    split: RoyaltySplit,
) -> Dict[str, float]:
    """Blanket-licence performance royalty allocation (pro-rata)."""
    if total_plays <= 0:
        share = 0.0
    else:
        share = total_licence_pool * (track_plays / total_plays)
    return {
        "gross_track_share": share,
        "songwriter": share * split.songwriter_pct,
        "publisher": share * split.publisher_pct,
    }


def sync_fee_estimate(
    track_popularity_percentile: float,
    placement_type: str = "film",
    duration_seconds: float = 30.0,
    territory: str = "worldwide",
) -> Tuple[float, float]:
    """Rough sync licence fee range.

    Placement types: film, tv_series, commercial, trailer, video_game, web.
    """
    base_ranges = {
        "film":       (5_000, 500_000),
        "tv_series":  (2_000, 100_000),
        "commercial": (10_000, 1_000_000),
        "trailer":    (5_000, 250_000),
        "video_game": (3_000, 150_000),
        "web":        (500, 25_000),
    }
    low, high = base_ranges.get(placement_type, (1_000, 50_000))
    popularity_mult = 0.2 + 0.8 * track_popularity_percentile
    duration_mult = min(duration_seconds / 30.0, 3.0)
    territory_mult = 1.0 if territory == "worldwide" else 0.5
    return (low * popularity_mult * duration_mult * territory_mult,
            high * popularity_mult * duration_mult * territory_mult)


def revenue_waterfall(
    gross: float,
    distributor_pct: float = 0.15,
    label_pct: float = 0.50,
    publisher_pct: float = 0.15,
    songwriter_pct: float = 0.10,
    producer_pct: float = 0.05,
) -> RevenueBreakdown:
    """Waterfall revenue split from gross."""
    dist_fee = gross * distributor_pct
    after_dist = gross - dist_fee
    label_share = after_dist * label_pct
    pub_share = after_dist * publisher_pct
    sw_share = after_dist * songwriter_pct
    prod_share = after_dist * producer_pct
    net_artist = after_dist - label_share - pub_share - sw_share - prod_share
    return RevenueBreakdown(
        gross=gross, distributor_fee=dist_fee,
        label_share=label_share, publisher_share=pub_share,
        songwriter_share=sw_share, producer_share=prod_share,
        net_artist=max(net_artist, 0.0),
    )


def break_even_streams(
    production_cost: float,
    marketing_cost: float = 0.0,
    platform: str = "spotify",
    artist_share_pct: float = 0.20,
) -> int:
    """Minimum streams to recoup costs at artist's effective per-stream rate."""
    rate = PLATFORM_RATES_2024.get(platform)
    if rate is None or rate.mid_rate <= 0:
        return 0
    effective = rate.mid_rate * artist_share_pct
    if effective <= 0:
        return 0
    return int(np.ceil((production_cost + marketing_cost) / effective))


def blanket_licence_allocation(
    tracks: List[Dict[str, float]],
    total_pool: float,
) -> np.ndarray:
    """Allocate blanket licence pool across tracks by weighted plays.

    Each track dict must have 'plays' and optionally 'weight' (default 1.0).
    """
    weighted = np.array([t.get("plays", 0) * t.get("weight", 1.0) for t in tracks])
    total = weighted.sum()
    if total <= 0:
        return np.zeros(len(tracks))
    return total_pool * (weighted / total)


def compounding_catalogue_value(
    annual_revenue: float,
    growth_rate: float = 0.03,
    discount_rate: float = 0.08,
    years: int = 10,
) -> float:
    """DCF valuation of a music catalogue."""
    total = 0.0
    for y in range(1, years + 1):
        cf = annual_revenue * (1 + growth_rate) ** y
        total += cf / (1 + discount_rate) ** y
    terminal = (annual_revenue * (1 + growth_rate) ** years *
                (1 + growth_rate) / (discount_rate - growth_rate))
    total += terminal / (1 + discount_rate) ** years
    return total
