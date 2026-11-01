"""Music revenue modeling: streaming economics, mechanical royalties, performance rights."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class RevenueStream(Enum):
    STREAMING = "streaming"
    MECHANICAL = "mechanical"
    PERFORMANCE = "performance"
    SYNC = "sync"
    PRINT = "print"
    MERCHANDISE = "merchandise"
    LIVE = "live"


class StreamingTier(Enum):
    FREE_AD = "free_ad_supported"
    INDIVIDUAL = "individual"
    FAMILY = "family"
    STUDENT = "student"
    HI_FI = "hi_fi"


@dataclass
class StreamingRate:
    platform: str
    tier: StreamingTier
    per_stream_usd: float
    min_duration_s: float = 30.0


PLATFORM_RATES: list[StreamingRate] = [
    StreamingRate("spotify", StreamingTier.FREE_AD, 0.003),
    StreamingRate("spotify", StreamingTier.INDIVIDUAL, 0.004),
    StreamingRate("apple_music", StreamingTier.INDIVIDUAL, 0.008),
    StreamingRate("amazon_music", StreamingTier.INDIVIDUAL, 0.004),
    StreamingRate("tidal", StreamingTier.HI_FI, 0.012),
    StreamingRate("youtube_music", StreamingTier.FREE_AD, 0.002),
    StreamingRate("youtube_music", StreamingTier.INDIVIDUAL, 0.006),
    StreamingRate("deezer", StreamingTier.INDIVIDUAL, 0.004),
]


def streaming_revenue(
    streams: int,
    platform: str = "spotify",
    tier: StreamingTier = StreamingTier.INDIVIDUAL,
) -> float:
    rate = next(
        (r for r in PLATFORM_RATES if r.platform == platform and r.tier == tier),
        None,
    )
    if rate is None:
        return 0.0
    return streams * rate.per_stream_usd


def mechanical_royalty(
    units: int,
    duration_minutes: float,
    statutory_rate_cents: float = 12.40,
    long_song_per_minute_cents: float = 2.39,
    threshold_minutes: float = 5.0,
) -> float:
    if duration_minutes <= threshold_minutes:
        return units * statutory_rate_cents / 100.0
    return units * duration_minutes * long_song_per_minute_cents / 100.0


def performance_royalty(
    plays: int,
    venue_weight: float = 1.0,
    rate_per_play: float = 0.10,
) -> float:
    return plays * rate_per_play * venue_weight


@dataclass
class RevenueProjection:
    stream: RevenueStream
    monthly_amount: float
    growth_rate: float
    months: int

    def project(self) -> list[float]:
        return [
            self.monthly_amount * (1 + self.growth_rate) ** m
            for m in range(self.months)
        ]

    def total(self) -> float:
        return sum(self.project())

    def npv(self, discount_rate: float = 0.08) -> float:
        monthly_r = (1 + discount_rate) ** (1 / 12) - 1
        return sum(
            amt / (1 + monthly_r) ** (m + 1)
            for m, amt in enumerate(self.project())
        )


def catalog_valuation(
    annual_revenue: float,
    multiple: float = 15.0,
    growth_adj: float = 1.0,
) -> float:
    return annual_revenue * multiple * growth_adj


def breakeven_streams_for_cost(
    production_cost: float,
    per_stream_rate: float = 0.004,
    artist_share: float = 0.20,
) -> int:
    if per_stream_rate * artist_share <= 0:
        return -1
    return math.ceil(production_cost / (per_stream_rate * artist_share))


def blended_rate(stream_counts: dict[str, int]) -> float:
    total_streams = sum(stream_counts.values())
    if total_streams == 0:
        return 0.0
    total_rev = sum(
        streaming_revenue(count, platform)
        for platform, count in stream_counts.items()
    )
    return total_rev / total_streams
