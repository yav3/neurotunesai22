"""Royalty calculation models for mechanical, performance, and sync licenses."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RoyaltyType(Enum):
    MECHANICAL = "mechanical"
    PERFORMANCE = "performance"
    SYNC = "sync"
    PRINT = "print"
    DIGITAL = "digital"


# US statutory mechanical rate (2024): $0.12 per copy for songs <= 5 min
MECHANICAL_RATE_PER_COPY = 0.12
MECHANICAL_RATE_PER_MINUTE = 0.0231  # for songs > 5 min


def mechanical_royalty(copies: int, duration_seconds: float) -> float:
    """US statutory mechanical royalty per 17 USC §115."""
    duration_min = duration_seconds / 60.0
    if duration_min <= 5.0:
        return copies * MECHANICAL_RATE_PER_COPY
    return copies * duration_min * MECHANICAL_RATE_PER_MINUTE


# Performance royalty split: typically writer 50% / publisher 50%
def performance_royalty_split(total_revenue: float,
                              writer_share: float = 0.5,
                              publisher_share: float = 0.5) -> dict[str, float]:
    return {
        "writer": total_revenue * writer_share,
        "publisher": total_revenue * publisher_share,
    }


@dataclass
class SyncLicenseTerms:
    """Synchronization license fee structure."""
    base_fee: float
    territory: str = "worldwide"
    duration_years: float = 1.0
    media_types: list[str] | None = None
    exclusivity: bool = False

    def adjusted_fee(self) -> float:
        multiplier = 1.0
        if self.exclusivity:
            multiplier *= 2.0
        if self.territory == "worldwide":
            multiplier *= 1.5
        elif self.territory == "regional":
            multiplier *= 1.0
        else:
            multiplier *= 0.7
        multiplier *= min(self.duration_years, 10.0) / 1.0
        return self.base_fee * multiplier


# Streaming royalty rates (approximate per-stream, varies by platform)
STREAMING_RATES: dict[str, float] = {
    "spotify": 0.004,
    "apple_music": 0.008,
    "amazon_music": 0.004,
    "youtube_music": 0.002,
    "tidal": 0.012,
    "deezer": 0.004,
}


def streaming_royalty(streams: int, platform: str = "spotify") -> float:
    rate = STREAMING_RATES.get(platform, 0.004)
    return streams * rate


@dataclass
class RoyaltyStatement:
    mechanical: float = 0.0
    performance: float = 0.0
    sync: float = 0.0
    streaming: float = 0.0
    print_: float = 0.0

    @property
    def total(self) -> float:
        return self.mechanical + self.performance + self.sync + self.streaming + self.print_


def compute_statement(copies: int = 0, duration_seconds: float = 180.0,
                      performance_revenue: float = 0.0,
                      sync_terms: SyncLicenseTerms | None = None,
                      streams: dict[str, int] | None = None,
                      print_copies: int = 0, print_rate: float = 0.10) -> RoyaltyStatement:
    mech = mechanical_royalty(copies, duration_seconds) if copies > 0 else 0.0
    perf = performance_revenue * 0.5 if performance_revenue > 0 else 0.0
    sync = sync_terms.adjusted_fee() if sync_terms else 0.0
    stream_total = 0.0
    if streams:
        for platform, count in streams.items():
            stream_total += streaming_royalty(count, platform)
    pr = print_copies * print_rate
    return RoyaltyStatement(
        mechanical=mech, performance=perf, sync=sync,
        streaming=stream_total, print_=pr,
    )
