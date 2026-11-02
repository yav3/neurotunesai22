"""Music industry revenue modeling.

Models streaming royalties, mechanical licenses, sync placements,
touring revenue, catalog valuation (DCF), and multi-stream
revenue forecasting.  Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class RevenueStream(Enum):
    STREAMING = "streaming"
    SYNC_LICENSE = "sync_license"
    MECHANICAL = "mechanical"
    PERFORMANCE = "performance"
    MERCH = "merch"
    TOURING = "touring"
    PUBLISHING = "publishing"


class StreamingTier(Enum):
    FREE_AD_SUPPORTED = "free_ad_supported"
    PREMIUM_INDIVIDUAL = "premium_individual"
    PREMIUM_FAMILY = "premium_family"
    HI_FI = "hi_fi"


TIER_RATES: Dict[StreamingTier, float] = {
    StreamingTier.FREE_AD_SUPPORTED: 0.002,
    StreamingTier.PREMIUM_INDIVIDUAL: 0.004,
    StreamingTier.PREMIUM_FAMILY: 0.003,
    StreamingTier.HI_FI: 0.006,
}


@dataclass
class RoyaltyStructure:
    stream_type: RevenueStream
    rate_per_unit: float
    minimum_guarantee: float = 0.0
    advance: float = 0.0
    recoup_rate: float = 1.0


@dataclass
class ArtistDeal:
    royalty_pct: float = 0.20
    advance_amount: float = 0.0
    recoup_balance: float = 0.0
    deal_term_years: int = 5


def streaming_revenue(
    streams: int,
    per_stream_rate: float = 0.004,
    royalty_pct: float = 0.20,
) -> float:
    gross = streams * per_stream_rate
    return gross * royalty_pct


def mechanical_royalty(units: int, rate: float = 0.091, songs_per_unit: int = 10) -> float:
    return units * rate * songs_per_unit


SYNC_BASE_VALUES: Dict[str, float] = {
    "major_film": 50000.0,
    "indie_film": 5000.0,
    "tv_series": 15000.0,
    "commercial_national": 75000.0,
    "commercial_regional": 10000.0,
    "video_game": 20000.0,
    "trailer": 40000.0,
    "web_content": 2000.0,
}


def sync_license_value(
    placement_type: str,
    territory: str = "us",
    duration_years: float = 1.0,
) -> float:
    base = SYNC_BASE_VALUES.get(placement_type, 5000.0)
    territory_mult = {"worldwide": 2.0, "us": 1.0, "eu": 0.9, "regional": 0.4}.get(territory, 0.5)
    duration_mult = min(duration_years / 1.0, 3.0)
    return base * territory_mult * duration_mult


def touring_revenue(
    capacity: int,
    ticket_price: float,
    n_shows: int,
    venue_cut: float = 0.15,
    fill_rate: float = 0.85,
) -> float:
    gross = capacity * fill_rate * ticket_price * n_shows
    return gross * (1.0 - venue_cut)


def recoup_schedule(
    advance: float,
    royalty_per_period: float,
    periods: int,
) -> np.ndarray:
    balance = advance
    schedule = np.zeros(periods)
    for i in range(periods):
        if balance <= 0:
            schedule[i] = 0.0
        else:
            payment = min(royalty_per_period, balance)
            balance -= payment
            schedule[i] = payment
    return np.cumsum(schedule)


def revenue_forecast(
    streams: Dict[RevenueStream, float],
    growth_rates: Dict[RevenueStream, float],
    periods: int = 12,
) -> np.ndarray:
    stream_keys = sorted(streams.keys(), key=lambda s: s.value)
    matrix = np.zeros((periods, len(stream_keys)))
    for j, key in enumerate(stream_keys):
        base = streams[key]
        g = growth_rates.get(key, 0.0)
        for i in range(periods):
            matrix[i, j] = base * (1.0 + g) ** i
    return matrix


def break_even_streams(
    production_cost: float,
    marketing_cost: float,
    per_stream_rate: float = 0.004,
    royalty_pct: float = 0.20,
) -> int:
    total_cost = production_cost + marketing_cost
    net_per_stream = per_stream_rate * royalty_pct
    if net_per_stream <= 0:
        return -1
    return math.ceil(total_cost / net_per_stream)


def catalog_valuation(
    annual_revenue: float,
    growth_rate: float = 0.02,
    discount_rate: float = 0.09,
    terminal_multiple: float = 12.0,
    projection_years: int = 10,
) -> float:
    dcf = 0.0
    for y in range(1, projection_years + 1):
        cf = annual_revenue * (1 + growth_rate) ** y
        dcf += cf / (1 + discount_rate) ** y
    terminal = annual_revenue * (1 + growth_rate) ** projection_years * terminal_multiple
    dcf += terminal / (1 + discount_rate) ** projection_years
    return dcf


def market_share_index(artist_streams: int, total_market_streams: int) -> float:
    if total_market_streams <= 0:
        return 0.0
    return artist_streams / total_market_streams


def royalty_waterfall(
    gross_revenue: float,
    distributor_pct: float = 0.15,
    label_pct: float = 0.50,
    publisher_pct: float = 0.10,
) -> Dict[str, float]:
    distributor = gross_revenue * distributor_pct
    remaining = gross_revenue - distributor
    label = remaining * label_pct
    publisher = remaining * publisher_pct
    artist = remaining - label - publisher
    return {
        "distributor": distributor,
        "label": label,
        "publisher": publisher,
        "artist": max(artist, 0.0),
    }
