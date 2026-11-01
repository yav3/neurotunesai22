"""Royalty analytics: payment models, songwriter splits, catalog aging, recoupment.

Fills gaps not covered by the existing royalty_engine / revenue_models /
contract_modeling / royalty_calculation modules.  Passman (2019) statutory
rates, Mulligan (2015) user-centric analysis, Ordanini & Nunes (2016) for
catalog decay.  Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RoyaltyRateCard:
    platform: str
    per_stream_rate: float
    pro_rata: bool = True
    currency: str = "USD"


@dataclass
class StreamingRevenueResult:
    streams: int
    rate: float
    gross_revenue: float
    distributor_share: float
    artist_net: float


@dataclass
class RecoupmentState:
    advance: float
    earned: float
    recouped: bool
    balance: float
    periods_to_recoup: Optional[int]


@dataclass
class UserCentricComparison:
    pro_rata_payout: float
    user_centric_payout: float
    delta: float
    delta_pct: float


@dataclass
class CatalogValuationResult:
    npv: float
    monthly_cashflows: np.ndarray
    discount_rate: float
    decay_rate: float
    horizon_months: int


# ---------------------------------------------------------------------------
# Platform rate cards (approximate, mid-2025 ranges)
# ---------------------------------------------------------------------------

RM_PLATFORM_RATES: Dict[str, RoyaltyRateCard] = {
    "spotify": RoyaltyRateCard("Spotify", 0.004, True),
    "apple_music": RoyaltyRateCard("Apple Music", 0.008, True),
    "amazon_music": RoyaltyRateCard("Amazon Music", 0.005, True),
    "youtube_music": RoyaltyRateCard("YouTube Music", 0.003, True),
    "tidal": RoyaltyRateCard("Tidal", 0.012, True),
    "deezer": RoyaltyRateCard("Deezer", 0.004, True),
    "pandora": RoyaltyRateCard("Pandora", 0.003, True),
}

US_MECHANICAL_RATE = 0.12  # USD per copy/download, CRB 2024-2027


# ---------------------------------------------------------------------------
# Streaming revenue
# ---------------------------------------------------------------------------

def rm_calculate_streaming_revenue(
    streams: int,
    rate_card: RoyaltyRateCard,
    distributor_pct: float = 0.20,
) -> StreamingRevenueResult:
    gross = streams * rate_card.per_stream_rate
    dist_share = gross * distributor_pct
    return StreamingRevenueResult(
        streams=streams,
        rate=rate_card.per_stream_rate,
        gross_revenue=gross,
        distributor_share=dist_share,
        artist_net=gross - dist_share,
    )


# ---------------------------------------------------------------------------
# Mechanical & performance royalties
# ---------------------------------------------------------------------------

def rm_mechanical_royalty(copies: int, rate: float = US_MECHANICAL_RATE) -> float:
    return copies * rate


def rm_performance_royalty_split(
    total_royalty: float,
    writer_pct: float = 0.50,
) -> Tuple[float, float]:
    writer = total_royalty * writer_pct
    publisher = total_royalty - writer
    return writer, publisher


# ---------------------------------------------------------------------------
# Sync licensing
# ---------------------------------------------------------------------------

_SYNC_BASE: Dict[str, float] = {
    "major_film": 50_000.0,
    "indie_film": 8_000.0,
    "network_tv": 25_000.0,
    "cable_tv": 10_000.0,
    "streaming_series": 15_000.0,
    "commercial_national": 100_000.0,
    "commercial_regional": 20_000.0,
    "video_game_aaa": 40_000.0,
    "video_game_indie": 5_000.0,
    "trailer": 75_000.0,
}

_TIER_MULT = {"unknown": 0.3, "indie": 0.5, "mid": 1.0, "major": 2.0, "superstar": 5.0}


def rm_sync_license_estimate(
    placement: str,
    artist_tier: str = "mid",
) -> float:
    base = _SYNC_BASE.get(placement, 10_000.0)
    mult = _TIER_MULT.get(artist_tier, 1.0)
    return base * mult


# ---------------------------------------------------------------------------
# Pro-rata vs user-centric payment model comparison
# ---------------------------------------------------------------------------

def pro_rata_vs_user_centric(
    artist_streams: int,
    total_pool_streams: int,
    pool_revenue: float,
    subscriber_streams_for_artist: int,
    subscriber_fee: float = 9.99,
    platform_cut: float = 0.30,
) -> UserCentricComparison:
    pro_rata = (artist_streams / max(total_pool_streams, 1)) * pool_revenue
    subscriber_royalty_pool = subscriber_fee * (1.0 - platform_cut)
    uc = (subscriber_streams_for_artist / max(subscriber_streams_for_artist + 1, 1)) * subscriber_royalty_pool
    delta = uc - pro_rata
    delta_pct = delta / max(abs(pro_rata), 1e-10) * 100.0
    return UserCentricComparison(pro_rata, uc, delta, delta_pct)


# ---------------------------------------------------------------------------
# Breakeven & projection
# ---------------------------------------------------------------------------

def rm_breakeven_streams(
    production_cost: float,
    per_stream_rate: float = 0.004,
    distributor_pct: float = 0.20,
) -> int:
    net_per_stream = per_stream_rate * (1.0 - distributor_pct)
    if net_per_stream <= 0:
        return -1
    return int(np.ceil(production_cost / net_per_stream))


def royalty_projection(
    initial_monthly_streams: int,
    per_stream_rate: float = 0.004,
    monthly_growth_rate: float = 0.0,
    months: int = 12,
    distributor_pct: float = 0.20,
) -> np.ndarray:
    t = np.arange(months, dtype=float)
    streams = initial_monthly_streams * ((1.0 + monthly_growth_rate) ** t)
    net = streams * per_stream_rate * (1.0 - distributor_pct)
    return net


# ---------------------------------------------------------------------------
# Catalog valuation with aging decay (Ordanini & Nunes 2016 model)
# ---------------------------------------------------------------------------

def rm_catalog_valuation(
    monthly_revenue: float,
    discount_rate_annual: float = 0.10,
    decay_rate_monthly: float = 0.005,
    horizon_months: int = 120,
) -> CatalogValuationResult:
    r_monthly = (1.0 + discount_rate_annual) ** (1.0 / 12.0) - 1.0
    t = np.arange(1, horizon_months + 1, dtype=float)
    cashflows = monthly_revenue * np.exp(-decay_rate_monthly * t)
    discount_factors = (1.0 + r_monthly) ** (-t)
    npv = float(np.sum(cashflows * discount_factors))
    return CatalogValuationResult(
        npv=npv,
        monthly_cashflows=cashflows,
        discount_rate=discount_rate_annual,
        decay_rate=decay_rate_monthly,
        horizon_months=horizon_months,
    )


# ---------------------------------------------------------------------------
# Songwriter split calculator
# ---------------------------------------------------------------------------

def songwriter_split_calculator(
    total_royalty: float,
    writer_shares: Sequence[float],
) -> np.ndarray:
    shares = np.asarray(writer_shares, dtype=float)
    total = shares.sum()
    if total <= 0:
        return np.zeros(len(shares))
    normalised = shares / total
    return normalised * total_royalty


def songwriter_fairness_index(shares: Sequence[float]) -> float:
    s = np.asarray(shares, dtype=float)
    n = len(s)
    if n <= 1:
        return 1.0
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    equal = 1.0 / n
    return 1.0 - float(np.sqrt(np.mean((p - equal) ** 2)) / equal)


# ---------------------------------------------------------------------------
# Recoupment tracker
# ---------------------------------------------------------------------------

def recoupment_tracker(
    advance: float,
    monthly_earnings: Sequence[float],
    royalty_rate: float = 1.0,
) -> RecoupmentState:
    earned = 0.0
    periods = None
    for i, e in enumerate(monthly_earnings):
        earned += e * royalty_rate
        if earned >= advance and periods is None:
            periods = i + 1
    balance = advance - earned
    return RecoupmentState(
        advance=advance,
        earned=earned,
        recouped=earned >= advance,
        balance=max(balance, 0.0),
        periods_to_recoup=periods,
    )


# ---------------------------------------------------------------------------
# Market concentration (HHI)
# ---------------------------------------------------------------------------

def rm_market_share_hhi(shares: Sequence[float]) -> float:
    s = np.asarray(shares, dtype=float)
    total = s.sum()
    if total <= 0:
        return 0.0
    pct = (s / total) * 100.0
    return float(np.sum(pct ** 2))


def streaming_market_hhi() -> float:
    approx_shares = np.array([31.0, 15.0, 13.0, 8.0, 2.0, 2.0, 2.0, 27.0])
    return float(np.sum(approx_shares ** 2))


# ---------------------------------------------------------------------------
# Cross-platform blended rate
# ---------------------------------------------------------------------------

def blended_platform_rate(
    stream_counts: Dict[str, int],
    rates: Optional[Dict[str, RoyaltyRateCard]] = None,
) -> float:
    if rates is None:
        rates = RM_PLATFORM_RATES
    total_streams = 0
    total_rev = 0.0
    for platform, count in stream_counts.items():
        card = rates.get(platform)
        if card is None:
            continue
        total_streams += count
        total_rev += count * card.per_stream_rate
    if total_streams == 0:
        return 0.0
    return total_rev / total_streams


# ---------------------------------------------------------------------------
# Advance sizing heuristic (Passman rule of thumb)
# ---------------------------------------------------------------------------

def advance_sizing(
    projected_annual_royalty: float,
    deal_length_years: int = 3,
    advance_multiple: float = 0.75,
) -> float:
    return projected_annual_royalty * deal_length_years * advance_multiple
