"""Royalty calculation engine for music industry analytics.

Mechanical royalties, performance royalties, streaming payouts,
sync fees, and split allocation across rights holders.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RightsHolder:
    """A participant in royalty splits."""
    name: str = ""
    role: str = "writer"
    share_pct: float = 0.0
    pro: str = ""


@dataclass
class RoyaltyConfig:
    """Platform-specific royalty parameters."""
    per_stream_rate: float = 0.004
    mechanical_rate_per_copy: float = 0.091
    performance_share_writer: float = 0.5
    performance_share_publisher: float = 0.5
    sync_base_fee: float = 0.0
    master_share: float = 0.5
    publishing_share: float = 0.5


def validate_splits(holders: list[RightsHolder]) -> bool:
    """Check that splits sum to 100% (within tolerance)."""
    total = sum(h.share_pct for h in holders)
    return abs(total - 100.0) < 0.01


def allocate_royalties(
    total_amount: float, holders: list[RightsHolder]
) -> dict[str, float]:
    """Distribute a royalty pool according to share percentages."""
    result = {}
    for h in holders:
        result[h.name] = total_amount * h.share_pct / 100.0
    return result


# ── Streaming royalties ──────────────────────────────────────────────

def streaming_payout(
    streams: int,
    per_stream_rate: float = 0.004,
    platform_cut: float = 0.30,
) -> float:
    """Net payout from streaming after platform deduction."""
    gross = streams * per_stream_rate
    return gross * (1.0 - platform_cut)


def streaming_payout_curve(
    monthly_streams: np.ndarray,
    per_stream_rate: float = 0.004,
    platform_cut: float = 0.30,
) -> np.ndarray:
    """Monthly payout time series."""
    s = np.asarray(monthly_streams, dtype=float)
    return s * per_stream_rate * (1.0 - platform_cut)


def pro_rata_pool_share(
    track_streams: int,
    total_platform_streams: int,
    pool_revenue: float,
) -> float:
    """Pro-rata share of a platform's royalty pool."""
    if total_platform_streams <= 0:
        return 0.0
    return pool_revenue * (track_streams / total_platform_streams)


# ── Mechanical royalties ─────────────────────────────────────────────

def mechanical_royalty(
    copies: int,
    rate_per_copy: float = 0.091,
    duration_minutes: float = 5.0,
    long_song_rate_per_min: float = 0.0175,
) -> float:
    """US statutory mechanical royalty.

    Songs > 5 minutes use the per-minute rate.
    """
    if duration_minutes <= 5.0:
        return copies * rate_per_copy
    return copies * duration_minutes * long_song_rate_per_min


# ── Performance royalties ────────────────────────────────────────────

def performance_royalty_estimate(
    radio_spins: int,
    rate_per_spin: float = 0.10,
    tv_plays: int = 0,
    rate_per_tv: float = 3.0,
    live_performances: int = 0,
    rate_per_live: float = 0.50,
) -> float:
    """Estimate total performance royalties across media."""
    return (radio_spins * rate_per_spin
            + tv_plays * rate_per_tv
            + live_performances * rate_per_live)


# ── Sync licensing ───────────────────────────────────────────────────

def sync_fee_estimate(
    placement_type: str = "tv_series",
    audience_size: int = 0,
    exclusivity: bool = False,
    duration_sec: float = 30.0,
) -> float:
    """Heuristic sync placement fee estimate.

    placement_type: 'tv_series', 'film', 'commercial', 'video_game', 'trailer'
    """
    base_rates = {
        "tv_series": 2000.0,
        "film": 10000.0,
        "commercial": 25000.0,
        "video_game": 5000.0,
        "trailer": 15000.0,
    }
    base = base_rates.get(placement_type, 2000.0)
    audience_mult = 1.0 + np.log1p(audience_size / 1_000_000) * 0.5
    excl_mult = 2.0 if exclusivity else 1.0
    dur_mult = min(duration_sec / 30.0, 3.0)
    return base * audience_mult * excl_mult * max(dur_mult, 0.5)


# ── Revenue projection ───────────────────────────────────────────────

@dataclass
class RevenueProjection:
    """Projected revenue across sources."""
    streaming: float = 0.0
    mechanical: float = 0.0
    performance: float = 0.0
    sync: float = 0.0
    total: float = 0.0


def project_revenue(
    monthly_streams: np.ndarray,
    copies_sold: int = 0,
    radio_spins: int = 0,
    sync_placements: int = 0,
    config: RoyaltyConfig | None = None,
) -> RevenueProjection:
    """Aggregate revenue projection across all sources."""
    if config is None:
        config = RoyaltyConfig()

    stream_rev = float(streaming_payout_curve(monthly_streams, config.per_stream_rate).sum())
    mech_rev = mechanical_royalty(copies_sold, config.mechanical_rate_per_copy)
    perf_rev = performance_royalty_estimate(radio_spins)
    sync_rev = sync_placements * config.sync_base_fee

    return RevenueProjection(
        streaming=stream_rev,
        mechanical=mech_rev,
        performance=perf_rev,
        sync=sync_rev,
        total=stream_rev + mech_rev + perf_rev + sync_rev,
    )


# ── Breakeven analysis ───────────────────────────────────────────────

def streams_to_breakeven(
    cost: float,
    per_stream_rate: float = 0.004,
    platform_cut: float = 0.30,
    artist_share_pct: float = 20.0,
) -> int:
    """Number of streams needed to recoup a given cost."""
    net_per_stream = per_stream_rate * (1.0 - platform_cut) * (artist_share_pct / 100.0)
    if net_per_stream <= 0:
        return -1
    return int(np.ceil(cost / net_per_stream))


def recoupment_schedule(
    monthly_streams: np.ndarray,
    advance: float,
    per_stream_rate: float = 0.004,
    platform_cut: float = 0.30,
) -> int:
    """Month index at which cumulative royalties exceed advance. -1 if never."""
    payouts = streaming_payout_curve(monthly_streams, per_stream_rate, platform_cut)
    cumsum = np.cumsum(payouts)
    recouped = np.where(cumsum >= advance)[0]
    if len(recouped) == 0:
        return -1
    return int(recouped[0])
