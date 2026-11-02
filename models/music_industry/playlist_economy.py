"""Playlist economy: playlist lift, algorithmic placement value, editorial vs algorithmic.

Models the economic impact of playlist placement on streaming revenue,
discovery multipliers, and the interplay between editorial and algorithmic curation.
Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class PlaylistType(str, Enum):
    EDITORIAL = "editorial"
    ALGORITHMIC = "algorithmic"
    USER_GENERATED = "user_generated"
    RELEASE_RADAR = "release_radar"
    DISCOVER_WEEKLY = "discover_weekly"


@dataclass(frozen=True)
class PlaylistPlacement:
    playlist_type: PlaylistType
    followers: int
    position: int
    duration_days: int


@dataclass(frozen=True)
class PlacementImpact:
    stream_lift: float
    discovery_multiplier: float
    save_rate: float
    estimated_revenue: float
    decay_half_life_days: float


EDITORIAL_LIFT_COEFFICIENT = 3.5
ALGORITHMIC_LIFT_COEFFICIENT = 1.8
POSITION_DECAY_RATE = 0.03


def position_discount(position: int) -> float:
    return np.exp(-POSITION_DECAY_RATE * max(position - 1, 0))


def stream_lift(placement: PlaylistPlacement, base_daily_streams: float) -> float:
    if placement.playlist_type == PlaylistType.EDITORIAL:
        coeff = EDITORIAL_LIFT_COEFFICIENT
    elif placement.playlist_type in (PlaylistType.RELEASE_RADAR, PlaylistType.DISCOVER_WEEKLY):
        coeff = 2.2
    elif placement.playlist_type == PlaylistType.ALGORITHMIC:
        coeff = ALGORITHMIC_LIFT_COEFFICIENT
    else:
        coeff = 1.2
    follower_factor = np.log1p(placement.followers) / np.log1p(1_000_000)
    pos_factor = position_discount(placement.position)
    daily_lift = base_daily_streams * coeff * follower_factor * pos_factor
    total = daily_lift * placement.duration_days
    return float(total)


def discovery_multiplier(placement: PlaylistPlacement) -> float:
    base = 1.0
    if placement.playlist_type == PlaylistType.EDITORIAL:
        base = 2.5
    elif placement.playlist_type == PlaylistType.DISCOVER_WEEKLY:
        base = 3.0
    elif placement.playlist_type == PlaylistType.RELEASE_RADAR:
        base = 2.0
    elif placement.playlist_type == PlaylistType.ALGORITHMIC:
        base = 1.5
    return base * position_discount(placement.position)


def save_rate_estimate(
    playlist_type: PlaylistType, skip_rate: float = 0.3,
) -> float:
    base_rates = {
        PlaylistType.EDITORIAL: 0.12,
        PlaylistType.DISCOVER_WEEKLY: 0.18,
        PlaylistType.RELEASE_RADAR: 0.15,
        PlaylistType.ALGORITHMIC: 0.08,
        PlaylistType.USER_GENERATED: 0.05,
    }
    base = base_rates.get(playlist_type, 0.05)
    return float(base * (1.0 - skip_rate))


def decay_half_life(playlist_type: PlaylistType) -> float:
    half_lives = {
        PlaylistType.EDITORIAL: 7.0,
        PlaylistType.DISCOVER_WEEKLY: 3.5,
        PlaylistType.RELEASE_RADAR: 5.0,
        PlaylistType.ALGORITHMIC: 14.0,
        PlaylistType.USER_GENERATED: 30.0,
    }
    return half_lives.get(playlist_type, 7.0)


def placement_revenue(
    placement: PlaylistPlacement,
    base_daily_streams: float,
    per_stream_rate: float = 0.004,
) -> float:
    lift = stream_lift(placement, base_daily_streams)
    return float(lift * per_stream_rate)


def analyze_placement(
    placement: PlaylistPlacement,
    base_daily_streams: float,
    per_stream_rate: float = 0.004,
    skip_rate: float = 0.3,
) -> PlacementImpact:
    return PlacementImpact(
        stream_lift=stream_lift(placement, base_daily_streams),
        discovery_multiplier=discovery_multiplier(placement),
        save_rate=save_rate_estimate(placement.playlist_type, skip_rate),
        estimated_revenue=placement_revenue(placement, base_daily_streams, per_stream_rate),
        decay_half_life_days=decay_half_life(placement.playlist_type),
    )


def playlist_portfolio_value(
    placements: list[PlaylistPlacement],
    base_daily_streams: float,
    per_stream_rate: float = 0.004,
) -> float:
    return sum(
        placement_revenue(p, base_daily_streams, per_stream_rate)
        for p in placements
    )


def editorial_vs_algorithmic_ratio(
    placements: list[PlaylistPlacement],
) -> float:
    if not placements:
        return 0.0
    editorial = sum(
        1 for p in placements
        if p.playlist_type == PlaylistType.EDITORIAL
    )
    return editorial / len(placements)


def catalog_playlist_decay(
    initial_streams: float, days: int, half_life: float,
) -> np.ndarray:
    t = np.arange(days, dtype=float)
    return initial_streams * np.power(0.5, t / max(half_life, 0.1))
