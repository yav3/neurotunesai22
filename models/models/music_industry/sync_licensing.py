"""Sync licensing: placement valuation, cue sheets, territory pricing."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class MediaType(Enum):
    FILM = "film"
    TV_SERIES = "tv_series"
    COMMERCIAL = "commercial"
    VIDEO_GAME = "video_game"
    PODCAST = "podcast"
    TRAILER = "trailer"


class UsageType(Enum):
    BACKGROUND = "background"
    FEATURED = "featured"
    THEME = "theme"
    END_CREDITS = "end_credits"
    SOURCE = "source"


_BASE_FEES: dict[tuple[MediaType, UsageType], float] = {
    (MediaType.FILM, UsageType.FEATURED): 50000.0,
    (MediaType.FILM, UsageType.BACKGROUND): 15000.0,
    (MediaType.FILM, UsageType.END_CREDITS): 25000.0,
    (MediaType.FILM, UsageType.THEME): 75000.0,
    (MediaType.FILM, UsageType.SOURCE): 10000.0,
    (MediaType.TV_SERIES, UsageType.FEATURED): 12000.0,
    (MediaType.TV_SERIES, UsageType.BACKGROUND): 2000.0,
    (MediaType.TV_SERIES, UsageType.THEME): 30000.0,
    (MediaType.TV_SERIES, UsageType.END_CREDITS): 5000.0,
    (MediaType.TV_SERIES, UsageType.SOURCE): 3000.0,
    (MediaType.COMMERCIAL, UsageType.FEATURED): 100000.0,
    (MediaType.COMMERCIAL, UsageType.BACKGROUND): 25000.0,
    (MediaType.VIDEO_GAME, UsageType.FEATURED): 20000.0,
    (MediaType.VIDEO_GAME, UsageType.BACKGROUND): 5000.0,
    (MediaType.VIDEO_GAME, UsageType.THEME): 40000.0,
    (MediaType.PODCAST, UsageType.BACKGROUND): 500.0,
    (MediaType.PODCAST, UsageType.THEME): 2000.0,
    (MediaType.TRAILER, UsageType.FEATURED): 35000.0,
    (MediaType.TRAILER, UsageType.BACKGROUND): 10000.0,
}

TERRITORY_MULTIPLIERS: dict[str, float] = {
    "US": 1.0,
    "EU": 0.85,
    "UK": 0.75,
    "worldwide": 1.5,
    "Asia": 0.6,
    "Latin America": 0.5,
}


@dataclass(frozen=True)
class SyncPlacement:
    media_type: MediaType
    usage_type: UsageType
    duration_seconds: float
    territory: str = "US"
    fee_usd: float = 0.0


@dataclass(frozen=True)
class CueSheet:
    title: str
    placements: list[SyncPlacement]
    total_duration: float
    total_fee: float


def territory_multiplier(territory: str) -> float:
    return TERRITORY_MULTIPLIERS.get(territory, 1.0)


def estimate_sync_fee(
    media_type: MediaType,
    usage_type: UsageType,
    duration_seconds: float,
    territory: str = "US",
) -> float:
    base = _BASE_FEES.get((media_type, usage_type), 5000.0)
    dur_factor = min(2.0, max(0.5, duration_seconds / 60.0))
    terr = territory_multiplier(territory)
    return base * dur_factor * terr


def build_cue_sheet(
    title: str,
    placements: Sequence[SyncPlacement],
) -> CueSheet:
    total_dur = sum(p.duration_seconds for p in placements)
    total_fee = sum(p.fee_usd for p in placements)
    return CueSheet(
        title=title,
        placements=list(placements),
        total_duration=total_dur,
        total_fee=total_fee,
    )


def placement_value_score(placement: SyncPlacement) -> float:
    benchmark = estimate_sync_fee(
        placement.media_type, placement.usage_type,
        placement.duration_seconds, placement.territory,
    )
    if benchmark <= 0:
        return 0.0
    ratio = placement.fee_usd / benchmark
    return float(np.clip(ratio, 0.0, 2.0) / 2.0)


def catalog_sync_potential(
    placements: Sequence[SyncPlacement],
) -> dict[str, float]:
    if not placements:
        return {"mean_fee": 0.0, "total_fee": 0.0, "mean_value_score": 0.0}
    fees = [p.fee_usd for p in placements]
    scores = [placement_value_score(p) for p in placements]
    return {
        "mean_fee": float(np.mean(fees)),
        "total_fee": float(np.sum(fees)),
        "mean_value_score": float(np.mean(scores)),
    }
