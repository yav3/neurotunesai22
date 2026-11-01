"""Digital distribution channel modeling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DistributionChannel(Enum):
    STREAMING = "streaming"
    DOWNLOAD = "download"
    SYNC = "sync"
    BROADCAST = "broadcast"
    LIVE = "live"
    PHYSICAL = "physical"


@dataclass
class RoyaltyModel:
    per_stream_rate: float = 0.004
    per_download_rate: float = 0.70
    sync_flat_fee: float = 500.0


def estimate_royalties(
    streams: int = 0,
    downloads: int = 0,
    sync_placements: int = 0,
    model: RoyaltyModel | None = None,
) -> float:
    if model is None:
        model = RoyaltyModel()
    return (
        streams * model.per_stream_rate
        + downloads * model.per_download_rate
        + sync_placements * model.sync_flat_fee
    )


def channel_revenue_split(
    gross: float, splits: dict[str, float]
) -> dict[str, float]:
    total_pct = sum(splits.values())
    if total_pct <= 0:
        return {k: 0.0 for k in splits}
    return {k: gross * v / total_pct for k, v in splits.items()}


def breakeven_streams(production_cost: float, per_stream_rate: float = 0.004) -> int:
    if per_stream_rate <= 0:
        return -1
    return int(production_cost / per_stream_rate) + 1
