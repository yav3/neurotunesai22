"""Digital distribution pipeline: aggregator fees, territory pricing, payment splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DistributionDeal", "TerritoryPricing", "PaymentSplit",
    "ReleaseMetadata", "FormatSpec",
    "compute_aggregator_fees", "territory_adjusted_price",
    "compute_payment_splits", "validate_release_metadata",
    "format_conversion_cost", "per_stream_rate",
    "breakeven_streams", "annual_revenue_projection",
    "RecoupmentSchedule", "compute_recoupment",
]


@dataclass
class FormatSpec:
    codec: str
    bitrate_kbps: int
    sample_rate_hz: int
    channels: int

    @property
    def file_size_per_minute_mb(self) -> float:
        return self.bitrate_kbps * 60 / 8 / 1024


STANDARD_FORMATS = {
    "mp3_128": FormatSpec("mp3", 128, 44100, 2),
    "mp3_320": FormatSpec("mp3", 320, 44100, 2),
    "aac_256": FormatSpec("aac", 256, 44100, 2),
    "flac_16": FormatSpec("flac", 1411, 44100, 2),
    "flac_24": FormatSpec("flac", 2116, 96000, 2),
    "wav_16": FormatSpec("wav", 1411, 44100, 2),
    "ogg_160": FormatSpec("ogg", 160, 44100, 2),
}


def format_conversion_cost(
    duration_minutes: float,
    source_format: str,
    target_formats: List[str],
    cost_per_minute_per_format: float = 0.02,
) -> Dict[str, float]:
    costs = {}
    for fmt in target_formats:
        spec = STANDARD_FORMATS.get(fmt)
        complexity = 1.0
        if spec and spec.codec == "flac":
            complexity = 1.5
        elif spec and spec.bitrate_kbps > 256:
            complexity = 1.2
        costs[fmt] = duration_minutes * cost_per_minute_per_format * complexity
    return costs


@dataclass
class DistributionDeal:
    aggregator: str
    annual_fee: float
    per_release_fee: float
    revenue_share_pct: float
    stores_covered: int
    payout_delay_days: int

    @property
    def effective_take_rate(self) -> float:
        return self.revenue_share_pct / 100.0


AGGREGATOR_PROFILES = {
    "distrokid": DistributionDeal("DistroKid", 22.99, 0.0, 0.0, 150, 14),
    "tunecore": DistributionDeal("TuneCore", 29.99, 9.99, 0.0, 150, 30),
    "cdbaby": DistributionDeal("CD Baby", 0.0, 9.95, 9.0, 150, 60),
    "amuse": DistributionDeal("Amuse", 0.0, 0.0, 0.0, 100, 45),
    "unitedmasters": DistributionDeal("UnitedMasters", 0.0, 0.0, 10.0, 100, 30),
    "ditto": DistributionDeal("Ditto", 19.99, 0.0, 0.0, 150, 14),
    "landr": DistributionDeal("LANDR", 23.88, 0.0, 0.0, 150, 21),
}


def compute_aggregator_fees(
    aggregator: str,
    n_releases_per_year: int,
    annual_revenue: float,
) -> Dict[str, float]:
    profile = AGGREGATOR_PROFILES.get(aggregator.lower())
    if profile is None:
        return {"error": -1.0}
    fixed = profile.annual_fee + profile.per_release_fee * n_releases_per_year
    variable = annual_revenue * profile.effective_take_rate
    total = fixed + variable
    net = annual_revenue - total
    return {
        "fixed_costs": round(fixed, 2),
        "variable_costs": round(variable, 2),
        "total_costs": round(total, 2),
        "net_revenue": round(net, 2),
        "effective_rate": round(total / max(annual_revenue, 1e-9) * 100, 2),
    }


TERRITORY_MULTIPLIERS = {
    "US": 1.0, "GB": 0.95, "DE": 0.90, "FR": 0.88, "JP": 0.85,
    "AU": 0.92, "CA": 0.93, "BR": 0.45, "MX": 0.40, "IN": 0.20,
    "NG": 0.15, "KR": 0.80, "SE": 0.95, "NL": 0.90, "IT": 0.82,
    "ES": 0.78, "AR": 0.30, "CO": 0.35, "CL": 0.40, "PH": 0.18,
    "ID": 0.15, "TH": 0.22, "VN": 0.12, "EG": 0.18, "ZA": 0.30,
}


@dataclass
class TerritoryPricing:
    territory: str
    base_price: float
    adjusted_price: float
    multiplier: float
    currency: str


def territory_adjusted_price(
    base_price_usd: float,
    territories: List[str],
) -> List[TerritoryPricing]:
    results = []
    for t in territories:
        m = TERRITORY_MULTIPLIERS.get(t, 0.50)
        adj = base_price_usd * m
        results.append(TerritoryPricing(
            territory=t, base_price=base_price_usd,
            adjusted_price=round(adj, 4), multiplier=m, currency="USD",
        ))
    return results


def per_stream_rate(
    platform: str = "spotify",
    territory: str = "US",
    subscription_tier: str = "premium",
) -> float:
    base_rates = {
        "spotify": 0.004, "apple_music": 0.008, "amazon_music": 0.004,
        "youtube_music": 0.002, "tidal": 0.012, "deezer": 0.004,
        "pandora": 0.003, "soundcloud": 0.003,
    }
    tier_m = {"free": 0.3, "premium": 1.0, "family": 0.7, "student": 0.5}
    base = base_rates.get(platform, 0.004)
    t_m = TERRITORY_MULTIPLIERS.get(territory, 0.5)
    s_m = tier_m.get(subscription_tier, 1.0)
    return float(base * t_m * s_m)


@dataclass
class PaymentSplit:
    role: str
    share_pct: float
    amount: float


def compute_payment_splits(
    gross_revenue: float,
    splits: Dict[str, float],
    aggregator_take: float = 0.0,
) -> List[PaymentSplit]:
    net = gross_revenue * (1.0 - aggregator_take)
    total_shares = sum(splits.values())
    results = []
    for role, share in splits.items():
        normalized = share / max(total_shares, 1e-9)
        amount = net * normalized
        results.append(PaymentSplit(
            role=role, share_pct=round(normalized * 100, 2),
            amount=round(amount, 2),
        ))
    return results


@dataclass
class ReleaseMetadata:
    title: str
    artist: str
    album: str
    isrc: str
    upc: str
    genre: str
    release_date: str
    label: str
    territories: List[str] = field(default_factory=lambda: ["worldwide"])
    explicit: bool = False
    language: str = "en"


def validate_release_metadata(meta: ReleaseMetadata) -> List[str]:
    errors = []
    if not meta.title or len(meta.title) > 200:
        errors.append("Title must be 1-200 characters")
    if not meta.artist:
        errors.append("Artist name is required")
    if meta.isrc and len(meta.isrc) != 12:
        errors.append("ISRC must be exactly 12 characters")
    if meta.upc and len(meta.upc) not in (12, 13):
        errors.append("UPC must be 12 or 13 digits")
    if not meta.genre:
        errors.append("Genre is required")
    return errors


def breakeven_streams(
    total_costs: float,
    per_stream_rate_val: float = 0.004,
) -> int:
    if per_stream_rate_val <= 0:
        return -1
    return int(np.ceil(total_costs / per_stream_rate_val))


def annual_revenue_projection(
    monthly_streams: np.ndarray,
    avg_per_stream: float = 0.004,
    growth_rate: float = 0.0,
) -> float:
    ts = np.asarray(monthly_streams, dtype=float)
    n = len(ts)
    if n == 0:
        return 0.0
    if n >= 12:
        annual_streams = ts[-12:].sum()
    else:
        monthly_avg = ts.mean()
        annual_streams = monthly_avg * 12
    projected = annual_streams * (1 + growth_rate)
    return float(projected * avg_per_stream)


@dataclass
class RecoupmentSchedule:
    advance: float
    monthly_royalty: float
    months_to_recoup: int
    total_paid_at_recoup: float
    post_recoup_monthly: float


def compute_recoupment(
    advance: float,
    monthly_revenue: float,
    royalty_rate: float = 0.20,
    escalation_rate: float = 0.0,
) -> RecoupmentSchedule:
    if monthly_revenue <= 0 or royalty_rate <= 0:
        return RecoupmentSchedule(advance, 0.0, -1, 0.0, 0.0)
    monthly_royalty = monthly_revenue * royalty_rate
    accumulated = 0.0
    months = 0
    rev = monthly_revenue
    while accumulated < advance and months < 600:
        royalty = rev * royalty_rate
        accumulated += royalty
        rev *= (1 + escalation_rate)
        months += 1
    post_recoup = rev * royalty_rate
    return RecoupmentSchedule(
        advance=advance,
        monthly_royalty=round(monthly_royalty, 2),
        months_to_recoup=months,
        total_paid_at_recoup=round(accumulated, 2),
        post_recoup_monthly=round(post_recoup, 2),
    )
