"""Music market analytics: streaming share, genre trends, artist growth, release timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "StreamingMarketShare", "GenreTrend", "ArtistGrowthModel",
    "compute_market_share", "detect_genre_trends", "model_artist_growth",
    "optimal_release_day", "playlist_placement_value",
    "herfindahl_index", "genre_diversity_index",
    "exponential_smoothing_forecast", "release_timing_score",
]


@dataclass
class StreamingMarketShare:
    platform_shares: Dict[str, float]
    hhi: float
    total_streams: int

    @property
    def leader(self) -> str:
        return max(self.platform_shares, key=self.platform_shares.get)

    @property
    def concentration(self) -> str:
        if self.hhi > 2500:
            return "highly_concentrated"
        if self.hhi > 1500:
            return "moderately_concentrated"
        return "competitive"


def herfindahl_index(shares: np.ndarray) -> float:
    s = np.asarray(shares, dtype=float)
    s = s / max(s.sum(), 1e-15)
    return float(np.sum((s * 100) ** 2))


def compute_market_share(
    stream_counts: Dict[str, int],
) -> StreamingMarketShare:
    total = sum(stream_counts.values())
    shares = {k: v / max(total, 1) for k, v in stream_counts.items()}
    hhi = herfindahl_index(np.array(list(shares.values())))
    return StreamingMarketShare(
        platform_shares=shares, hhi=hhi, total_streams=total,
    )


@dataclass
class GenreTrend:
    genre: str
    slope: float
    r_squared: float
    momentum: float
    is_trending_up: bool


def detect_genre_trends(
    genre_streams: Dict[str, np.ndarray],
    window: int = 4,
) -> List[GenreTrend]:
    trends = []
    for genre, ts in genre_streams.items():
        ts = np.asarray(ts, dtype=float)
        n = len(ts)
        if n < 3:
            trends.append(GenreTrend(genre, 0.0, 0.0, 0.0, False))
            continue
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = ts.mean()
        num = np.sum((x - x_mean) * (ts - y_mean))
        den = np.sum((x - x_mean) ** 2)
        slope = num / max(den, 1e-15)
        ss_res = np.sum((ts - (y_mean + slope * (x - x_mean))) ** 2)
        ss_tot = np.sum((ts - y_mean) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15) if ss_tot > 1e-15 else 0.0
        w = min(window, n)
        recent = ts[-w:].mean()
        older = ts[:-w].mean() if n > w else ts.mean()
        momentum = (recent - older) / max(abs(older), 1e-9)
        trends.append(GenreTrend(
            genre=genre, slope=float(slope), r_squared=float(r2),
            momentum=float(momentum), is_trending_up=slope > 0 and r2 > 0.3,
        ))
    trends.sort(key=lambda t: t.momentum, reverse=True)
    return trends


def genre_diversity_index(genre_shares: np.ndarray) -> float:
    s = np.asarray(genre_shares, dtype=float)
    s = s / max(s.sum(), 1e-15)
    s = s[s > 0]
    return float(-np.sum(s * np.log(s)))


@dataclass
class ArtistGrowthModel:
    current_monthly: float
    growth_rate: float
    carrying_capacity: float
    time_to_half: float

    def predict(self, months_ahead: int) -> np.ndarray:
        t = np.arange(months_ahead, dtype=float)
        K = self.carrying_capacity
        N0 = self.current_monthly
        r = self.growth_rate
        return K / (1.0 + ((K - N0) / max(N0, 1.0)) * np.exp(-r * t))


def model_artist_growth(
    monthly_streams: np.ndarray,
    carrying_capacity: Optional[float] = None,
) -> ArtistGrowthModel:
    ts = np.asarray(monthly_streams, dtype=float)
    n = len(ts)
    if n < 2:
        return ArtistGrowthModel(ts[-1] if n else 0, 0.0, ts[-1] if n else 0, 0.0)
    current = ts[-1]
    if carrying_capacity is None:
        carrying_capacity = current * 10.0
    K = carrying_capacity
    ratios = ts[1:] / np.maximum(ts[:-1], 1.0)
    log_ratios = np.log(np.maximum(ratios, 1e-9))
    r = float(np.mean(log_ratios))
    if r > 0:
        t_half = np.log((K - ts[0]) / max(ts[0], 1.0)) / max(r, 1e-9) / 2.0
    else:
        t_half = float("inf")
    return ArtistGrowthModel(
        current_monthly=float(current),
        growth_rate=float(r),
        carrying_capacity=K,
        time_to_half=float(t_half),
    )


def exponential_smoothing_forecast(
    series: np.ndarray,
    alpha: float = 0.3,
    n_ahead: int = 6,
) -> np.ndarray:
    ts = np.asarray(series, dtype=float)
    if len(ts) == 0:
        return np.zeros(n_ahead)
    level = ts[0]
    for v in ts[1:]:
        level = alpha * v + (1 - alpha) * level
    return np.full(n_ahead, level)


RELEASE_DAY_WEIGHTS = {
    0: 1.0,   # Monday
    1: 0.9,
    2: 0.85,
    3: 0.95,
    4: 1.2,   # Friday — industry standard
    5: 0.7,
    6: 0.6,
}


def optimal_release_day(
    genre: str,
    target_playlist_refresh_day: int = 4,
) -> int:
    genre_offsets = {
        "pop": 4, "hip_hop": 4, "country": 4,
        "electronic": 4, "indie": 3, "classical": 0,
        "jazz": 1, "metal": 4, "latin": 4,
    }
    return genre_offsets.get(genre, target_playlist_refresh_day)


def release_timing_score(
    day_of_week: int,
    month: int,
    genre: str = "pop",
) -> float:
    day_w = RELEASE_DAY_WEIGHTS.get(day_of_week % 7, 0.8)
    month_seasonality = {
        1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0,
        5: 1.05, 6: 1.1, 7: 0.95, 8: 0.9,
        9: 1.15, 10: 1.1, 11: 1.0, 12: 0.8,
    }
    month_w = month_seasonality.get(month, 1.0)
    ideal_day = optimal_release_day(genre)
    day_penalty = 1.0 - 0.1 * min(abs(day_of_week - ideal_day), 7 - abs(day_of_week - ideal_day))
    return float(np.clip(day_w * month_w * day_penalty, 0.0, 2.0))


def playlist_placement_value(
    playlist_followers: int,
    position_in_playlist: int,
    playlist_length: int,
    genre_match_score: float = 1.0,
) -> float:
    follower_value = np.log1p(playlist_followers) * 0.001
    position_decay = np.exp(-0.03 * position_in_playlist)
    length_factor = 1.0 / np.sqrt(max(playlist_length, 1))
    return float(follower_value * position_decay * length_factor * genre_match_score * 1000)
