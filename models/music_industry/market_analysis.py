"""Music market analysis: concentration, growth, trends, seasonality."""
import numpy as np


def genre_market_share(genre_streams: dict[str, int]) -> dict[str, float]:
    total = sum(genre_streams.values())
    if total == 0:
        return {k: 0.0 for k in genre_streams}
    return {k: v / total for k, v in genre_streams.items()}


def herfindahl_index(market_shares: np.ndarray) -> float:
    shares = market_shares / (np.sum(market_shares) + 1e-30)
    return float(np.clip(np.sum(shares ** 2), 0.0, 1.0))


def genre_growth_rate(current: dict[str, int],
                      previous: dict[str, int]) -> dict[str, float]:
    result = {}
    for genre in set(current) | set(previous):
        c = current.get(genre, 0)
        p = previous.get(genre, 0)
        if p > 0:
            result[genre] = (c - p) / p
        elif c > 0:
            result[genre] = 1.0
        else:
            result[genre] = 0.0
    return result


def artist_market_position(artist_streams: int,
                           genre_total: int) -> str:
    if genre_total <= 0:
        return "niche"
    share = artist_streams / genre_total
    if share >= 0.05:
        return "major"
    elif share >= 0.01:
        return "mid-tier"
    elif share >= 0.001:
        return "emerging"
    return "niche"


def trend_momentum(monthly_streams: np.ndarray) -> float:
    if len(monthly_streams) < 2:
        return 0.0
    log_streams = np.log(monthly_streams.astype(float) + 1.0)
    x = np.arange(len(log_streams))
    slope, _ = np.polyfit(x, log_streams, 1)
    return float(slope)


def seasonal_index(monthly_data: np.ndarray) -> np.ndarray:
    if len(monthly_data) < 12:
        return np.ones(12)
    n_years = len(monthly_data) // 12
    data = monthly_data[:n_years * 12].reshape(n_years, 12)
    monthly_avg = data.mean(axis=0)
    grand_mean = monthly_avg.mean()
    if grand_mean < 1e-10:
        return np.ones(12)
    return monthly_avg / grand_mean


def market_saturation(new_releases: int, total_catalog: int) -> float:
    if total_catalog <= 0:
        return 0.0
    return float(np.clip(new_releases / total_catalog, 0.0, 1.0))
