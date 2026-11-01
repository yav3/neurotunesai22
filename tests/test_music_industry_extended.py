"""Tests for revenue models and market analysis."""
import numpy as np
import pytest
from models.music_industry.market_analysis import (
    genre_market_share, herfindahl_index, genre_growth_rate,
    artist_market_position, trend_momentum, seasonal_index,
    market_saturation,
)


class TestMarketAnalysis:
    def test_genre_market_share_sums_to_one(self):
        shares = genre_market_share({"pop": 5000, "rock": 3000, "jazz": 2000})
        assert abs(sum(shares.values()) - 1.0) < 1e-10

    def test_genre_market_share_empty(self):
        shares = genre_market_share({"pop": 0, "rock": 0})
        assert shares["pop"] == 0.0

    def test_herfindahl_monopoly(self):
        h = herfindahl_index(np.array([1.0]))
        assert abs(h - 1.0) < 0.01

    def test_herfindahl_competitive(self):
        h = herfindahl_index(np.array([0.25, 0.25, 0.25, 0.25]))
        assert abs(h - 0.25) < 0.01

    def test_herfindahl_bounds(self):
        h = herfindahl_index(np.array([0.5, 0.3, 0.2]))
        assert 0.0 <= h <= 1.0

    def test_genre_growth_rate_positive(self):
        g = genre_growth_rate({"pop": 200}, {"pop": 100})
        assert g["pop"] == 1.0

    def test_genre_growth_rate_new_genre(self):
        g = genre_growth_rate({"pop": 100, "kpop": 50}, {"pop": 100})
        assert g["kpop"] == 1.0

    def test_artist_position_major(self):
        assert artist_market_position(60000, 1000000) == "major"

    def test_artist_position_niche(self):
        assert artist_market_position(10, 1000000) == "niche"

    def test_trend_momentum_positive(self):
        data = np.array([100, 200, 400, 800, 1600], dtype=float)
        assert trend_momentum(data) > 0

    def test_trend_momentum_negative(self):
        data = np.array([1600, 800, 400, 200, 100], dtype=float)
        assert trend_momentum(data) < 0

    def test_seasonal_index_length(self):
        data = np.random.default_rng(42).integers(100, 200, 24).astype(float)
        si = seasonal_index(data)
        assert len(si) == 12

    def test_seasonal_index_mean_approx_one(self):
        data = np.random.default_rng(42).integers(100, 200, 36).astype(float)
        si = seasonal_index(data)
        assert abs(np.mean(si) - 1.0) < 0.1

    def test_market_saturation_range(self):
        s = market_saturation(500, 10000)
        assert 0.0 <= s <= 1.0

    def test_market_saturation_zero_catalog(self):
        assert market_saturation(10, 0) == 0.0

    def test_artist_position_emerging(self):
        assert artist_market_position(5000, 1000000) == "emerging"

    def test_trend_momentum_flat(self):
        data = np.ones(12) * 100.0
        m = trend_momentum(data)
        assert abs(m) < 0.01
