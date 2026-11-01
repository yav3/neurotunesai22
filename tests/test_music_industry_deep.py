"""Tests for music industry royalty engine."""
import numpy as np
import pytest

from models.music_industry.royalty_engine import (
    RightsHolder,
    RoyaltyConfig,
    validate_splits,
    allocate_royalties,
    streaming_payout,
    streaming_payout_curve,
    pro_rata_pool_share,
    mechanical_royalty,
    performance_royalty_estimate,
    sync_fee_estimate,
    project_revenue,
    streams_to_breakeven,
    recoupment_schedule,
)


class TestSplits:
    def test_valid_splits(self):
        holders = [
            RightsHolder(name="A", share_pct=50.0),
            RightsHolder(name="B", share_pct=50.0),
        ]
        assert validate_splits(holders) is True

    def test_invalid_splits(self):
        holders = [
            RightsHolder(name="A", share_pct=50.0),
            RightsHolder(name="B", share_pct=30.0),
        ]
        assert validate_splits(holders) is False

    def test_allocate(self):
        holders = [
            RightsHolder(name="Writer", share_pct=60.0),
            RightsHolder(name="Producer", share_pct=40.0),
        ]
        alloc = allocate_royalties(1000.0, holders)
        assert alloc["Writer"] == pytest.approx(600.0)
        assert alloc["Producer"] == pytest.approx(400.0)


class TestStreaming:
    def test_basic_payout(self):
        pay = streaming_payout(1_000_000, per_stream_rate=0.004, platform_cut=0.30)
        assert pay == pytest.approx(1_000_000 * 0.004 * 0.7)

    def test_zero_streams(self):
        assert streaming_payout(0) == 0.0

    def test_payout_curve_shape(self):
        monthly = np.array([100_000, 200_000, 150_000])
        curve = streaming_payout_curve(monthly)
        assert len(curve) == 3
        assert curve[1] > curve[0]

    def test_pro_rata(self):
        share = pro_rata_pool_share(1000, 1_000_000, 5_000_000)
        assert share == pytest.approx(5000.0)

    def test_pro_rata_zero_total(self):
        assert pro_rata_pool_share(1000, 0, 5000) == 0.0


class TestMechanical:
    def test_short_song(self):
        roy = mechanical_royalty(1000, rate_per_copy=0.091, duration_minutes=3.0)
        assert roy == pytest.approx(91.0)

    def test_long_song(self):
        roy = mechanical_royalty(1000, duration_minutes=7.0, long_song_rate_per_min=0.0175)
        assert roy == pytest.approx(1000 * 7.0 * 0.0175)


class TestPerformance:
    def test_radio_only(self):
        r = performance_royalty_estimate(radio_spins=1000, rate_per_spin=0.10)
        assert r == pytest.approx(100.0)

    def test_mixed_media(self):
        r = performance_royalty_estimate(radio_spins=500, tv_plays=10, live_performances=20)
        assert r > 0


class TestSync:
    def test_commercial_more_than_tv(self):
        tv = sync_fee_estimate("tv_series", audience_size=1000000)
        comm = sync_fee_estimate("commercial", audience_size=1000000)
        assert comm > tv

    def test_exclusivity_doubles(self):
        base = sync_fee_estimate("film", exclusivity=False)
        excl = sync_fee_estimate("film", exclusivity=True)
        assert excl == pytest.approx(base * 2.0)

    def test_unknown_type_fallback(self):
        fee = sync_fee_estimate("podcast")
        assert fee > 0


class TestProjection:
    def test_basic_projection(self):
        streams = np.array([100_000] * 12)
        proj = project_revenue(streams, copies_sold=500, radio_spins=100)
        assert proj.streaming > 0
        assert proj.mechanical > 0
        assert proj.performance > 0
        assert proj.total == pytest.approx(
            proj.streaming + proj.mechanical + proj.performance + proj.sync
        )


class TestBreakeven:
    def test_recoup_basic(self):
        n = streams_to_breakeven(1000.0, per_stream_rate=0.004,
                                 platform_cut=0.30, artist_share_pct=20.0)
        net_per = 0.004 * 0.7 * 0.2
        assert n == int(np.ceil(1000.0 / net_per))

    def test_zero_share(self):
        assert streams_to_breakeven(1000.0, artist_share_pct=0.0) == -1

    def test_recoupment_schedule(self):
        monthly = np.array([500_000] * 24)
        month = recoupment_schedule(monthly, advance=5000.0)
        assert 0 <= month < 24

    def test_never_recoups(self):
        monthly = np.array([100] * 3)
        month = recoupment_schedule(monthly, advance=1_000_000.0)
        assert month == -1
