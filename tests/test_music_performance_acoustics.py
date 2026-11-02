"""Tests for stage acoustics analysis."""
import numpy as np
import pytest

from models.music_performance.stage_acoustics import (
    estimate_rt60, clarity_c80, definition_d50, center_time,
    bass_ratio, stage_support, analyze_venue, VenueProfile,
)


def _synthetic_ir(rt60: float = 1.0, sr: int = 44100) -> np.ndarray:
    n = int(sr * rt60 * 1.5)
    t = np.arange(n) / sr
    decay = np.exp(-6.908 * t / rt60)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n)
    return noise * decay


class TestRT60:
    def test_known_decay(self):
        ir = _synthetic_ir(rt60=1.0)
        rt = estimate_rt60(ir, sr=44100)
        assert 0.5 < rt < 2.0

    def test_short_ir(self):
        assert estimate_rt60(np.array([1.0]), sr=44100) == 0.0

    def test_silence(self):
        assert estimate_rt60(np.zeros(4410), sr=44100) == 0.0


class TestClarity:
    def test_dry_room(self):
        ir = _synthetic_ir(rt60=0.3)
        c = clarity_c80(ir, sr=44100)
        assert c > 0.0

    def test_reverberant(self):
        ir = _synthetic_ir(rt60=3.0)
        c = clarity_c80(ir, sr=44100)
        assert isinstance(c, float)

    def test_short(self):
        assert clarity_c80(np.array([1.0])) == 0.0


class TestDefinition:
    def test_dry_room_high(self):
        ir = np.zeros(4410)
        ir[0] = 1.0
        d = definition_d50(ir, sr=44100)
        assert d == pytest.approx(1.0)

    def test_reverberant_lower(self):
        ir = _synthetic_ir(rt60=2.0)
        d = definition_d50(ir, sr=44100)
        assert 0.0 < d < 1.0


class TestCenterTime:
    def test_impulse(self):
        ir = np.zeros(4410)
        ir[0] = 1.0
        ts = center_time(ir, sr=44100)
        assert ts == pytest.approx(0.0)

    def test_late_energy(self):
        ir = np.zeros(44100)
        ir[22050] = 1.0
        ts = center_time(ir, sr=44100)
        assert ts == pytest.approx(0.5, abs=0.01)


class TestBassRatio:
    def test_warm(self):
        assert bass_ratio(1.5, 1.0) == pytest.approx(1.5)

    def test_bright(self):
        assert bass_ratio(0.8, 1.0) == pytest.approx(0.8)

    def test_zero_mid(self):
        assert bass_ratio(1.0, 0.0) == 0.0


class TestStageSupport:
    def test_short_ir(self):
        assert stage_support(np.array([1.0])) == 0.0


class TestAnalyzeVenue:
    def test_returns_profile(self):
        ir = _synthetic_ir(rt60=1.5)
        profile = analyze_venue(ir, sr=44100)
        assert isinstance(profile, VenueProfile)
        assert profile.rt60 > 0
        assert isinstance(profile.clarity_c80, float)
