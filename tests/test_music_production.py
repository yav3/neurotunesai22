"""Tests for music_production module: mixing, mastering, analysis, tempo."""
import numpy as np
import pytest


# ── Mixing ──────────────────────────────────────────────────────────────

class TestMixBus:
    def test_linear_gain_unity(self):
        from models.music_production.mixing import MixBus
        bus = MixBus(gain_db=0.0)
        assert abs(bus.linear_gain() - 1.0) < 1e-6

    def test_linear_gain_minus6(self):
        from models.music_production.mixing import MixBus
        bus = MixBus(gain_db=-6.0)
        assert abs(bus.linear_gain() - 0.5012) < 0.01


class TestStereoMixer:
    def test_empty_mixdown(self):
        from models.music_production.mixing import StereoMixer
        m = StereoMixer()
        out = m.mix_down()
        assert out.shape == (0, 2)

    def test_single_channel_center(self):
        from models.music_production.mixing import StereoMixer, MixerChannel, MixBus
        sig = np.ones(1000, dtype=np.float32) * 0.5
        ch = MixerChannel(name="a", signal=sig, bus=MixBus(pan=0.0, gain_db=0.0))
        m = StereoMixer()
        m.add_channel(ch)
        out = m.mix_down()
        assert out.shape == (1000, 2)
        assert abs(out[:, 0].mean() - out[:, 1].mean()) < 1e-4

    def test_mute_excludes_channel(self):
        from models.music_production.mixing import StereoMixer, MixerChannel, MixBus
        sig = np.ones(100, dtype=np.float32)
        ch = MixerChannel(name="a", signal=sig, bus=MixBus(mute=True))
        m = StereoMixer()
        m.add_channel(ch)
        out = m.mix_down()
        assert np.allclose(out, 0.0)

    def test_solo_isolates_channel(self):
        from models.music_production.mixing import StereoMixer, MixerChannel, MixBus
        s1 = np.ones(100, dtype=np.float32)
        s2 = np.ones(100, dtype=np.float32) * 0.5
        ch1 = MixerChannel(name="a", signal=s1, bus=MixBus(solo=True))
        ch2 = MixerChannel(name="b", signal=s2, bus=MixBus())
        m = StereoMixer()
        m.add_channel(ch1)
        m.add_channel(ch2)
        out_solo = m.mix_down()
        m2 = StereoMixer()
        m2.add_channel(MixerChannel(name="a", signal=s1, bus=MixBus()))
        out_alone = m2.mix_down()
        assert np.allclose(out_solo, out_alone, atol=1e-6)

    def test_set_levels(self):
        from models.music_production.mixing import StereoMixer, MixerChannel, MixBus
        ch = MixerChannel(name="drums", signal=np.ones(10, dtype=np.float32), bus=MixBus())
        m = StereoMixer()
        m.add_channel(ch)
        m.set_levels({"drums": -6.0})
        assert m.channels[0].bus.gain_db == -6.0


class TestGainStaging:
    def test_auto_gain_silent(self):
        from models.music_production.mixing import GainStaging
        sig = np.zeros(100, dtype=np.float32)
        out, db = GainStaging.auto_gain(sig)
        assert db == 0.0

    def test_headroom_check(self):
        from models.music_production.mixing import GainStaging
        sig = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        info = GainStaging.headroom_check(sig, ceiling_db=-1.0)
        assert info["peak_linear"] == pytest.approx(0.5, abs=1e-6)
        assert not info["clipping"]


# ── Mastering ───────────────────────────────────────────────────────────

class TestLoudnessMeter:
    def test_rms_sine(self):
        from models.music_production.mastering import LoudnessMeter
        t = np.linspace(0, 1, 44100, endpoint=False)
        sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        rms = LoudnessMeter.measure_rms(sig)
        assert abs(rms - 1.0 / np.sqrt(2)) < 0.01

    def test_peak(self):
        from models.music_production.mastering import LoudnessMeter
        sig = np.array([0.3, -0.7, 0.5], dtype=np.float32)
        assert LoudnessMeter.measure_peak(sig) == pytest.approx(0.7, abs=1e-6)

    def test_lufs_silent(self):
        from models.music_production.mastering import LoudnessMeter
        sig = np.zeros(1000, dtype=np.float32)
        assert LoudnessMeter.measure_lufs(sig) == -np.inf

    def test_dynamic_range(self):
        from models.music_production.mastering import LoudnessMeter
        t = np.linspace(0, 1, 44100, endpoint=False)
        sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        dr = LoudnessMeter.measure_dynamic_range(sig)
        assert dr > 2.0


class TestStereoWidener:
    def test_mid_side_roundtrip(self):
        from models.music_production.mastering import StereoWidener
        stereo = np.random.default_rng(42).standard_normal((1000, 2)).astype(np.float32)
        mid, side = StereoWidener.mid_side_encode(stereo)
        recovered = StereoWidener.mid_side_decode(mid, side)
        assert np.allclose(stereo, recovered, atol=1e-4)

    def test_width_zero_is_mono(self):
        from models.music_production.mastering import StereoWidener
        stereo = np.random.default_rng(7).standard_normal((500, 2)).astype(np.float32)
        out = StereoWidener.process(stereo, width_factor=0.0)
        assert np.allclose(out[:, 0], out[:, 1], atol=1e-6)


class TestBrickwallLimiter:
    def test_ceiling_respected(self):
        from models.music_production.mastering import BrickwallLimiter
        sig = np.array([0.5, 1.5, -2.0, 0.3], dtype=np.float32)
        out = BrickwallLimiter.process(sig, ceiling_db=-0.3)
        ceiling_lin = 10.0 ** (-0.3 / 20.0)
        assert np.max(np.abs(out)) <= ceiling_lin + 1e-6


class TestMasteringChain:
    def test_chain_process(self):
        from models.music_production.mastering import MasteringChain, EQStage, CompressionStage
        t = np.linspace(0, 0.5, 22050, endpoint=False)
        sig = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        chain = MasteringChain(stages=[EQStage(), CompressionStage()])
        out = chain.process(sig, sr=44100)
        assert out.shape == sig.shape
        report = chain.analyze_before_after()
        assert "before" in report and "after" in report

    def test_eq_stereo_uses_shelf_db(self):
        from models.music_production.mastering import EQStage
        rng = np.random.default_rng(99)
        stereo = rng.standard_normal((4096, 2)).astype(np.float32) * 0.1
        eq = EQStage(high_shelf_db=-6.0)
        out = eq.process(stereo, sr=44100)
        assert out.shape == stereo.shape
        assert np.all(np.isfinite(out))


# ── Analysis ────────────────────────────────────────────────────────────

class TestSpectralBalance:
    def test_band_ratios_sum_to_one(self):
        from models.music_production.analysis import SpectralBalance
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(44100).astype(np.float32)
        result = SpectralBalance.analyze(sig, sr=44100)
        assert abs(result["low"] + result["mid"] + result["high"] - 1.0) < 1e-4

    def test_empty_signal(self):
        from models.music_production.analysis import SpectralBalance
        result = SpectralBalance.analyze(np.array([], dtype=np.float32), sr=44100)
        assert result["centroid_hz"] == 0.0


class TestDynamicProfile:
    def test_sine_crest_factor(self):
        from models.music_production.analysis import DynamicProfile
        t = np.linspace(0, 1, 44100, endpoint=False)
        sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        d = DynamicProfile.analyze(sig)
        assert abs(d["crest_factor"] - np.sqrt(2)) < 0.05

    def test_loudness_envelope_shape(self):
        from models.music_production.analysis import DynamicProfile
        sig = np.random.default_rng(1).standard_normal(22050).astype(np.float32)
        env = DynamicProfile.loudness_envelope(sig, window_ms=50.0, sr=44100)
        assert env.shape == sig.shape


class TestStereoAnalysis:
    def test_mono_signal(self):
        from models.music_production.analysis import StereoAnalysis
        mono = np.ones(100, dtype=np.float32)
        result = StereoAnalysis.analyze(mono[:, None])
        assert result["correlation"] == 1.0

    def test_anti_phase(self):
        from models.music_production.analysis import StereoAnalysis
        left = np.sin(np.linspace(0, 10, 1000))
        right = -left
        stereo = np.column_stack([left, right]).astype(np.float32)
        result = StereoAnalysis.analyze(stereo)
        assert result["correlation"] < -0.9
        assert result["phase_issues"]


class TestReferenceComparison:
    def test_compare_returns_diffs(self):
        from models.music_production.analysis import ReferenceComparison
        ref = ReferenceComparison()
        sig = np.random.default_rng(5).standard_normal(44100).astype(np.float32) * 0.1
        result = ref.compare(sig, sr=44100)
        assert "lufs_diff" in result
        assert "measured" in result


# ── Tempo ───────────────────────────────────────────────────────────────

class TestBeatGrid:
    def test_beat_times_120bpm(self):
        from models.music_production.tempo import BeatGrid
        g = BeatGrid(bpm=120.0)
        beats = g.beat_times(2.0)
        assert len(beats) == 4
        assert abs(beats[1] - 0.5) < 1e-6

    def test_bar_times(self):
        from models.music_production.tempo import BeatGrid
        g = BeatGrid(bpm=120.0, time_signature=(4, 4))
        bars = g.bar_times(4.0)
        assert abs(bars[1] - 2.0) < 1e-6

    def test_snap_to_grid(self):
        from models.music_production.tempo import BeatGrid
        g = BeatGrid(bpm=120.0)
        assert abs(g.snap_to_grid(0.48) - 0.5) < 1e-6

    def test_is_downbeat(self):
        from models.music_production.tempo import BeatGrid
        g = BeatGrid(bpm=120.0, time_signature=(4, 4))
        assert g.is_downbeat(0.0)
        assert not g.is_downbeat(0.5)


class TestTempoEstimator:
    def test_onset_strength_shape(self):
        from models.music_production.tempo import TempoEstimator
        sig = np.random.default_rng(3).standard_normal(44100).astype(np.float32)
        odf = TempoEstimator.onset_strength(sig, sr=44100, hop=512)
        assert len(odf) == 44100 // 512

    def test_estimate_bpm_click(self):
        from models.music_production.tempo import ClickTrack, TempoEstimator
        click = ClickTrack.generate(bpm=120.0, duration_s=4.0, sr=22050)
        est = TempoEstimator.estimate_bpm(click, sr=22050, min_bpm=80, max_bpm=160)
        assert abs(est - 120.0) < 15.0


class TestGrooveTemplate:
    def test_straight_no_change(self):
        from models.music_production.tempo import GrooveTemplate
        g = GrooveTemplate(swing_amount=0.0, humanize_ms=0.0)
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        out = g.apply(beats)
        assert np.allclose(out, beats)

    def test_swing_shifts_offbeats(self):
        from models.music_production.tempo import GrooveTemplate
        g = GrooveTemplate(swing_amount=0.5)
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        out = g.apply(beats)
        assert out[1] > 0.5
        assert abs(out[0] - 0.0) < 1e-10


class TestClickTrack:
    def test_generate_length(self):
        from models.music_production.tempo import ClickTrack
        out = ClickTrack.generate(bpm=120.0, duration_s=2.0, sr=44100)
        assert len(out) == 88200

    def test_generate_has_energy(self):
        from models.music_production.tempo import ClickTrack
        out = ClickTrack.generate(bpm=120.0, duration_s=1.0, sr=44100)
        assert np.max(np.abs(out)) > 0.1

    def test_zero_bpm(self):
        from models.music_production.tempo import ClickTrack
        out = ClickTrack.generate(bpm=0.0, duration_s=1.0)
        assert len(out) == 0
