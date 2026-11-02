"""Tests for deep acoustics models: impulse response, room analysis, materials."""
import numpy as np
import pytest

from models.music_acoustics.impulse_response import (
    ImpulseResponse, IRSynthesizer, Convolver,
)
from models.music_acoustics.acoustic_features import (
    RoomAcousticAnalyzer, SpeechIntelligibility, AcousticQualityMetrics,
    MaterialAbsorption,
)


SR = 16000


class TestImpulseResponse:
    def _make_exp_ir(self, rt60=1.0, sr=SR):
        n = int(sr * rt60 * 1.5)
        t = np.arange(n) / sr
        tau = rt60 / (3 * np.log(10))
        rng = np.random.default_rng(42)
        return rng.standard_normal(n) * np.exp(-t / tau)

    def test_rt60_approximate(self):
        ir_data = self._make_exp_ir(rt60=1.0)
        ir = ImpulseResponse(ir_data, SR)
        estimated = ir.rt60()
        assert 0.5 < estimated < 2.0

    def test_edt_positive(self):
        ir_data = self._make_exp_ir(rt60=1.0)
        ir = ImpulseResponse(ir_data, SR)
        assert ir.edt() > 0

    def test_clarity_finite(self):
        ir_data = self._make_exp_ir(rt60=0.5)
        ir = ImpulseResponse(ir_data, SR)
        c50 = ir.clarity(50)
        assert np.isfinite(c50)

    def test_definition_range(self):
        ir_data = self._make_exp_ir(rt60=0.5)
        ir = ImpulseResponse(ir_data, SR)
        d50 = ir.definition(50)
        assert 0 <= d50 <= 1.0

    def test_centre_time_positive(self):
        ir_data = self._make_exp_ir(rt60=0.5)
        ir = ImpulseResponse(ir_data, SR)
        ts = ir.centre_time()
        assert ts > 0

    def test_iacc_identical(self):
        ir_data = self._make_exp_ir(rt60=0.3)
        ir = ImpulseResponse(ir_data, SR)
        iacc = ir.iacc(ir_data, ir_data)
        assert iacc > 0.9

    def test_freq_dependent_rt60(self):
        ir_data = self._make_exp_ir(rt60=1.0)
        ir = ImpulseResponse(ir_data, SR)
        bands = ir.rt60(freq_bands=[500, 1000, 2000])
        assert 500 in bands
        assert 1000 in bands


class TestIRSynthesizer:
    def test_exponential_decay_length(self):
        synth = IRSynthesizer()
        ir = synth.exponential_decay(1.0, SR)
        assert len(ir) == int(SR * 1.5)

    def test_exponential_decay_decreasing(self):
        synth = IRSynthesizer()
        ir = synth.exponential_decay(1.0, SR)
        rms_first = np.sqrt(np.mean(ir[:SR // 4] ** 2))
        rms_last = np.sqrt(np.mean(ir[-SR // 4:] ** 2))
        assert rms_first > rms_last

    def test_room_model_nonzero(self):
        synth = IRSynthesizer()
        ir = synth.room_model((5.0, 4.0, 3.0), 0.3, SR, n_reflections=10)
        assert np.max(np.abs(ir)) > 0

    def test_diffuse_field_nonzero(self):
        synth = IRSynthesizer()
        ir = synth.diffuse_field(1.0, SR, density=500)
        assert np.max(np.abs(ir)) > 0


class TestConvolver:
    def test_delta_passthrough(self):
        delta = np.zeros(100)
        delta[0] = 1.0
        conv = Convolver(delta)
        signal = np.random.default_rng(42).standard_normal(1000)
        result = conv.process(signal)
        np.testing.assert_allclose(result, signal, atol=1e-10)

    def test_partitioned_matches_direct(self):
        rng = np.random.default_rng(42)
        ir = rng.standard_normal(500)
        signal = rng.standard_normal(2000)
        conv = Convolver(ir)
        direct = conv.process(signal)
        partitioned = conv.process_partitioned(signal, block_size=256)
        np.testing.assert_allclose(partitioned, direct, atol=1e-6)

    def test_output_length(self):
        ir = np.ones(50)
        signal = np.ones(200)
        conv = Convolver(ir)
        result = conv.process(signal)
        assert len(result) == 200


class TestRoomAcousticAnalyzer:
    def _make_ir(self, rt60=1.0):
        n = int(SR * rt60 * 1.5)
        t = np.arange(n) / SR
        tau = rt60 / (3 * np.log(10))
        return np.random.default_rng(42).standard_normal(n) * np.exp(-t / tau)

    def test_schroeder_decreasing(self):
        ir = self._make_ir(1.0)
        ra = RoomAcousticAnalyzer(ir, SR)
        sc = ra.schroeder_curve()
        assert sc[0] >= sc[-1]

    def test_parameters_keys(self):
        ir = self._make_ir(0.5)
        ra = RoomAcousticAnalyzer(ir, SR)
        params = ra.parameters()
        for key in ("rt60", "edt", "c50", "c80", "d50", "d80", "ts"):
            assert key in params

    def test_d50_d80_ordering(self):
        ir = self._make_ir(0.5)
        ra = RoomAcousticAnalyzer(ir, SR)
        p = ra.parameters()
        assert p["d50"] <= p["d80"]

    def test_freq_dependent_rt60(self):
        ir = self._make_ir(1.0)
        ra = RoomAcousticAnalyzer(ir, SR)
        frt = ra.frequency_dependent_rt60([500, 1000])
        assert 500 in frt
        assert 1000 in frt


class TestSpeechIntelligibility:
    def test_sti_range(self):
        n = int(SR * 1.0)
        t = np.arange(n) / SR
        ir = np.random.default_rng(42).standard_normal(n) * np.exp(-t / 0.15)
        si = SpeechIntelligibility(ir, SR)
        sti = si.sti()
        assert 0 <= sti <= 1.0

    def test_shorter_rt60_higher_sti(self):
        rng = np.random.default_rng(42)
        n = int(SR * 2.0)
        t = np.arange(n) / SR
        ir_short = rng.standard_normal(n) * np.exp(-t / 0.05)
        ir_long = rng.standard_normal(n) * np.exp(-t / 0.5)
        sti_short = SpeechIntelligibility(ir_short, SR).sti()
        sti_long = SpeechIntelligibility(ir_long, SR).sti()
        assert sti_short > sti_long

    def test_alcons_nonneg(self):
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        ir = np.random.default_rng(42).standard_normal(n) * np.exp(-t / 0.1)
        si = SpeechIntelligibility(ir, SR)
        assert si.alcons() >= 0

    def test_mtf_range(self):
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        ir = np.random.default_rng(42).standard_normal(n) * np.exp(-t / 0.1)
        si = SpeechIntelligibility(ir, SR)
        mtf = si.modulation_transfer_function(1000)
        assert np.all(mtf >= 0)
        assert np.all(mtf <= 1.0)


class TestAcousticQualityMetrics:
    @pytest.fixture
    def aqm(self):
        return AcousticQualityMetrics()

    def test_warmth_range(self, aqm):
        w = aqm.warmth(1.2, 1.0)
        assert 0 <= w <= 1.0

    def test_brilliance_range(self, aqm):
        b = aqm.brilliance(0.8, 1.0)
        assert 0 <= b <= 1.0

    def test_intimacy_range(self, aqm):
        i = aqm.intimacy(15.0)
        assert 0 <= i <= 1.0

    def test_envelopment_range(self, aqm):
        e = aqm.envelopment(0.6)
        assert 0 <= e <= 1.0

    def test_overall_quality(self, aqm):
        metrics = {"warmth": 0.8, "brilliance": 0.7, "intimacy": 0.9, "envelopment": 0.6}
        q = aqm.overall_quality(metrics)
        assert 0 < q < 1.0


class TestMaterialAbsorption:
    @pytest.fixture
    def ma(self):
        return MaterialAbsorption()

    def test_alpha_range(self, ma):
        for mat in ("concrete", "carpet", "glass", "curtain", "wood", "acoustic_tile"):
            for freq in (125, 500, 2000):
                a = ma.alpha(mat, freq)
                assert 0 <= a <= 1.0

    def test_carpet_more_absorptive(self, ma):
        assert ma.alpha("carpet", 1000) > ma.alpha("concrete", 1000)

    def test_sabine_rt60_positive(self, ma):
        volume = 100.0
        surfaces = {"floor": 25.0, "ceiling": 25.0, "walls": 60.0}
        materials = {"floor": "carpet", "ceiling": "acoustic_tile", "walls": "concrete"}
        rt = ma.sabine_rt60(volume, surfaces, materials)
        assert rt > 0

    def test_sabine_more_absorption_shorter_rt(self, ma):
        volume = 100.0
        surfaces = {"floor": 25.0, "walls": 60.0}
        rt_hard = ma.sabine_rt60(volume, surfaces, {"floor": "concrete", "walls": "concrete"})
        rt_soft = ma.sabine_rt60(volume, surfaces, {"floor": "carpet", "walls": "curtain"})
        assert rt_soft < rt_hard

    def test_interpolation(self, ma):
        a300 = ma.alpha("concrete", 300)
        a125 = ma.alpha("concrete", 125)
        a500 = ma.alpha("concrete", 500)
        assert a125 <= a300 <= a500 or a500 <= a300 <= a125
