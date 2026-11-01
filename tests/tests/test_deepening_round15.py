"""Tests for deepening round 15: collaboration, copyright, ethnomusicology, timbre, production."""
import math
import numpy as np
import pytest

# ── Collaboration: conflict resolution ──
from models.music_collaboration.conflict_resolution import (
    EditOp, regions_overlap, detect_conflicts, resolve_last_writer_wins,
    merge_non_overlapping, full_resolution_pipeline,
)

class TestConflictResolution:
    def test_regions_overlap(self):
        assert regions_overlap(0, 4, 2, 6)
        assert not regions_overlap(0, 2, 3, 5)

    def test_detect_conflicts_different_users(self):
        ops = [
            EditOp("alice", 1.0, 0.0, 4.0, "modify"),
            EditOp("bob", 2.0, 2.0, 6.0, "modify"),
        ]
        conflicts = detect_conflicts(ops)
        assert len(conflicts) == 1

    def test_no_conflict_same_user(self):
        ops = [
            EditOp("alice", 1.0, 0.0, 4.0, "modify"),
            EditOp("alice", 2.0, 2.0, 6.0, "modify"),
        ]
        assert len(detect_conflicts(ops)) == 0

    def test_last_writer_wins(self):
        a = EditOp("alice", 1.0, 0.0, 4.0, "modify")
        b = EditOp("bob", 2.0, 2.0, 6.0, "modify")
        winners = resolve_last_writer_wins([(a, b)])
        assert winners[0].user_id == "bob"

    def test_full_pipeline_no_conflicts(self):
        ops = [
            EditOp("alice", 1.0, 0.0, 2.0, "insert"),
            EditOp("bob", 2.0, 4.0, 6.0, "insert"),
        ]
        report = full_resolution_pipeline(ops)
        assert report.conflicts_found == 0
        assert len(report.resolved_ops) == 2


# ── Collaboration: skill matching ──
from models.music_collaboration.skill_matching import (
    MusicianProfile, EnsembleRequirement, genre_overlap, role_fitness,
    greedy_assign, ensemble_compatibility, skill_gap_analysis,
)

class TestSkillMatching:
    def test_genre_overlap_identical(self):
        assert genre_overlap(["jazz", "blues"], ["jazz", "blues"]) == 1.0

    def test_genre_overlap_disjoint(self):
        assert genre_overlap(["jazz"], ["metal"]) == 0.0

    def test_role_fitness_no_skill(self):
        m = MusicianProfile("Alice", ["piano"], {"piano": 0.0})
        assert role_fitness(m, "piano", min_skill=0.5) == 0.0

    def test_greedy_assign(self):
        musicians = [
            MusicianProfile("Alice", ["piano"], {"piano": 0.9}),
            MusicianProfile("Bob", ["drums"], {"drums": 0.8}),
        ]
        req = EnsembleRequirement(roles={"keys": "piano", "rhythm": "drums"})
        assignment = greedy_assign(musicians, req)
        assert assignment["keys"] == "Alice"
        assert assignment["rhythm"] == "Bob"

    def test_ensemble_compatibility_same_genres(self):
        m1 = MusicianProfile("A", genres=["jazz"])
        m2 = MusicianProfile("B", genres=["jazz"])
        assert ensemble_compatibility([m1, m2]) == 1.0


# ── Copyright: fair use ──
from models.music_copyright.fair_use_analysis import (
    Purpose, FairUseFactors, composite_fair_use_score, fair_use_risk_level,
    analyze_fair_use,
)

class TestFairUse:
    def test_parody_low_risk(self):
        f = FairUseFactors(purpose=Purpose.PARODY, amount_fraction=0.1, market_harm=0.1)
        score = composite_fair_use_score(f)
        assert score > 0.6
        assert fair_use_risk_level(score) == "low_risk"

    def test_commercial_high_amount_high_risk(self):
        f = FairUseFactors(purpose=Purpose.COMMERCIAL, amount_fraction=0.9,
                           is_heart_of_work=True, market_harm=0.9)
        score = composite_fair_use_score(f)
        assert score < 0.3
        assert fair_use_risk_level(score) == "high_risk"

    def test_analyze_returns_report(self):
        f = FairUseFactors(purpose=Purpose.EDUCATIONAL, amount_fraction=0.3)
        report = analyze_fair_use(f)
        assert len(report.factor_scores) == 4
        assert 0 <= report.composite <= 1


# ── Copyright: royalty calculation ──
from models.music_copyright.royalty_calculation import (
    mechanical_royalty, streaming_royalty, SyncLicenseTerms, compute_statement,
)

class TestRoyaltyCalculation:
    def test_mechanical_short_song(self):
        r = mechanical_royalty(1000, 180)  # 3 min song
        assert r == pytest.approx(120.0)  # 1000 * 0.12

    def test_mechanical_long_song(self):
        r = mechanical_royalty(1000, 420)  # 7 min
        assert r > 120.0

    def test_streaming_spotify(self):
        r = streaming_royalty(1_000_000, "spotify")
        assert r == pytest.approx(4000.0)

    def test_streaming_tidal_higher(self):
        assert streaming_royalty(1000, "tidal") > streaming_royalty(1000, "spotify")

    def test_sync_exclusive_worldwide(self):
        terms = SyncLicenseTerms(base_fee=5000, exclusivity=True, territory="worldwide")
        assert terms.adjusted_fee() > 5000

    def test_full_statement(self):
        stmt = compute_statement(copies=500, streams={"spotify": 100000, "apple_music": 50000})
        assert stmt.total > 0
        assert stmt.mechanical > 0
        assert stmt.streaming > 0


# ── Ethnomusicology: Indian tala ──
from models.music_ethnomusicology.indian_tala import (
    HINDUSTANI_TALAS, build_carnatic_tala, tala_complexity,
    layakari_subdivision, tihai_pattern,
)

class TestIndianTala:
    def test_teental_16_beats(self):
        t = HINDUSTANI_TALAS["teental"]
        assert t.beats == 16
        assert len(t.bol_pattern) == 16

    def test_rupaktaal_7_beats(self):
        t = HINDUSTANI_TALAS["rupaktaal"]
        assert t.beats == 7

    def test_carnatic_adi_tala(self):
        ct = build_carnatic_tala("triputa", jati=4)  # Adi tala = chatusra-jati triputa
        assert ct.aksharas == 8  # 4 + 2 + 2

    def test_layakari_dugun(self):
        base = ["dha", "dhin"]
        dugun = layakari_subdivision(base, 2)
        assert len(dugun) == 4
        assert dugun == ["dha", "dha", "dhin", "dhin"]

    def test_tihai(self):
        phrase = ["dha", "ge", "na"]
        t = tihai_pattern(phrase, rest_beats=1)
        assert len(t) == 11  # 3 + 1 + 3 + 1 + 3

    def test_complexity_symmetric_low(self):
        t = HINDUSTANI_TALAS["teental"]  # 4+4+4+4, symmetric
        c = tala_complexity(t)
        assert c < 0.1  # low asymmetry


# ── Ethnomusicology: Arabic modes ──
from models.music_ethnomusicology.arabic_modes import (
    AJNAS, COMMON_MAQAMAT, modulation_distance, common_jins,
    is_quarter_tone_maqam, quartertone_to_cents, sayr,
)

class TestArabicModes:
    def test_hijaz_augmented_second(self):
        h = AJNAS["hijaz"]
        assert 6 in h.intervals  # aug 2nd = 6 quarter-tones

    def test_bayati_quarter_tones(self):
        assert is_quarter_tone_maqam("bayati")
        assert is_quarter_tone_maqam("rast")

    def test_ajam_no_quarter_tones(self):
        assert not is_quarter_tone_maqam("ajam")

    def test_modulation_distance_self(self):
        assert modulation_distance("rast", "rast") == 0

    def test_modulation_distance_one(self):
        assert modulation_distance("bayati", "saba") >= 1

    def test_common_jins(self):
        shared = common_jins("rast", "sikah")
        assert "rast" in shared  # both use rast jins

    def test_quartertone_to_cents(self):
        assert quartertone_to_cents(4) == 200.0  # whole tone

    def test_sayr_ascending(self):
        s = sayr("rast")
        assert s["ascending"][0] == 1


# ── Timbre: spectral envelope ──
from models.music_timbre.spectral_envelope import (
    autocorrelation, levinson_durbin, lpc_envelope,
    cepstral_smoothing, true_envelope, spectral_tilt,
)

class TestSpectralEnvelope:
    def test_autocorrelation_peak_at_zero(self):
        x = np.random.randn(256)
        r = autocorrelation(x, 10)
        assert r[0] >= r[1]

    def test_levinson_durbin_returns_coeffs(self):
        x = np.sin(2 * np.pi * 440 * np.arange(512) / 8000)
        r = autocorrelation(x, 8)
        coeffs, e = levinson_durbin(r, 8)
        assert len(coeffs) == 8
        assert e >= 0

    def test_lpc_envelope_shape(self):
        x = np.sin(2 * np.pi * 440 * np.arange(1024) / 16000)
        env = lpc_envelope(x, order=12, n_fft=256)
        assert len(env) == 129  # 256//2 + 1
        assert np.all(env >= 0)

    def test_cepstral_smoothing_preserves_shape(self):
        log_spec = np.random.randn(129)
        smoothed = cepstral_smoothing(log_spec, n_keep=10)
        assert len(smoothed) == len(log_spec)

    def test_spectral_tilt_decreasing(self):
        freqs = np.linspace(1, 8000, 128)
        log_spec = -3.0 * np.log2(freqs / freqs[0])  # -3dB/octave
        tilt = spectral_tilt(log_spec, sr=16000)
        assert tilt < 0


# ── Timbre: attack transient ──
from models.music_timbre.attack_transient import (
    compute_attack_time, rise_slope, classify_attack, analyze_transient,
    AttackType,
)

class TestAttackTransient:
    def test_percussive_attack_fast(self):
        sr = 44100
        t = np.arange(0, 0.01, 1/sr)
        signal = np.exp(-t * 500)  # very fast decay
        atk = compute_attack_time(np.abs(signal), sr)
        assert atk < 5.0

    def test_classify_percussive(self):
        assert classify_attack(2.0, 0.6) == AttackType.PERCUSSIVE

    def test_classify_bowed(self):
        assert classify_attack(50.0, 0.2) == AttackType.BOWED

    def test_classify_soft(self):
        assert classify_attack(200.0, 0.1) == AttackType.SOFT

    def test_analyze_transient_returns_descriptor(self):
        sr = 44100
        t = np.arange(0, 0.05, 1/sr)
        signal = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 50)
        desc = analyze_transient(signal, sr)
        assert desc.peak_amplitude > 0
        assert desc.spectral_centroid_hz > 0


# ── Production: sidechain ──
from models.music_production.sidechain import (
    SidechainParams, envelope_follower, apply_sidechain, ducking_curve,
    pumping_rate, db_to_linear,
)

class TestSidechain:
    def test_envelope_follows_peak(self):
        sr = 44100
        signal = np.zeros(sr)
        signal[sr//2] = 1.0  # impulse at midpoint
        env = envelope_follower(signal, sr, attack_ms=1.0, release_ms=100.0)
        assert np.max(env) > 0

    def test_sidechain_reduces_target(self):
        sr = 44100
        n = sr  # 1 second
        target = np.ones(n) * 0.5
        kick = np.zeros(n)
        kick[0:100] = 1.0  # loud kick at start
        params = SidechainParams(threshold_db=-6, ratio=8)
        output = apply_sidechain(target, kick, sr, params)
        assert np.mean(output[:200]) < np.mean(target[:200])

    def test_ducking_curve_shape(self):
        sr = 44100
        signal = np.zeros(sr)
        signal[1000:2000] = 0.8
        curve = ducking_curve(signal, sr, depth_db=-12.0, threshold_db=-10.0)
        assert np.min(curve) < 1.0

    def test_pumping_rate_with_kicks(self):
        sr = 44100
        signal = np.zeros(sr * 2)
        for beat in range(8):
            idx = int(beat * sr / 4)
            if idx + 100 < len(signal):
                signal[idx:idx+100] = 0.9
        rate = pumping_rate(signal, sr, threshold_db=-10.0)
        assert rate > 1.0


# ── Production: stereo imaging ──
from models.music_production.stereo_imaging import (
    stereo_correlation, stereo_width, stereo_balance,
    mono_compatibility, analyze_stereo, to_mid_side, from_mid_side,
    widen_stereo,
)

class TestStereoImaging:
    def test_mono_signal_correlation_one(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100)
        assert stereo_correlation(sig, sig) == pytest.approx(1.0, abs=0.01)

    def test_mono_signal_zero_width(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100)
        assert stereo_width(sig, sig) == pytest.approx(0.0, abs=0.01)

    def test_opposite_phase_max_width(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100)
        w = stereo_width(sig, -sig)
        assert w > 0.9

    def test_balance_centered(self):
        sig = np.random.randn(1000)
        b = stereo_balance(sig, sig)
        assert abs(b) < 0.01

    def test_mono_compatibility_perfect(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100)
        assert mono_compatibility(sig, sig) == pytest.approx(1.0, abs=0.01)

    def test_mid_side_roundtrip(self):
        left = np.random.randn(100)
        right = np.random.randn(100)
        mid, side = to_mid_side(left, right)
        l2, r2 = from_mid_side(mid, side)
        np.testing.assert_allclose(l2, left, atol=1e-10)
        np.testing.assert_allclose(r2, right, atol=1e-10)

    def test_widen_increases_width(self):
        left = np.random.randn(1000)
        right = np.random.randn(1000) * 0.5 + left * 0.5
        orig_width = stereo_width(left, right)
        l2, r2 = widen_stereo(left, right, factor=2.0)
        new_width = stereo_width(l2, r2)
        assert new_width > orig_width

    def test_analyze_returns_all_fields(self):
        left = np.random.randn(1000)
        right = np.random.randn(1000)
        a = analyze_stereo(left, right)
        assert 0 <= a.width <= 1
        assert -1 <= a.correlation <= 1
        assert -1 <= a.balance <= 1
