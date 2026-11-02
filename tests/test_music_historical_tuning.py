"""Tests for historical tuning systems."""
import math

import numpy as np
import pytest

from models.music_historical.tuning_systems import (
    pythagorean_ratios, pythagorean_cents, pythagorean_comma,
    quarter_comma_meantone_ratios, meantone_cents,
    JUST_RATIOS_5LIMIT, just_intonation_cents, syntonic_comma,
    werckmeister_iii_ratios, werckmeister_iii_cents,
    compare_tunings, cents_to_frequency, beat_frequency, interval_purity,
)


class TestPythagorean:
    def test_unison_is_one(self):
        r = pythagorean_ratios()
        assert r[0] == pytest.approx(1.0)

    def test_fifth_is_pure(self):
        r = pythagorean_ratios()
        assert r[7] == pytest.approx(3.0 / 2.0, rel=1e-10)

    def test_octave_bounded(self):
        r = pythagorean_ratios()
        assert np.all(r >= 1.0)
        assert np.all(r < 2.0)

    def test_cents_unison_zero(self):
        c = pythagorean_cents()
        assert c[0] == pytest.approx(0.0, abs=0.01)

    def test_comma_approximately_24(self):
        assert pythagorean_comma() == pytest.approx(23.46, abs=0.1)


class TestMeantone:
    def test_unison_is_one(self):
        r = quarter_comma_meantone_ratios()
        assert r[0] == pytest.approx(1.0)

    def test_major_third_is_pure(self):
        r = quarter_comma_meantone_ratios()
        assert r[4] == pytest.approx(5.0 / 4.0, rel=1e-6)

    def test_octave_bounded(self):
        r = quarter_comma_meantone_ratios()
        assert np.all(r >= 1.0)
        assert np.all(r < 2.0)


class TestJustIntonation:
    def test_known_ratios(self):
        assert JUST_RATIOS_5LIMIT[0] == 1.0
        assert JUST_RATIOS_5LIMIT[7] == pytest.approx(3.0 / 2.0)
        assert JUST_RATIOS_5LIMIT[4] == pytest.approx(5.0 / 4.0)
        assert JUST_RATIOS_5LIMIT[5] == pytest.approx(4.0 / 3.0)

    def test_cents_unison_zero(self):
        c = just_intonation_cents()
        assert c[0] == pytest.approx(0.0, abs=0.01)

    def test_syntonic_comma(self):
        assert syntonic_comma() == pytest.approx(21.51, abs=0.1)


class TestWerckmeister:
    def test_unison_is_one(self):
        r = werckmeister_iii_ratios()
        assert r[0] == pytest.approx(1.0, abs=0.01)

    def test_octave_bounded(self):
        r = werckmeister_iii_ratios()
        assert np.all(r >= 0.99)
        assert np.all(r < 2.01)

    def test_not_equal_temperament(self):
        c = werckmeister_iii_cents()
        assert not np.allclose(c, 0.0, atol=0.5)


class TestComparison:
    def test_compare_same(self):
        c = pythagorean_cents()
        r = compare_tunings(c, c, "Pyth", "Pyth")
        assert r.max_deviation_cents == pytest.approx(0.0)
        assert r.rms_deviation_cents == pytest.approx(0.0)

    def test_compare_different(self):
        r = compare_tunings(pythagorean_cents(), meantone_cents(), "Pyth", "MT")
        assert r.max_deviation_cents > 0.0
        assert len(r.per_note_cents) == 12


class TestUtilities:
    def test_cents_to_frequency_a440(self):
        f = cents_to_frequency(440.0, 0, 0.0)
        assert f == pytest.approx(440.0)

    def test_beat_frequency(self):
        assert beat_frequency(440.0, 442.0) == pytest.approx(2.0)

    def test_interval_purity_perfect(self):
        assert interval_purity(3.0 / 2.0, 3.0 / 2.0) == pytest.approx(0.0)

    def test_interval_purity_nonzero(self):
        tet_fifth = 2.0 ** (7.0 / 12.0)
        p = interval_purity(tet_fifth, 3.0 / 2.0)
        assert p == pytest.approx(1.955, abs=0.01)
