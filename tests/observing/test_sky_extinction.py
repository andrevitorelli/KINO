import astropy.units as u
import numpy as np
import pytest

from kino.observing.extinction import (
    CTIO_EXTINCTION_MAG_PER_AIRMASS,
    CTIO_WAVELENGTH_ANGSTROM,
    atmospheric_transmission,
    extinction_coefficient,
)
from kino.observing.sky import (
    CTIO_DARK_SKY,
    dark_sky_brightness,
    mag_from_nanolamberts,
    moonlight_brightness,
    nanolamberts_from_mag,
    sky_brightness_v,
)

K_V = 0.14  # CTIO extinction at ~5556 A


class TestUnitConversion:
    def test_matches_the_dark_sky_value_quoted_by_krisciunas_and_schaefer(self):
        # K&S91: V = 21.587 mag/arcsec^2 <-> B_zen = 79.0 nL
        assert nanolamberts_from_mag(21.587) == pytest.approx(79.0, abs=0.1)

    def test_s10_unit(self):
        # a V = 10 star spread over one square degree: 0.263 nL, i.e. 27.78151 mag/arcsec^2
        assert nanolamberts_from_mag(27.78151) == pytest.approx(0.263)

    def test_round_trip_and_monotonic(self):
        mags = np.array([17.0, 19.0, 21.0, 22.5])
        assert mag_from_nanolamberts(nanolamberts_from_mag(mags)) == pytest.approx(mags)
        assert np.all(np.diff(nanolamberts_from_mag(mags)) < 0)  # fainter mag -> fewer nL


class TestDarkSky:
    def test_ctio_table_values(self):
        assert CTIO_DARK_SKY["V"] == (21.79, 0.22)
        assert CTIO_DARK_SKY["I"][0] == pytest.approx(19.85)
        # sky is bluer than the Sun-like default: I band is much brighter than B
        assert CTIO_DARK_SKY["I"][0] < CTIO_DARK_SKY["V"][0] < CTIO_DARK_SKY["B"][0]

    def test_equals_the_zenith_value_at_the_zenith(self):
        b = dark_sky_brightness(21.79, 0 * u.deg, K_V)
        assert b == pytest.approx(nanolamberts_from_mag(21.79))

    def test_brightens_toward_the_horizon_but_stays_finite(self):
        z = np.array([0, 30, 60, 80, 89.9]) * u.deg
        b = dark_sky_brightness(21.79, z, K_V)
        assert np.all(np.diff(b) > 0)
        assert np.all(np.isfinite(b))
        assert b[-1] < 6 * b[0]  # X -> 5 at the horizon, partly offset by extinction


class TestMoonlight:
    def _b(self, alpha=0, rho=60, zm=30, z=30):
        return moonlight_brightness(alpha * u.deg, rho * u.deg, zm * u.deg, z * u.deg, K_V)

    def test_zero_when_the_moon_is_below_the_horizon(self):
        assert self._b(zm=95) == 0.0
        assert self._b(zm=90.0) == 0.0
        assert self._b(zm=89.0) > 0.0

    def test_full_moon_is_far_brighter_than_the_dark_sky(self):
        assert self._b(alpha=0) > 5 * nanolamberts_from_mag(21.79)

    def test_new_moon_is_negligible(self):
        assert self._b(alpha=170) < 0.02 * nanolamberts_from_mag(21.79)

    def test_brightness_falls_with_phase_angle(self):
        b = np.array([self._b(alpha=a) for a in (0, 30, 60, 90, 120, 150)])
        assert np.all(np.diff(b) < 0)

    def test_brightness_falls_with_separation_out_to_90_degrees(self):
        b = np.array([self._b(rho=r) for r in (10, 20, 40, 60, 90)])
        assert np.all(np.diff(b) < 0)

    def test_the_aureole_branch_is_roughly_continuous_at_ten_degrees(self):
        below, above = self._b(rho=9.999), self._b(rho=10.001)
        assert below == pytest.approx(above, rel=0.4)
        assert self._b(rho=2.0) > self._b(rho=10.0)

    def test_lower_moon_gives_less_light_through_more_air(self):
        assert self._b(zm=80) < self._b(zm=20)

    def test_matches_the_closed_form_at_one_configuration(self):
        # alpha=0, rho=90, Zm=Z=0 (X=1): B = f * I* * 10^(-0.4k) * (1 - 10^(-0.4k))
        f = 10**5.36 * 1.06 + 10 ** (6.15 - 90 / 40)
        i_star = 10 ** (-0.4 * 3.84)
        expected = f * i_star * 10 ** (-0.4 * K_V) * (1 - 10 ** (-0.4 * K_V))
        assert self._b(alpha=0, rho=90, zm=0, z=0) == pytest.approx(expected)

    def test_broadcasts(self):
        out = moonlight_brightness(
            np.array([0.0, 60.0]) * u.deg, 60 * u.deg, 30 * u.deg, 30 * u.deg, K_V
        )
        assert out.shape == (2,)


class TestTotalSky:
    def test_total_is_dark_plus_moon_and_brighter(self):
        r = sky_brightness_v(21.79, 0 * u.deg, 60 * u.deg, 30 * u.deg, 30 * u.deg, K_V)
        assert r["total"] == pytest.approx(r["dark"] + r["moon"])
        assert r["total_mag"] < 21.79
        assert r["delta_v"] < 0

    def test_no_moon_leaves_the_dark_sky_unchanged(self):
        r = sky_brightness_v(21.79, 0 * u.deg, 60 * u.deg, 120 * u.deg, 30 * u.deg, K_V)
        assert r["moon"] == 0.0
        assert r["delta_v"] == pytest.approx(0.0)

    def test_quarter_moon_is_comparable_to_geminis_dark_bin(self):
        # Gemini South's 50%-ile ("dark") optical sky background: V > 20.7
        r = sky_brightness_v(21.79, 90 * u.deg, 60 * u.deg, 30 * u.deg, 30 * u.deg, K_V)
        assert r["total_mag"] == pytest.approx(20.7, abs=0.5)

    def test_full_moon_falls_in_geminis_bright_regime(self):
        r = sky_brightness_v(21.79, 0 * u.deg, 40 * u.deg, 30 * u.deg, 30 * u.deg, K_V)
        assert 17.0 < r["total_mag"] < 19.5


class TestExtinction:
    def test_table_is_ordered_and_matches_the_source_file_endpoints(self):
        assert np.all(np.diff(CTIO_WAVELENGTH_ANGSTROM) > 0)
        assert len(CTIO_WAVELENGTH_ANGSTROM) == len(CTIO_EXTINCTION_MAG_PER_AIRMASS)
        assert CTIO_EXTINCTION_MAG_PER_AIRMASS[0] == 1.05  # 3200 A
        assert CTIO_EXTINCTION_MAG_PER_AIRMASS[-1] == 0.037  # 10870 A

    def test_exact_at_nodes_and_interpolated_between(self):
        assert extinction_coefficient(5556 * u.AA)[0] == pytest.approx(0.14)
        assert extinction_coefficient(6790 * u.AA)[0] == pytest.approx(0.071)
        mid = extinction_coefficient(6613 * u.AA)[0]  # between 6436 (0.11) and 6790 (0.071)
        assert 0.071 < mid < 0.11

    def test_much_stronger_in_the_blue_than_the_red(self):
        blue, red = extinction_coefficient(np.array([3500.0, 8000.0]) * u.AA)
        assert blue > 5 * red

    def test_unit_conversion_and_out_of_range(self):
        assert extinction_coefficient(0.5556 * u.um)[0] == pytest.approx(0.14)
        with pytest.raises(ValueError):
            extinction_coefficient(3000 * u.AA)
        with pytest.raises(ValueError):
            extinction_coefficient(1.2 * u.um)

    def test_transmission(self):
        assert atmospheric_transmission(5556 * u.AA, 1.0)[0] == pytest.approx(10 ** (-0.4 * 0.14))
        t = atmospheric_transmission(5556 * u.AA, np.array([1.0, 1.5, 2.0]))
        assert np.all(np.diff(t) < 0)
        assert t[2] == pytest.approx(10 ** (-0.4 * 0.14 * 2.0))

    def test_infinite_airmass_gives_zero_transmission(self):
        assert atmospheric_transmission(5556 * u.AA, np.inf)[0] == 0.0

    def test_rejects_airmass_below_one(self):
        with pytest.raises(ValueError):
            atmospheric_transmission(5556 * u.AA, 0.8)
