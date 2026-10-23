import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import ICRS, TETE, SkyCoord
from astropy.time import Time

from kino.observing import (
    GMT,
    SOAR,
    airmass_from_altitude,
    moon_state,
    night_window,
    seeing_at,
    target_track,
)


class TestSite:
    def test_soar_coordinates_match_the_published_geodetic_position(self):
        # Mamajek (2013), arXiv:1210.1616v3, Table 6
        assert SOAR.latitude.deg == pytest.approx(-(30 + 14 / 60 + 16.41 / 3600), abs=1e-9)
        assert SOAR.longitude.deg == pytest.approx(-(70 + 44 / 60 + 1.11 / 3600), abs=1e-9)
        assert SOAR.location.height.to_value(u.m) == pytest.approx(2738.0, abs=0.01)

    def test_gmt_coordinates_match_the_published_geodetic_position(self):
        # GMTO's own site page, https://giantmagellan.org/location/, retrieved 2026-10-23
        assert GMT.latitude.deg == pytest.approx(-(29 + 2 / 60 + 54.0 / 3600), abs=1e-9)
        assert GMT.longitude.deg == pytest.approx(-(70 + 41 / 60 + 1.0 / 3600), abs=1e-9)
        assert GMT.location.height.to_value(u.m) == pytest.approx(2514.0, abs=0.01)


class TestNightWindow:
    def test_window_is_dark_ordered_and_bounded_by_the_twilight_altitude(self):
        night = night_window(SOAR, "2026-10-03")
        assert np.all(night.sun_altitude < -18 * u.deg)
        assert np.all(np.diff(night.times.jd) > 0)
        # the grid stops within one step of the -18 deg crossing
        assert night.sun_altitude[0].to_value(u.deg) > -19.5
        assert night.sun_altitude[-1].to_value(u.deg) > -19.5

    def test_winter_night_is_longer_than_summer_night_in_the_south(self):
        winter = night_window(SOAR, "2026-06-21").duration
        summer = night_window(SOAR, "2026-12-21").duration
        assert winter > summer + 2 * u.hour

    def test_night_spans_the_ut_date_boundary(self):
        night = night_window(SOAR, "2026-10-03")
        assert night.start.iso[:10] != night.end.iso[:10] or night.start.mjd % 1 > 0.0
        # local evening of 3 Oct is after midnight UT on 4 Oct at this longitude
        assert night.start.iso[:10] == "2026-10-04"

    def test_looser_twilight_gives_a_longer_night(self):
        astro = night_window(SOAR, "2026-10-03")
        nautical = night_window(SOAR, "2026-10-03", sun_altitude_max=-12 * u.deg)
        assert nautical.duration > astro.duration

    def test_impossible_threshold_raises(self):
        with pytest.raises(ValueError, match="never drops"):
            night_window(SOAR, "2026-10-03", sun_altitude_max=-89 * u.deg)


class TestAirmass:
    def test_zenith_is_unity(self):
        assert airmass_from_altitude(90 * u.deg)[0] == pytest.approx(1.0, abs=2e-3)

    @pytest.mark.parametrize("alt, expected", [(45.0, 1.414), (30.0, 1.995)])
    def test_close_to_sec_z_at_moderate_zenith_distance(self, alt, expected):
        assert airmass_from_altitude(alt * u.deg)[0] == pytest.approx(expected, abs=0.01)

    def test_finite_at_the_horizon_and_monotonic(self):
        alts = np.array([1.0, 5.0, 10.0, 30.0, 60.0, 90.0]) * u.deg
        x = airmass_from_altitude(alts)
        assert np.all(np.isfinite(x))
        assert np.all(np.diff(x) < 0)
        assert x[0] < 45  # sec z would give ~57

    def test_below_horizon_is_infinite(self):
        assert np.isinf(airmass_from_altitude(-5 * u.deg)[0])


class TestTargetTrack:
    def _track(self, dec):
        night = night_window(SOAR, "2026-10-03")
        mid = night.times[len(night.times) // 2]
        # a target whose *of-date* RA equals the local sidereal time at `mid`
        # transits exactly at `mid`; express it in ICRS for the API
        lst = mid.sidereal_time("apparent", longitude=SOAR.longitude)
        coord = SkyCoord(ra=lst, dec=dec, frame=TETE(obstime=mid)).transform_to(ICRS())
        times = mid + np.linspace(-4, 4, 161) * u.hour
        return coord, times, target_track(SOAR, coord, times)

    def test_target_at_the_site_latitude_transits_at_the_zenith(self):
        _, _, track = self._track(SOAR.latitude)
        assert track.altitude.max().to_value(u.deg) == pytest.approx(90.0, abs=0.5)
        assert track.airmass.min() == pytest.approx(1.0, abs=0.01)

    def test_maximum_altitude_is_ninety_minus_the_latitude_offset(self):
        dec = -50 * u.deg
        _, _, track = self._track(dec)
        expected = 90.0 - abs(SOAR.latitude.deg - dec.value)
        assert track.altitude.max().to_value(u.deg) == pytest.approx(expected, abs=0.5)

    def test_airmass_is_symmetric_about_transit(self):
        _, times, track = self._track(-45 * u.deg)
        # the fixture puts transit exactly at the middle sample
        i = len(times) // 2
        assert track.airmass[i] == track.airmass.min()
        assert track.airmass[i - 30] == pytest.approx(track.airmass[i + 30], rel=1e-3)

    def test_parallactic_angle_flips_sign_across_the_meridian(self):
        _, times, track = self._track(-45 * u.deg)
        i = len(times) // 2
        assert abs(track.parallactic_angle[i].to_value(u.deg)) < 2.0
        assert track.parallactic_angle[i - 40].value * track.parallactic_angle[i + 40].value < 0
        # symmetric magnitude either side
        assert abs(track.parallactic_angle[i - 40].value) == pytest.approx(
            abs(track.parallactic_angle[i + 40].value), rel=0.05
        )

    def test_below_horizon_targets_have_infinite_airmass(self):
        night = night_window(SOAR, "2026-10-03")
        coord = SkyCoord(ra=0 * u.deg, dec=89 * u.deg)  # never rises from -30 lat
        track = target_track(SOAR, coord, night.times)
        assert np.all(np.isinf(track.airmass))


class TestMoon:
    def test_full_and_new_moon_illumination(self):
        # Full moon 2025-01-13 ~22:27 UT, new moon 2025-01-29 ~12:36 UT
        full = moon_state(SOAR, Time("2025-01-13 22:27:00"))
        new = moon_state(SOAR, Time("2025-01-29 12:36:00"))
        assert full.illumination[()] > 0.99
        assert new.illumination[()] < 0.01
        assert full.phase_angle.to_value(u.deg) < 12
        assert new.phase_angle.to_value(u.deg) > 168

    def test_illumination_is_a_fraction(self):
        times = Time("2026-10-01") + np.linspace(0, 30, 60) * u.day
        ill = moon_state(SOAR, times).illumination
        assert np.all((ill >= 0) & (ill <= 1))
        assert ill.max() > 0.95
        assert ill.min() < 0.05

    def test_separation_from_the_moon_itself_is_zero(self):
        from astropy.coordinates import get_body

        t = Time("2026-10-03 03:00:00")
        moon = get_body("moon", t, location=SOAR.location)
        coord = SkyCoord(ra=moon.ra, dec=moon.dec)
        assert moon_state(SOAR, t, coord).separation.to_value(u.deg) < 0.05

    def test_separation_is_a_valid_angle_and_optional(self):
        t = Time("2026-10-03 03:00:00")
        assert moon_state(SOAR, t).separation is None
        sep = moon_state(SOAR, t, SkyCoord(ra=10 * u.deg, dec=-30 * u.deg)).separation
        assert 0 <= sep.to_value(u.deg) <= 180


class TestSeeingAt:
    def test_unchanged_at_zenith_and_reference_wavelength(self):
        assert seeing_at(0.8 * u.arcsec, 1.0, 500 * u.nm).to_value(u.arcsec) == pytest.approx(0.8)

    def test_airmass_scaling_is_three_fifths(self):
        s = seeing_at(1.0 * u.arcsec, 2.0, 500 * u.nm).to_value(u.arcsec)
        assert s == pytest.approx(2.0**0.6)

    def test_seeing_improves_toward_the_red(self):
        blue = seeing_at(1.0 * u.arcsec, 1.0, 400 * u.nm)
        red = seeing_at(1.0 * u.arcsec, 1.0, 900 * u.nm)
        assert red < blue
        assert red.to_value(u.arcsec) == pytest.approx((900 / 500) ** -0.2)

    def test_array_airmass(self):
        out = seeing_at(1.0 * u.arcsec, np.array([1.0, 1.5, 2.0]), 650 * u.nm)
        assert out.shape == (3,)
        assert np.all(np.diff(out.value) > 0)

    def test_rejects_unphysical_inputs(self):
        with pytest.raises(ValueError):
            seeing_at(1.0 * u.arcsec, 0.9, 500 * u.nm)
        with pytest.raises(ValueError):
            seeing_at(0.0 * u.arcsec, 1.0, 500 * u.nm)
