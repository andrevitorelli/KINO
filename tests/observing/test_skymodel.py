from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from kino.instruments.ifu import IFU
from kino.observing import sky as ks91
from kino.observing.extinction import extinction_coefficient
from kino.observing.skymodel import (
    SkyConditions,
    SkyLibrary,
    bin_to_channels,
    ifu_sky_electron_rate,
    sky_electron_rate,
)

LIBRARY_DIR = Path("data/skycalc")
needs_library = pytest.mark.skipif(
    not (LIBRARY_DIR / "moon.npz").exists(),
    reason="sky library not built (scripts/build_skycalc_library.py)",
)

LAM = np.linspace(300.0, 1000.0, 701)  # 1 nm


def make_library(moon_fn=None, glow_fn=None, zodi_value=2.0, star_value=1.0) -> SkyLibrary:
    """A tiny synthetic library whose interpolation behaviour is known exactly."""
    elong = np.array([20.0, 100.0, 180.0])
    rho = np.array([10.0, 90.0, 170.0])
    zm = np.array([0.0, 40.0, 80.0])
    x = np.array([1.0, 2.0])
    moon_fn = moon_fn or (lambda e, r, z, a: 1000.0 * 10.0 ** (e / 180.0) * 10.0 ** (r / 170.0))
    grid = np.array([[[[moon_fn(e, r, z, a) for a in x] for z in zm] for r in rho] for e in elong])
    moon = {
        "lam_nm": LAM, "elongation": elong, "target_sep": rho, "zenith_dist_moon": zm,
        "airmass": x, "radiance": grid[..., None] * np.ones_like(LAM),
    }  # fmt: skip
    glow_fn = glow_fn or (lambda a, s: 10.0 * a * s / 100.0)
    ax, sf = np.array([1.0, 2.5]), np.array([70.0, 200.0])
    glow = np.array([[glow_fn(a, s) for s in sf] for a in ax])[..., None] * np.ones_like(LAM)
    airglow = {"lam_nm": LAM, "airmass": ax, "solar_flux": sf, "radiance": glow}
    lon, lat = np.array([30.0, 180.0]), np.array([0.0, 90.0])
    zodi = {
        "lam_nm": LAM, "ecl_lon": lon, "ecl_lat": lat,
        "radiance": zodi_value * np.ones((2, 2, len(LAM))),
        "starlight": star_value * np.ones(len(LAM)),
    }  # fmt: skip
    return SkyLibrary(moon, airglow, zodi, meta={"synthetic": True})


class TestBinning:
    def test_constant_spectrum(self):
        out = bin_to_channels(LAM, np.full(LAM.shape, 3.0), np.array([400.0, 450.0, 520.5]))
        assert out == pytest.approx([3.0, 3.0])

    def test_linear_ramp_gives_the_midpoint_value(self):
        out = bin_to_channels(LAM, 2.0 * LAM, np.array([400.0, 410.0, 430.0]))
        assert out == pytest.approx([2 * 405.0, 2 * 420.0])

    def test_narrow_line_flux_is_conserved_in_a_wide_channel(self):
        fine = np.linspace(500.0, 510.0, 10001)
        rad = np.exp(-0.5 * ((fine - 505.0) / 0.01) ** 2)  # unresolved line, area 0.01*sqrt(2 pi)
        area = np.trapezoid(rad, fine)
        mean = bin_to_channels(fine, rad, np.array([500.0, 510.0]))
        assert mean[0] * 10.0 == pytest.approx(area, rel=1e-6)

    def test_channels_outside_the_library_raise(self):
        with pytest.raises(ValueError, match="outside the sky library"):
            bin_to_channels(LAM, np.ones_like(LAM), np.array([290.0, 400.0]))


class TestMoonlight:
    def _c(self, **kw):
        base = dict(
            airmass=1.0, solar_flux=100.0, moon_phase_angle=0 * u.deg,
            moon_target_separation=90 * u.deg, moon_zenith_distance=40 * u.deg,
        )  # fmt: skip
        base.update(kw)
        return SkyConditions(**base)

    def test_exact_at_a_grid_node(self):
        lib = make_library()
        # airmass 2 -> z = 60 deg; with zm = 40 the valid separations are 20-100 deg, so 90 stands
        got = lib.moonlight(self._c(airmass=2.0))[0]  # elong 180, rho 90, zm 40, X 2
        assert got == pytest.approx(1000.0 * 10.0 ** (180 / 180) * 10.0 ** (90 / 170), rel=1e-6)

    def test_log_linear_between_nodes(self):
        lib = make_library()
        mid = lib.moonlight(self._c(moon_phase_angle=40 * u.deg))[
            0
        ]  # elong 140: between 100 and 180
        lo = lib.moonlight(self._c(moon_phase_angle=80 * u.deg))[0]  # elong 100
        hi = lib.moonlight(self._c(moon_phase_angle=0 * u.deg))[0]  # elong 180
        assert mid == pytest.approx(np.sqrt(lo * hi), rel=1e-6)  # log-linear -> geometric mean

    def test_zero_when_the_moon_is_down(self):
        lib = make_library()
        assert np.all(lib.moonlight(self._c(moon_zenith_distance=90 * u.deg)) == 0.0)
        assert np.all(lib.moonlight(self._c(moon_zenith_distance=110 * u.deg)) == 0.0)

    def test_tapers_to_zero_between_80_degrees_and_the_horizon(self):
        lib = make_library()
        at80 = lib.moonlight(self._c(moon_zenith_distance=80 * u.deg))[0]
        at85 = lib.moonlight(self._c(moon_zenith_distance=85 * u.deg))[0]
        at89 = lib.moonlight(self._c(moon_zenith_distance=89 * u.deg))[0]
        assert at80 > at85 > at89 > 0
        assert at85 == pytest.approx(0.5 * at80, rel=1e-6)

    def test_fainter_toward_new_moon(self):
        lib = make_library()
        full = lib.moonlight(self._c(moon_phase_angle=0 * u.deg))[0]
        crescent = lib.moonlight(self._c(moon_phase_angle=150 * u.deg))[0]
        assert full > crescent

    def test_separation_outside_the_geometric_range_is_moved_into_it(self):
        lib = make_library()
        # airmass 1 -> z = 0; with zm = 40 the only valid separation is 40 deg
        a = lib.moonlight(self._c(moon_target_separation=170 * u.deg))
        b = lib.moonlight(self._c(moon_target_separation=40 * u.deg))
        assert a == pytest.approx(b)

    def test_airmass_above_the_grid_warns_and_clips(self):
        lib = make_library()
        with pytest.warns(UserWarning, match="beyond the moonlight grid"):
            far = lib.moonlight(self._c(airmass=3.0))
        assert far == pytest.approx(lib.moonlight(self._c(airmass=2.0)))


class TestOtherComponents:
    def test_airglow_is_bilinear_in_airmass_and_solar_flux(self):
        lib = make_library()
        c = SkyConditions(airmass=1.75, solar_flux=135.0)
        assert lib.airglow(c)[0] == pytest.approx(10.0 * 1.75 * 135.0 / 100.0, rel=1e-9)

    def test_airglow_rises_with_solar_flux(self):
        lib = make_library()
        lo = lib.airglow(SkyConditions(airmass=1.2, solar_flux=80.0))[0]
        hi = lib.airglow(SkyConditions(airmass=1.2, solar_flux=180.0))[0]
        assert hi > lo

    def test_zodiacal_is_symmetric_in_ecliptic_longitude(self):
        lib = make_library(zodi_value=5.0)
        east = lib.zodiacal(SkyConditions(1.0, 100.0, ecliptic_lon_from_sun=60 * u.deg))
        west = lib.zodiacal(SkyConditions(1.0, 100.0, ecliptic_lon_from_sun=-60 * u.deg))
        assert east == pytest.approx(west)

    def test_zodiacal_and_starlight_are_extincted_with_airmass(self):
        lib = make_library(zodi_value=5.0, star_value=2.0)
        z1 = lib.zodiacal(SkyConditions(1.0, 100.0))
        z2 = lib.zodiacal(SkyConditions(2.0, 100.0))
        k = extinction_coefficient(np.clip(LAM, 320.0, None) * u.nm)
        assert z2 == pytest.approx(z1 * 10.0 ** (-0.4 * k))
        s1 = lib.starlight(SkyConditions(1.0, 100.0))
        s2 = lib.starlight(SkyConditions(1.5, 100.0))
        assert np.all(s2 < s1)


class TestSpectrum:
    def test_components_and_total(self):
        lib = make_library()
        c = SkyConditions(1.0, 100.0, moon_phase_angle=0 * u.deg, moon_zenith_distance=40 * u.deg)
        sp = lib.spectrum(c, np.linspace(500, 700, 21) * u.nm)
        assert set(sp.components) == {"moon", "airglow", "zodiacal", "starlight"}
        assert sp.total == pytest.approx(sum(sp.components.values()))
        assert len(sp.total) == 20
        assert sp.channel_width.to_value(u.nm) == pytest.approx(10.0)

    def test_rejects_channels_below_320_nm_and_bad_airmass(self):
        lib = make_library()
        with pytest.raises(ValueError, match="320"):
            lib.spectrum(SkyConditions(1.0, 100.0), np.array([310.0, 330.0]) * u.nm)
        with pytest.raises(ValueError, match="airmass"):
            lib.spectrum(SkyConditions(0.9, 100.0), np.array([500.0, 510.0]) * u.nm)


class TestElectronRate:
    def test_matches_the_closed_form_for_flat_radiance_and_throughput(self):
        lib = make_library(
            moon_fn=lambda *a: 0.0, glow_fn=lambda a, s: 100.0, zodi_value=0.0, star_value=0.0
        )
        ifu = IFU.from_config("kino/config/sifs.yaml")
        rate = ifu_sky_electron_rate(ifu, SkyConditions(1.0, 100.0), lib)
        assert rate.shape == (ifu.n_wave,)
        assert rate.unit == u.electron / u.s
        width_um = ((ifu.wavelength_max - ifu.wavelength_min) / ifu.n_wave).to_value(u.um)
        centres = 0.5 * (
            np.linspace(ifu.wavelength_min, ifu.wavelength_max, ifu.n_wave + 1)[:-1]
            + np.linspace(ifu.wavelength_min, ifu.wavelength_max, ifu.n_wave + 1)[1:]
        )
        expected = (
            100.0 * width_um * ifu.pixel_scale.to_value(u.arcsec) ** 2
            * ifu.effective_area.to_value(u.m**2) * ifu.throughput(centres)
        )  # fmt: skip
        assert rate.value == pytest.approx(expected, rel=1e-6)

    def test_scales_linearly_with_area_and_throughput(self):
        lib = make_library(
            moon_fn=lambda *a: 0.0, glow_fn=lambda a, s: 50.0, zodi_value=0.0, star_value=0.0
        )
        edges = np.linspace(600, 700, 11) * u.nm
        sp = lib.spectrum(SkyConditions(1.0, 100.0), edges)
        flat = lambda w: np.full(len(w), 0.1)  # noqa: E731
        one = sky_electron_rate(
            sp, pixel_area=1 * u.arcsec**2, effective_area=10 * u.m**2, throughput=flat
        )
        two = sky_electron_rate(
            sp, pixel_area=2 * u.arcsec**2, effective_area=10 * u.m**2, throughput=flat
        )
        assert two.value == pytest.approx(2 * one.value)

    def test_can_drive_an_ifu_observation(self):
        import dataclasses

        lib = make_library(glow_fn=lambda a, s: 200.0)
        ifu = IFU.from_config("kino/config/sifs.yaml")
        ifu = dataclasses.replace(
            ifu, sky_electron_rate=ifu_sky_electron_rate(ifu, SkyConditions(1.0, 100.0), lib)
        )
        from kino.galaxy import DiscGalaxy

        gal = DiscGalaxy(r_e=1.5, inc=55.0, pa=30.0, v_flat=180.0, z=0.05, sigma0=30.0)
        cube = ifu.observe(
            gal, line_flux=1e-14 * u.erg / u.s / u.cm**2, line_wavelength=0.65628 * u.um
        )
        assert cube.flux.min() > 0  # the sky floor is present in every channel


class TestLoad:
    def test_missing_library_gives_a_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_skycalc_library"):
            SkyLibrary.load(tmp_path)

    def test_round_trip_through_npz(self, tmp_path):
        lib = make_library()
        for name, obj in (
            ("moon", dict(lam_nm=LAM, elongation=[20.0, 180.0], target_sep=[10.0, 170.0],
                          zenith_dist_moon=[0.0, 80.0], airmass=[1.0, 2.0],
                          radiance=np.ones((2, 2, 2, 2, len(LAM))))),
            ("airglow", dict(lam_nm=LAM, airmass=[1.0, 2.5], solar_flux=[70.0, 200.0],
                             radiance=np.ones((2, 2, len(LAM))))),
            ("zodi", dict(lam_nm=LAM, ecl_lon=[30.0, 180.0], ecl_lat=[0.0, 90.0],
                          radiance=np.ones((2, 2, len(LAM))), starlight=np.ones(len(LAM)))),
        ):  # fmt: skip
            np.savez(tmp_path / f"{name}.npz", **obj)
        loaded = SkyLibrary.load(tmp_path)
        c = SkyConditions(1.0, 100.0, moon_zenith_distance=30 * u.deg)
        assert loaded.moonlight(c)[0] == pytest.approx(1.0)
        assert lib.meta == {"synthetic": True}


# ---- checks against the real library and independent references --------------


def _v_mag(radiance_mean):
    # Vega V = 0 <-> 3.631e-9 erg/s/cm2/A at 550 nm, in ph/s/m2/micron
    zp = 3.631e-9 * 1e-7 * 1e4 * 1e4 / (6.62607e-34 * 2.99792458e8 / 550e-9)
    return -2.5 * np.log10(radiance_mean / zp)


@needs_library
class TestAgainstIndependentReferences:
    @pytest.fixture(scope="class")
    def lib(self):
        return SkyLibrary.load(LIBRARY_DIR)

    def _v(self, lib, cond):
        sp = lib.spectrum(cond, np.array([505.0, 595.0]) * u.nm)  # crude V band
        return _v_mag(sp.total[0]), _v_mag(sp.components["moon"][0] + 1e-9)

    def test_dark_zenith_sky_matches_the_ctio_mean(self, lib):
        dark = SkyConditions(1.0, 100.0)  # Moon down by default
        v, _ = self._v(lib, dark)
        mean, scatter = ks91.CTIO_DARK_SKY["V"]
        assert v == pytest.approx(mean, abs=2 * scatter)

    @pytest.mark.parametrize(
        "alpha, rho, zm", [(30, 60, 50), (90, 60, 50), (0, 60, 50), (0, 30, 40)]
    )
    def test_moonlight_brightening_agrees_with_krisciunas_and_schaefer(self, lib, alpha, rho, zm):
        z = 33.6
        x = 1.0 / np.cos(np.radians(z))
        base = SkyConditions(x, 100.0)
        moon = SkyConditions(
            x, 100.0, moon_phase_angle=alpha * u.deg, moon_target_separation=rho * u.deg,
            moon_zenith_distance=zm * u.deg,
        )  # fmt: skip
        dv = self._v(lib, moon)[0] - self._v(lib, base)[0]
        k = float(extinction_coefficient(5556 * u.AA)[0])
        ref = float(
            ks91.sky_brightness_v(21.79, alpha * u.deg, rho * u.deg, zm * u.deg, z * u.deg, k)[
                "delta_v"
            ]
        )
        assert dv == pytest.approx(ref, abs=0.5)

    def test_airglow_continuum_rises_with_solar_activity(self, lib):
        lo = lib.airglow(SkyConditions(1.0, 70.0))
        hi = lib.airglow(SkyConditions(1.0, 200.0))
        assert hi.mean() > 1.3 * lo.mean()

    def test_the_oxygen_green_line_is_present_and_narrow(self, lib):
        sp = lib.airglow(SkyConditions(1.0, 100.0))
        lam = lib._glow_lam
        peak = lam[np.argmax(np.where((lam > 556) & (lam < 559), sp, 0.0))]
        assert 557.5 < peak < 558.1  # [OI] 5577 A (vacuum 557.73 nm)
