import dataclasses

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from kino.analysis.forecast import ForecastTarget, forecast_ifu, narrowed_ifu
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU
from kino.observing import SOAR, SkyLibrary, night_window

Z = 0.05  # Halpha (vacuum 6564.61 A) at ~0.689 um


def flat_library(glow=20.0, max_nm=1300.0) -> SkyLibrary:
    LAM = np.linspace(300.0, max_nm, int(max_nm - 300.0) + 1)
    elong, rho = np.array([20.0, 180.0]), np.array([10.0, 170.0])
    zm, x = np.array([0.0, 80.0]), np.array([1.0, 2.0])
    moon = {
        "lam_nm": LAM, "elongation": elong, "target_sep": rho, "zenith_dist_moon": zm, "airmass": x,
        "radiance": np.full((2, 2, 2, 2, len(LAM)), 5.0),
    }  # fmt: skip
    airglow = {
        "lam_nm": LAM, "airmass": np.array([1.0, 2.5]), "solar_flux": np.array([70.0, 200.0]),
        "radiance": np.full((2, 2, len(LAM)), glow),
    }  # fmt: skip
    zodi = {
        "lam_nm": LAM, "ecl_lon": np.array([30.0, 180.0]), "ecl_lat": np.array([0.0, 90.0]),
        "radiance": np.full((2, 2, len(LAM)), 2.0), "starlight": np.full(len(LAM), 1.0),
    }  # fmt: skip
    return SkyLibrary(moon, airglow, zodi)


def make_ifu(**kw) -> IFU:
    base = dict(
        pixel_scale=0.1 * u.arcsec, nx=8, ny=8,
        wavelength_min=0.688 * u.um, wavelength_max=0.690 * u.um, n_wave=20,
        psf=PSF(seeing_fwhm=0.02 * u.arcsec), lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[0.5, 0.9] * u.um, transmission=[0.3, 0.3]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=3 * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s,
    )  # fmt: skip
    base.update(kw)
    return IFU(**base)


def make_target(flux=2e-15, **kw):
    gal = DiscGalaxy(r_e=0.25, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=Z, sigma0=25.0)
    return ForecastTarget(
        SkyCoord("02h00m00s", "-25d00m00s"), gal, flux * u.erg / u.s / u.cm**2, "Halpha", **kw
    )


@pytest.fixture(scope="module")
def night():
    return night_window(SOAR, "2026-10-03")


def run(night, **kw):
    ifu = kw.pop("ifu", make_ifu())
    target = kw.pop("target", make_target())
    library = kw.pop("library", flat_library())
    args = dict(
        g=(0.05, 0.0), seeing_ref=[0.02, 0.06] * u.arcsec, solar_flux=110.0,
        time_step=180 * u.min, window_kms=500.0,
    )  # fmt: skip
    args.update(kw)
    return forecast_ifu(ifu, target, night, library, **args)


class TestNarrowedIfu:
    def test_keeps_the_native_dispersion_and_centres_on_the_line(self):
        ifu = make_ifu(wavelength_min=0.6 * u.um, wavelength_max=0.8 * u.um, n_wave=4000)
        native = (ifu.wavelength_max - ifu.wavelength_min) / ifu.n_wave
        out = narrowed_ifu(ifu, 0.7 * u.um, 500.0)
        new = (out.wavelength_max - out.wavelength_min) / out.n_wave
        assert new.to_value(u.AA) == pytest.approx(native.to_value(u.AA), rel=1e-9)
        assert 0.5 * (out.wavelength_min + out.wavelength_max).to_value(u.um) == pytest.approx(0.7)
        assert out.n_wave < ifu.n_wave

    def test_window_is_about_twice_the_requested_half_width(self):
        ifu = make_ifu(wavelength_min=0.6 * u.um, wavelength_max=0.8 * u.um, n_wave=4000)
        out = narrowed_ifu(ifu, 0.7 * u.um, 500.0)
        width_kms = (out.wavelength_max - out.wavelength_min).to_value(u.um) / 0.7 * 299792.458
        assert width_kms == pytest.approx(1000.0, rel=0.05)


class TestForecast:
    def test_shapes_and_below_horizon_points_are_nan(self, night):
        fc = run(night)
        n_t, n_s = len(fc.times), 2
        assert fc.shear_snr.shape == (n_t, n_s)
        assert fc.sigma_g.shape == (n_t, n_s, 2)
        assert fc.effective_seeing.shape == (n_t, n_s)
        low = ~np.isfinite(fc.airmass) | (fc.airmass > 2.0)
        assert np.all(np.isnan(fc.shear_snr[low]))
        ok = ~low
        assert ok.any()
        assert np.all(np.isfinite(fc.shear_snr[ok & fc.detectable[:, 0]][:, 0]))

    def test_effective_seeing_follows_the_kolmogorov_scaling(self, night):
        fc = run(night)
        i = int(np.nanargmin(np.where(np.isfinite(fc.airmass), fc.airmass, np.inf)))
        lam = fc.observed_wavelength
        expected = 0.02 * fc.airmass[i] ** 0.6 * (lam.to_value(u.nm) / 500.0) ** -0.2
        assert fc.effective_seeing[i, 0].to_value(u.arcsec) == pytest.approx(expected)

    def test_worse_seeing_never_helps_at_a_given_time(self, night):
        fc = run(night, seeing_ref=[0.02, 0.1, 0.25] * u.arcsec)
        i = int(np.nanargmax(np.where(np.isfinite(fc.shear_snr[:, 0]), fc.shear_snr[:, 0], -1)))
        row = fc.shear_snr[i]
        finite = row[np.isfinite(row)]
        assert np.all(np.diff(finite) <= 1e-9)

    def test_brighter_target_gives_higher_snr(self, night):
        faint = run(night, target=make_target(flux=1e-15))
        bright = run(night, target=make_target(flux=5e-15))
        assert np.nanmax(bright.shear_snr) > np.nanmax(faint.shear_snr)
        assert np.nanmax(bright.line_snr) > np.nanmax(faint.line_snr)

    def test_extinction_lowers_the_signal(self, night):
        on = run(night, include_extinction=True)
        off = run(night, include_extinction=False)
        assert np.nanmax(off.line_snr) > np.nanmax(on.line_snr)

    def test_a_brighter_sky_lowers_the_snr(self, night):
        dark = run(night, library=flat_library(glow=5.0))
        bright = run(night, library=flat_library(glow=5000.0))
        assert np.nanmax(dark.line_snr) > np.nanmax(bright.line_snr)

    def test_line_outside_the_throughput_range_raises(self, night):
        ifu = make_ifu(throughput=Throughput(wavelength=[0.4, 0.5] * u.um, transmission=[0.3, 0.3]))
        with pytest.raises(ValueError, match="outside its tabulated coverage"):
            run(night, ifu=ifu)

    def test_bad_seeing_input_raises(self, night):
        with pytest.raises(ValueError, match="1-D"):
            run(night, seeing_ref=0.8 * u.arcsec)

    def test_target_that_never_rises_gives_all_nan(self, night):
        gal = DiscGalaxy(r_e=0.25, inc=55.0, pa=30.0, v_flat=180.0, z=Z, sigma0=25.0)
        target = ForecastTarget(
            SkyCoord("02h00m00s", "+80d00m00s"), gal, 2e-15 * u.erg / u.s / u.cm**2
        )
        fc = run(night, target=target)
        assert np.all(np.isnan(fc.shear_snr))
        assert fc.best_time_index(0) is None
        assert fc.night_summary()[0.02] is None

    def test_summary_and_best_time(self, night):
        fc = run(night)
        j = 0
        i = fc.best_time_index(j)
        assert i is not None
        assert fc.shear_snr[i, j] == np.nanmax(fc.shear_snr[:, j])
        best = fc.snr_vs_seeing()
        assert best.shape == (2,)
        assert best[0] >= best[1]
        summary = fc.night_summary()
        assert set(summary) == {0.02, 0.06}

    def test_does_not_mutate_the_instrument(self, night):
        ifu = make_ifu()
        before = dataclasses.replace(ifu)
        run(night, ifu=ifu)
        assert ifu == before


class TestRangeChecks:
    def test_line_beyond_the_sky_library_raises(self, night):
        gal = DiscGalaxy(r_e=0.25, inc=55.0, pa=30.0, v_flat=180.0, z=0.7, sigma0=25.0)
        target = ForecastTarget(
            SkyCoord("02h00m00s", "-25d00m00s"), gal, 2e-15 * u.erg / u.s / u.cm**2
        )
        # Halpha at z = 0.7 sits at 1.116 um, past a 1.0 um library
        ifu = make_ifu(throughput=Throughput(wavelength=[0.5, 1.3] * u.um, transmission=[0.3, 0.3]))
        with pytest.raises(ValueError, match="near-infrared is not supported"):
            forecast_ifu(
                ifu, target, night, flat_library(max_nm=1000.0), g=(0.05, 0.0),
                seeing_ref=[0.5] * u.arcsec, solar_flux=110.0,
            )  # fmt: skip

    def test_extinction_beyond_its_table_needs_the_flag_off(self, night):
        gal = DiscGalaxy(r_e=0.25, inc=55.0, pa=30.0, v_flat=180.0, z=0.7, sigma0=25.0)
        target = ForecastTarget(
            SkyCoord("02h00m00s", "-25d00m00s"), gal, 2e-15 * u.erg / u.s / u.cm**2
        )
        ifu = make_ifu(throughput=Throughput(wavelength=[0.5, 1.3] * u.um, transmission=[0.3, 0.3]))
        with pytest.raises(ValueError, match="include_extinction=False"):
            forecast_ifu(
                ifu, target, night, flat_library(max_nm=1300.0), g=(0.05, 0.0),
                seeing_ref=[0.5] * u.arcsec, solar_flux=110.0,
            )  # fmt: skip


class TestLowSnrFlag:
    def test_moment_estimator_flags_faint_targets_and_the_cube_estimator_never_does(self, night):
        from kino.analysis.fisher import LOW_SNR_WARNING

        bright = run(night, target=make_target(flux=1e-14), estimator="moments")
        assert np.nanmin(bright.median_spaxel_snr) > LOW_SNR_WARNING
        assert not np.any(bright.low_snr[bright.detectable])
        faint = run(night, target=make_target(flux=1.5e-16), estimator="moments")
        assert faint.detectable.any()
        assert np.any(faint.low_snr[faint.detectable])
        faint_cube = run(night, target=make_target(flux=1.5e-16))
        assert not faint_cube.low_snr.any()

    def test_low_snr_is_false_for_undetectable_points(self, night):
        fc = run(night, target=make_target(flux=1e-20))
        assert not fc.detectable.any()
        assert not fc.low_snr.any()


class TestErrorBudget:
    def test_intrinsic_floor_lowers_the_snr_and_zero_changes_nothing(self, night):
        stat = run(night)
        floor = run(night, intrinsic_sigma=0.03)
        zero = run(night, intrinsic_sigma=0.0)
        ok = np.isfinite(stat.shear_snr)
        assert np.all(floor.shear_snr[ok] < stat.shear_snr[ok])
        assert zero.shear_snr[ok] == pytest.approx(stat.shear_snr[ok])
        assert floor.shear_snr_stat[ok] == pytest.approx(stat.shear_snr[ok])

    def test_a_large_floor_caps_the_snr_at_g_over_sigma(self, night):
        fc = run(night, intrinsic_sigma=0.5)
        ok = np.isfinite(fc.shear_snr)
        assert np.all(fc.shear_snr[ok] < 0.05 / 0.5 + 1e-9)

    def test_psf_error_adds_a_bias_and_lowers_the_snr(self, night):
        base = run(night)
        with_psf = run(night, psf_error=0.2)
        ok = np.isfinite(base.shear_snr)
        assert np.all(np.isnan(base.psf_bias))
        assert np.all(np.isfinite(with_psf.psf_bias[ok]))
        assert np.all(with_psf.shear_snr[ok] <= base.shear_snr[ok] + 1e-9)
        assert np.any(np.abs(with_psf.psf_bias[ok]) > 0)

    def test_moments_and_cube_estimators_are_both_selectable(self, night):
        cube = run(night, estimator="cube")
        mom = run(night, estimator="moments")
        assert cube.meta["estimator"] == "cube"
        assert mom.meta["estimator"] == "moments"
        ok = np.isfinite(cube.shear_snr) & np.isfinite(mom.shear_snr)
        assert ok.any()
        assert np.all(cube.shear_snr[ok] > mom.shear_snr[ok])
